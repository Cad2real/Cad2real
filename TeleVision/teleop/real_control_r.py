import numpy as np
import cv2
import threading
import time
import sys
import os
import re
import queue
from collections import deque
from pathlib import Path
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
from trans import compute_T_D_C
from motion_generation import MotionGenerator
from hand_retargeting import HandRetarget
import pyrealsense2 as rs

# Add Dobot control path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dobot_control', 'robot'))
from dobot_control.robot.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, DobotApiFeedBack, MyType, alarmAlarmJsonFile

# Global variables (for Dobot state monitoring)
current_actual = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
robotMode = 0
globalLockValue = threading.Lock()

class CommandManager:
    """Real-time command manager"""
    def __init__(self):
        self.latest_command = None
        self.lock = threading.Lock()
        self.command_counter = 0
        self.last_executed_command = None
    
    def update_command(self, command):
        """Update the latest command."""
        with self.lock:
            self.latest_command = command
            self.command_counter += 1
    
    def get_command_for_execution(self):
        """Get the next command to execute (one every 10 commands)."""
        with self.lock:
            # Execute one every 10 commands
            if self.command_counter % 10 == 0 and self.latest_command is not None:
                return self.latest_command
            return None
    
    def get_immediate_command(self):
        """Get an immediate command (used to interrupt current motion)."""
        with self.lock:
            return self.latest_command
        
class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # Create shared memory
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        # Initialize Vuer
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        # Load URDF configuration
        RetargetingConfig.set_default_urdf_dir('/home/b600/wl/TeleVision/assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        right_landmarks = self.tv.right_landmarks.copy()

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose_w = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos, right_pose_w, right_landmarks

class RealSense:
    def __init__(self, resolution=(720, 1280)):
        self.resolution = resolution
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, resolution[1], resolution[0], rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())
    
    def stop(self):
        self.pipeline.stop()

def connect_robot():
    try:
        ip = "192.168.5.1"
        dashboard = DobotApiDashboard(ip, 29999)
        move = DobotApiMove(ip, 30003)
        feed = DobotApi(ip, 30004)
        feed_four = DobotApiFeedBack(ip, 30004)
        print("Robot connected successfully")
        return dashboard, move, feed_four
    except Exception as e:
        print(f"Robot connection failed: {e}")
        raise

def get_feed(feed_four):
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState, robotMode
    while True:
        with globalLockValue:
            feed_info = feed_four.feedBackData()
            if hex((feed_info['test_value'][0])) == '0x123456789abcdef':
                robotMode = feed_info['robot_mode'][0]
                current_actual = feed_info["tool_vector_actual"][0]
                algorithm_queue = feed_info['run_queued_cmd'][0]
                enableStatus_robot = feed_info['enable_status'][0]
                robotErrorState = feed_info['error_status'][0]
        time.sleep(0.001)

def clear_robot_error(dashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()
    while True:
        with globalLockValue:
            if robotErrorState:
                numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                numbers = [int(num) for num in numbers]
                if numbers[0] == 0 and len(numbers) > 1:
                    for i in numbers[1:]:
                        # Error handling logic
                        if i == -2:
                            print(f"Robot alert: collision detected ({i})")
                        else:
                            # Other error handling
                            pass
                    dashboard.ClearError()
                    time.sleep(0.01)
                    dashboard.Continue()
            else:
                if enableStatus_robot == 1 and algorithm_queue == 0:
                    dashboard.Continue()
        time.sleep(5)

def send_robot_command(move, joint_angles, is_interrupt=False):
    """
    Send a joint angle command to the robot arm.
    :param is_interrupt: whether this is an interrupt command (execute immediately)
    """
    if len(joint_angles) < 6:
        print(f"Error: insufficient joint angles ({len(joint_angles)}/6)")
        return False
    
    try:
        if is_interrupt:
            # Interrupt current motion and execute the new command immediately
            move.JointMovJ(*joint_angles[:6], isQueued=0)
        else:
            # Execute normally in queue
            move.JointMovJ(*joint_angles[:6])
        return True
    except Exception as e:
        print(f"Failed to send command: {e}")
        return False

def robot_control_loop(move, command_manager, stop_event):
    """Robot control thread - optimized version."""
    print("Robot control thread started")
    last_command = None
    is_moving = False
    
    while not stop_event.is_set():
        # Check for an immediate interrupt command
        interrupt_command = command_manager.get_immediate_command()
        
        # If a new command arrives while the robot is moving, interrupt the current motion
        if interrupt_command is not None and interrupt_command != last_command and is_moving:
            print(f"Interrupting current motion and executing new command: {interrupt_command[:6]}")
            if send_robot_command(move, interrupt_command, is_interrupt=True):
                last_command = interrupt_command
                is_moving = True
        
        # Check for a periodic execution command
        exec_command = command_manager.get_command_for_execution()
        
        if exec_command is not None and exec_command != last_command:
            print(f"Executing periodic command: {exec_command[:6]}")
            if send_robot_command(move, exec_command):
                last_command = exec_command
                is_moving = True
        
        # Update motion status
        with globalLockValue:
            # algorithm_queue = 0 means no commands are running
            is_moving = (algorithm_queue > 0)
        
        # Short sleep to avoid high CPU usage
        time.sleep(0.001)
    
    print("Robot control thread exited")

def main():
    # Connect to robot arm
    dashboard, move, feed_four = connect_robot()
    
    # Start status monitoring threads
    feed_thread = threading.Thread(target=get_feed, args=(feed_four,))
    feed_thread.daemon = True
    feed_thread.start()
    
    error_thread = threading.Thread(target=clear_robot_error, args=(dashboard,))
    error_thread.daemon = True
    error_thread.start()
    
    # Enable robot arm
    print("Enabling robot arm...")
    dashboard.EnableRobot()
    # Set higher motion speed
    dashboard.SpeedFactor(80)  # 80% speed
    dashboard.AccJ(80)         # 80% acceleration
    time.sleep(1)  # Wait for enable to complete
    
    # Initialize gesture recognition system
    teleoperator = VuerTeleop('/home/b600/wl/TeleVision/teleop/inspire_hand.yml')
    motion_generator = MotionGenerator()
    realsense = RealSense(teleoperator.resolution)
    
    # Create command manager
    command_manager = CommandManager()
    stop_event = threading.Event()
    
    # Start robot control thread
    robot_thread = threading.Thread(target=robot_control_loop, 
                                  args=(move, command_manager, stop_event))
    robot_thread.daemon = True
    robot_thread.start()
    
    # Initialize state variables
    last_joint_deg = [0.0] * 6
    last_right_pose_w = None
    frame_count = 0
    start_control = False
    
    try:
        print("Starting real-time control, press Ctrl+C to stop...")
        while True:
            # Get RealSense image
            color_image = realsense.get_frame()
            if color_image is not None:
                # Process image and update shared memory
                if color_image.shape[:2] != teleoperator.resolution_cropped:
                    color_image = cv2.resize(color_image, 
                                            (teleoperator.resolution_cropped[1], 
                                             teleoperator.resolution_cropped[0]))
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                np.copyto(teleoperator.img_array[:, :teleoperator.resolution_cropped[1]], color_image_rgb)
                np.copyto(teleoperator.img_array[:, teleoperator.resolution_cropped[1]:], color_image_rgb)
            
            # Get gesture data
            _, _, _, _, _, right_pose_w, right_landmarks = teleoperator.step()
            
            frame_count += 1
            
            # Wait for initial frames to complete
            if frame_count >= 300 and not start_control:  # Reduce initial wait frame count
                start_control = True
                print("Starting real-time control")
            
            if start_control:
                # Generate new commands only when a significant change is detected
                if last_right_pose_w is None or not np.allclose(right_pose_w, last_right_pose_w, atol=1e-2):  # Increase tolerance
                    T_D_C_result = compute_T_D_C(right_pose_w)
                    new_joint_deg = motion_generator.plan_motion(last_joint_deg, T_D_C_result)
                    
                    if new_joint_deg is not None:
                        # Update command manager
                        command_manager.update_command(new_joint_deg)
                        last_joint_deg = new_joint_deg
                        # Reduce output frequency
                        if frame_count % 30 == 0:
                            print(f"Updated command: {[f'{x:.2f}' for x in new_joint_deg[:6]]}")
                    
                    last_right_pose_w = right_pose_w.copy()
            
            # Maintain 30Hz processing frequency
            time.sleep(0.033)  # about 30Hz
    
    except KeyboardInterrupt:
        print("Stopping control...")
    finally:
        # Clean up resources
        print("Releasing resources...")
        robot_thread.join(timeout=1.0)
        realsense.stop()
        dashboard.DisableRobot()
        teleoperator.shm.close()
        teleoperator.shm.unlink()

if __name__ == '__main__':
    main()