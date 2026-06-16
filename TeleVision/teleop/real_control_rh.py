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
sys.path.append(os.path.join(project_root, 'dobot_control', 'hand'))
from dobot_control.robot.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, DobotApiFeedBack, MyType, alarmAlarmJsonFile
from dobot_control.hand.hand_control import hand_control

# Global variables (for Dobot state monitoring)
current_actual = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
robotMode = 0
globalLockValue = threading.Lock()

class CommandManager:
    """Real-time command manager - optimized version"""
    def __init__(self):
        self.latest_command = None
        self.lock = threading.Lock()
        self.command_counter = 0
        self.last_sent_command = None
    
    def update_command(self, command):
        """Update the latest command"""
        with self.lock:
            self.latest_command = command
            self.command_counter += 1
    
    def get_command_for_execution(self):
        """Get the command for execution - lower frequency while keeping responsiveness"""
        with self.lock:
            # Execute one of every 5 commands to improve responsiveness
            if (self.command_counter % 5 == 0 and 
                self.latest_command is not None and 
                not np.array_equal(self.latest_command, self.last_sent_command)):
                self.last_sent_command = self.latest_command.copy() if self.latest_command is not None else None
                return self.latest_command
            return None
    
    def get_immediate_command(self):
        """Get the command to execute immediately."""
        with self.lock:
            if (self.latest_command is not None and 
                not np.array_equal(self.latest_command, self.last_sent_command)):
                return self.latest_command
            return None

class HandCommandManager:
    """Dexterous hand command manager - high response version"""
    def __init__(self):
        self.latest_hand_angles = None
        self.lock = threading.Lock()
        self.command_counter = 0
        self.last_sent_angles = None
    
    def update_hand_angles(self, hand_angles):
        """Update the latest hand angles."""
        with self.lock:
            self.latest_hand_angles = hand_angles
            self.command_counter += 1
    
    def get_hand_angles_for_execution(self):
        """Get the hand angles ready for execution - high frequency."""
        with self.lock:
            # Execute one of every 2 commands to improve response frequency
            if (self.command_counter % 2 == 0 and 
                self.latest_hand_angles is not None and
                (self.last_sent_angles is None or 
                 not np.allclose(self.latest_hand_angles, self.last_sent_angles, atol=5.0))):  # Increase tolerance to avoid jitter
                self.last_sent_angles = self.latest_hand_angles.copy()
                return self.latest_hand_angles
            return None
        
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

def connect_hand():
    """Connect to the dexterous hand - optimized version."""
    try:
        hand = hand_control(password=3)
        hand.service_start()
        print("Hand connected successfully")
        return hand
    except Exception as e:
        print(f"Hand connection failed: {e}")
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
                        if i == -2:
                            print(f"Robot alert: collision detected ({i})")
                        else:
                            pass
                    dashboard.ClearError()
                    time.sleep(0.01)
                    dashboard.Continue()
            else:
                if enableStatus_robot == 1 and algorithm_queue == 0:
                    dashboard.Continue()
        time.sleep(5)

def send_robot_command(move, joint_angles, is_interrupt=False):
    """Send a joint angle command to the robot arm."""
    if len(joint_angles) < 6:
        print(f"Error: insufficient joint angles ({len(joint_angles)}/6)")
        return False
    
    try:
        if is_interrupt:
            move.JointMovJ(*joint_angles[:6], isQueued=0)
        else:
            move.JointMovJ(*joint_angles[:6])
        return True
    except Exception as e:
        print(f"Failed to send robot command: {e}")
        return False

def robot_control_loop(move, command_manager, stop_event):
    """Robot arm control thread - independent high-frequency version."""
    print("Robot control thread started")
    last_command = None
    
    while not stop_event.is_set():
        # Check for new command
        new_command = command_manager.get_command_for_execution()
        
        if new_command is not None:
            if send_robot_command(move, new_command):
                last_command = new_command
                print(f"Robot executing: {[f'{x:.2f}' for x in new_command[:6]]}")
        
        # Short sleep to keep high responsiveness
        time.sleep(0.01)  # 100Hz check frequency
    
    print("Robot control thread exited")

def hand_control_loop(hand, hand_command_manager, stop_event):
    """Dexterous hand control thread - timeout fix version."""
    print("Hand control thread started")
    
    # Test connection
    if not hand.test_connection():
        print("Hand connection test failed, please check ROS service status")
        return
    
    command_count = 0
    successful_commands = 0
    
    while not stop_event.is_set():
        # Get hand angles ready for execution
        hand_angles = hand_command_manager.get_hand_angles_for_execution()
        
        if hand_angles is not None:
            command_count += 1
            
            # Use optimized non-blocking call
            try:
                hand.angle_move(np.array([hand_angles]))
                successful_commands += 1
                print(f"Hand executing: {[f'{x:.1f}' for x in hand_angles]} [{successful_commands}/{command_count}]")
            except Exception as e:
                print(f"Hand execution error: {e}")
        
        # Sleep to balance response speed and system load
        time.sleep(0.02)  # 50Hz check frequency, lower system load
    
    print(f"Hand control thread exited - success rate: {successful_commands}/{command_count}")


def data_processing_loop(teleoperator, realsense, command_manager, hand_command_manager, 
                        motion_generator, hand_retarget, stop_event):
    """Data processing thread - independent data acquisition and processing."""
    print("Data processing thread started")
    
    last_joint_deg = [0.0] * 6
    last_right_pose_w = None
    last_hand_angles = None
    frame_count = 0
    start_control = False
    
    while not stop_event.is_set():
        try:
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
            if frame_count >= 100 and not start_control:  # Reduce initial wait time
                start_control = True
                print("Starting real-time control")
            
            if start_control:
                # Process robot arm and hand data in parallel
                
                # Handle robot arm control - lower detection threshold to improve response
                if last_right_pose_w is None or not np.allclose(right_pose_w, last_right_pose_w, atol=5e-3):
                    T_D_C_result = compute_T_D_C(right_pose_w)
                    new_joint_deg = motion_generator.plan_motion(last_joint_deg, T_D_C_result)
                    
                    if new_joint_deg is not None:
                        command_manager.update_command(new_joint_deg)
                        last_joint_deg = new_joint_deg
                    
                    last_right_pose_w = right_pose_w.copy()
                
                # Handle hand control - process independently, not relying on robot arm
                hand_angles = hand_retarget.solve_fingers_angles(right_landmarks)
                
                if hand_angles is not None:
                    if last_hand_angles is None or not np.allclose(hand_angles, last_hand_angles, atol=2.0):
                        hand_command_manager.update_hand_angles(hand_angles)
                        last_hand_angles = hand_angles.copy()
            
            # Maintain 50Hz data processing frequency
            time.sleep(0.02)
            
        except Exception as e:
            print(f"Data processing error: {e}")
            time.sleep(0.01)  # Brief sleep after error
    
    print("Data processing thread exited")

def main():
    # Connect devices
    dashboard, move, feed_four = connect_robot()
    hand = connect_hand()
    
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
    dashboard.SpeedFactor(90)  # Increase speed to 90%
    dashboard.AccJ(90)         # Increase acceleration to 90%
    time.sleep(1)
    
    # Initialize system components
    teleoperator = VuerTeleop('/home/b600/wl/TeleVision/teleop/inspire_hand.yml')
    motion_generator = MotionGenerator()
    hand_retarget = HandRetarget()
    realsense = RealSense(teleoperator.resolution)
    
    # Create command managers
    command_manager = CommandManager()
    hand_command_manager = HandCommandManager()
    stop_event = threading.Event()
    
    # Start all control threads - run fully independently
    threads = []
    
    # Robot control thread
    robot_thread = threading.Thread(target=robot_control_loop, 
                                  args=(move, command_manager, stop_event))
    robot_thread.daemon = True
    robot_thread.start()
    threads.append(robot_thread)
    
    # Hand control thread
    hand_thread = threading.Thread(target=hand_control_loop,
                                  args=(hand, hand_command_manager, stop_event))
    hand_thread.daemon = True
    hand_thread.start()
    threads.append(hand_thread)
    
    # Data processing thread
    data_thread = threading.Thread(target=data_processing_loop,
                                  args=(teleoperator, realsense, command_manager, 
                                       hand_command_manager, motion_generator, 
                                       hand_retarget, stop_event))
    data_thread.daemon = True
    data_thread.start()
    threads.append(data_thread)
    
    try:
        print("System startup complete, beginning real-time control...")
        print("Robot arm and hand will run independently in parallel")
        print("Press Ctrl+C to stop control...")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping control...")
    finally:
        # Clean up resources
        print("Releasing resources...")
        stop_event.set()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=2.0)
        
        realsense.stop()
        dashboard.DisableRobot()
        hand.service_stop()
        teleoperator.shm.close()
        teleoperator.shm.unlink()
        print("Resource release complete")

if __name__ == '__main__':
    main()