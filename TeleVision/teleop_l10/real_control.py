#!/usr/bin/env python3
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

# TeleVision related imports
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

# Motion control related imports
from trans import compute_T_D_C
from motion_generation import MotionGenerator

# RealSense camera
import pyrealsense2 as rs

# L10 dexterous hand SDK
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "linker_hand_python_sdk")
sys.path.append(target_dir)
from LinkerHand.linker_hand_api import LinkerHandApi

# L10 hand retargeting
from hand_retargeting import HandRetarget

# Dobot robot arm control
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
joint_state = {
            'q_actual': [0.0] * 6,
            'qd_actual': [0.0] * 6,
            'i_actual': [0.0] * 6,
        }

globalLockValue = threading.Lock()

class CommandManager:
    """Real-time command manager - robot control"""
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
        """Get the command ready for execution"""
        with self.lock:
            if (self.command_counter % 3 == 0 and 
                self.latest_command is not None and 
                not np.array_equal(self.latest_command, self.last_sent_command)):
                self.last_sent_command = self.latest_command.copy() if self.latest_command is not None else None
                return self.latest_command
            return None

class L10CommandManager:
    """L10 dexterous hand command manager"""
    def __init__(self):
        self.latest_l10_pose = None
        self.lock = threading.Lock()
        self.pose_counter = 0
        self.last_sent_pose = None
        # Set smoothing buffer
        self.pose_buffer = deque(maxlen=3)
    
    def update_l10_pose(self, l10_pose):
        """Update latest L10 pose"""
        with self.lock:
            self.latest_l10_pose = l10_pose
            self.pose_counter += 1
            self.pose_buffer.append(l10_pose)
    
    def get_l10_pose_for_execution(self):
        """Get L10 pose for execution"""
        with self.lock:
            if (len(self.pose_buffer) >= 2 and 
                self.latest_l10_pose is not None and 
                (self.last_sent_pose is None or 
                 not np.allclose(self.latest_l10_pose, self.last_sent_pose, atol=2.0))):
                
                # Smooth by averaging the buffer
                averaged_pose = np.mean(list(self.pose_buffer), axis=0).tolist()
                self.last_sent_pose = averaged_pose.copy()
                return averaged_pose
            return None

class IntegratedTeleop:
    """Integrated teleoperation control system - based on TeleVision"""
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
        self.image_queue = Queue()
        self.toggle_streaming = Event()
        
        # Initialize TeleVision
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, self.image_queue, self.toggle_streaming)
        self.processor = VuerPreprocessor()

        # Load URDF configuration
        RetargetingConfig.set_default_urdf_dir('/home/b600/wl/TeleVision/assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.right_retargeting = right_retargeting_config.build()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            print("清理TeleVision...")
            self.toggle_streaming.set()
            time.sleep(0.1)
            
            if hasattr(self.tv, 'stop'):
                self.tv.stop()
            
            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()
                print("共享内存已清理")
        except Exception as e:
            print(f"TeleVision清理错误: {e}")

    def step(self):
        """Process a single frame"""
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        right_landmarks = self.tv.right_landmarks.copy()

        # Compute right wrist pose
        right_pose_w = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])])
        
        # Compute right hand finger pose
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])

        return right_pose_w, right_landmarks, right_qpos

class RealSense:
    """RealSense camera control"""
    def __init__(self, resolution=(1080, 1920)):
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
    """Connect to the robot arm"""
    try:
        ip = "192.168.5.1"
        dashboard = DobotApiDashboard(ip, 29999)
        move = DobotApiMove(ip, 30003)
        feed_four = DobotApiFeedBack(ip, 30004)
        print("机械臂连接成功")
        return dashboard, move, feed_four
    except Exception as e:
        print(f"机械臂连接失败: {e}")
        raise

def connect_l10_hand(hand_type="right", hand_joint="L10", can_interface="can0"):
    """Connect to the L10 dexterous hand"""
    try:
        hand = LinkerHandApi(hand_joint=hand_joint, hand_type=hand_type, can=can_interface)
        hand.set_speed(speed=[120, 250, 250, 250, 250])
        print(f"L10灵巧手连接成功: {hand_type}, {hand_joint}, {can_interface}")
        return hand
    except Exception as e:
        print(f"L10灵巧手连接失败: {e}")
        raise

def get_feed(feed_four):
    """Robot arm state monitoring"""
    global joint_state,current_actual, algorithm_queue, enableStatus_robot, robotErrorState, robotMode
    while True:
        with globalLockValue:
            feed_info = feed_four.feedBackData()
            if hex((feed_info['test_value'][0])) == '0x123456789abcdef':
                robotMode = feed_info['robot_mode'][0]
                current_actual = feed_info["tool_vector_actual"][0]
                algorithm_queue = feed_info['run_queued_cmd'][0]
                enableStatus_robot = feed_info['enable_status'][0]
                robotErrorState = feed_info['error_status'][0]

                joint_state['q_actual'] = feed_info["q_actual"][0]
                joint_state['qd_actual'] = feed_info["qd_actual"][0]
                joint_state['i_actual'] = feed_info["i_actual"][0]
        time.sleep(0.001)


def SaveLog(robot_dir, hand, hand_dir):
        seq = 0   # Simulated increasing seq
        with open(robot_dir, "w", encoding="utf-8") as f_r:
            with open(hand_dir, "w", encoding="utf-8") as f_h:
                while True:
                    time.sleep(0.001)
                    with globalLockValue:
                        joint_values = joint_state['q_actual']
                        try:
                            hand_joint = hand.get_state()
                        except Exception as e:
                            # Prevent thread termination by exception
                            f_h.write(
                                f"seq: {seq}\n",
                                f"error: {repr(e)}\n\n"
                            )
                            f_h.flush()
                            time.sleep(0.1)

                        t = time.time()
                        secs = int(t)
                        nsecs = int((t - secs) * 1e9)

                        if joint_values is not None and len(joint_values) == 6:
                            # Format joint values
                            formatted_values = [float(f"{val:.8f}") for val in joint_values]
                            
                            # Construct ROS-like YAML format
                            log_line = (
                                f"seq: {seq}\n"
                                f"secs: {secs}\n"
                                f"nsecs: {nsecs}\n"
                                f"position: {formatted_values}\n\n"
                            )
                            f_r.write(log_line)
                            f_r.flush()
                        else:
                            f_r.write(f"header:\n  seq: {seq}\n  无效关节值: {joint_values}\n\n")
                            f_r.flush()

                        if hand_joint is not None and len(hand_joint) == 10:
                            # Format joint values
                            formatted_values = [float(f"{val:.8f}") for val in hand_joint]

                            # Construct ROS-like YAML format
                            log_line = (
                                f"seq: {seq}\n"
                                f"secs: {secs}\n"
                                f"nsecs: {nsecs}\n"
                                f"position: {formatted_values}\n\n"
                            )

                            f_h.write(log_line)
                            f_h.flush()
                        else:
                            f_h.write(f"header:\n  seq: {seq}\n  无效关节值: {joint_values}\n\n")
                            f_h.flush()

                        seq += 1

def clear_robot_error(dashboard):
    """Clear robot arm errors"""
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()
    
    while True:
        with globalLockValue:
            if robotErrorState:
                numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                numbers = [int(num) for num in numbers]
                if numbers[0] == 0 and len(numbers) > 1:
                    for i in numbers[1:]:
                        alarmState = False
                        if i == -2:
                            print("机器告警: 机器碰撞", i)
                            alarmState = True
                        if alarmState:
                            continue
                        for item in dataController:
                            if i == item["id"]:
                                print("机器告警 Controller errorid", i, item["zh_CN"]["description"])
                                alarmState = True
                                break
                        if alarmState:
                            continue
                        for item in dataServo:
                            if i == item["id"]:
                                print("机器告警 Servo errorid", i, item["zh_CN"]["description"])
                                break
                    
                    # Automatically clear errors
                    dashboard.ClearError()
                    time.sleep(0.01)
                    dashboard.Continue()
            else:
                if enableStatus_robot == 1 and algorithm_queue == 0:
                    dashboard.Continue()
        time.sleep(5)

def convert_angles_to_l10_pose(hand_angles):
    """
    Convert hand angles to L10 control pose
    """
    if len(hand_angles) < 10:
        print(f"警告: 手部角度数据长度不足 {len(hand_angles)}")
        return [255.0] * 10  # Return open state
    
    try:
        # Use hand_angles directly, since HandRetarget has already handled ordering
        l10_pose = hand_angles[:10].tolist() if hasattr(hand_angles, 'tolist') else list(hand_angles[:10])
        
        # Ensure all values are within valid range [0, 255]
        l10_pose = [max(0.0, min(255.0, float(val))) for val in l10_pose]
        
    except Exception as e:
        print(f"角度转换错误: {e}")
        l10_pose = [255.0] * 10  # Return open state on error
        
    return l10_pose

def send_robot_command(move, joint_angles):
    """Send joint angle command to the robot arm"""
    if len(joint_angles) < 6:
        print(f"错误: 关节角度数量不足 ({len(joint_angles)}/6)")
        return False
    
    try:
        move.JointMovJ(*joint_angles[:6])
        return True
    except Exception as e:
        print(f"发送机械臂命令失败: {e}")
        return False

def send_l10_command(hand, l10_pose):
    """Send control command to L10 dexterous hand"""
    try:
        hand.finger_move(pose=l10_pose)
        return True
    except Exception as e:
        print(f"L10控制指令发送失败: {e}")
        return False

def robot_control_loop(move, command_manager, stop_event):
    """Robot arm control thread"""
    print("机械臂控制线程启动")
    
    while not stop_event.is_set():
        try:
            new_command = command_manager.get_command_for_execution()
            
            if new_command is not None:
                if send_robot_command(move, new_command):
                    print(f"机械臂执行: {[f'{x:.2f}' for x in new_command[:6]]}")
            
            time.sleep(0.02)  # 50Hz check rate
        except Exception as e:
            print(f"机械臂控制线程错误: {e}")
            if stop_event.is_set():
                break
            time.sleep(0.1)
    
    print("机械臂控制线程退出")

def l10_control_loop(hand, l10_command_manager, hand_retarget, stop_event):
    """L10 dexterous hand control thread"""
    print("L10灵巧手控制线程启动")
    
    while not stop_event.is_set():
        try:
            l10_pose = l10_command_manager.get_l10_pose_for_execution()
            
            if l10_pose is not None:
                if send_l10_command(hand, l10_pose):
                    print(f"L10执行: {[round(x, 1) for x in l10_pose]}")
            
            time.sleep(0.05)  # 20Hz check rate
        except Exception as e:
            print(f"L10控制线程错误: {e}")
            if stop_event.is_set():
                break
            time.sleep(0.1)
    
    print("L10控制线程退出")

def data_processing_loop(teleoperator, realsense, command_manager, l10_command_manager,
                        motion_generator, hand_retarget, stop_event):
    """Data processing main loop"""
    print("数据处理线程启动")
    
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
            
            # Get teleoperation data
            right_pose_w, right_landmarks, right_qpos = teleoperator.step()
            
            frame_count += 1
            
            # Wait for initial frames to complete
            if frame_count >= 100 and not start_control:
                start_control = True
                print("开始综合控制 - 机械臂 + L10灵巧手")
            
            if start_control:
                # Handle robot arm control
                if last_right_pose_w is None or not np.allclose(right_pose_w, last_right_pose_w, atol=3e-3):
                    T_D_C_result = compute_T_D_C(right_pose_w)
                    new_joint_deg = motion_generator.plan_motion(last_joint_deg, T_D_C_result)
                    
                    if new_joint_deg is not None:
                        command_manager.update_command(new_joint_deg)
                        last_joint_deg = new_joint_deg
                    
                    last_right_pose_w = right_pose_w.copy()
                
                # Handle L10 dexterous hand control
                try:
                    hand_angles = hand_retarget.solve_fingers_angles(right_landmarks)
                    
                    # Check for hand angle changes
                    if (last_hand_angles is None or 
                        not np.allclose(hand_angles, last_hand_angles, atol=3.0)):
                        
                        last_hand_angles = hand_angles.copy()
                        
                        # Convert to L10 control pose
                        l10_pose = convert_angles_to_l10_pose(hand_angles)
                        l10_command_manager.update_l10_pose(l10_pose)
                        
                except Exception as e:
                    print(f"手势处理错误: {e}")
            
            # Maintain 30Hz data processing rate
            time.sleep(0.033)
            
        except Exception as e:
            print(f"数据处理错误: {e}")
            if stop_event.is_set():
                break
            time.sleep(0.1)
    
    print("数据处理线程退出")

def initialize_l10_hand(hand):
    """Initialize L10 dexterous hand to open state"""
    print("正在初始化L10灵巧手到张开状态...")
    try:
        # Open pose
        open_pose = [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
        hand.finger_move(pose=open_pose)
        time.sleep(2)  # Wait for hand movement to complete
        print("L10灵巧手初始化完成")
    except Exception as e:
        print(f"L10灵巧手初始化失败: {e}")

def disconnect_robot(dashboard, move, feed_four):
    """Disconnect robot arm"""
    try:
        print("断开机械臂连接...")
        dashboard.DisableRobot()
        time.sleep(0.1)
        print("机械臂连接已断开")
    except Exception as e:
        print(f"断开机械臂连接时出错: {e}")

def main():
    """Main function"""
    print("初始化综合控制系统 - 机械臂 + L10灵巧手...")
    
    # Initialize device variables
    dashboard = None
    move = None
    feed_four = None
    hand = None
    realsense = None
    teleoperator = None
    threads = []
    
    try:
        # Connect all devices
        print("连接机械臂...")
        dashboard, move, feed_four = connect_robot()
        
        print("连接L10灵巧手...")
        hand = connect_l10_hand(hand_type="right", hand_joint="L10", can_interface="can1")
        
        # Start robot state monitoring thread
        feed_thread = threading.Thread(target=get_feed, args=(feed_four,))
        feed_thread.daemon = True
        feed_thread.start()
        threads.append(feed_thread)
        
        error_thread = threading.Thread(target=clear_robot_error, args=(dashboard,))
        error_thread.daemon = True
        error_thread.start()
        threads.append(error_thread)

        log_thread = threading.Thread(target=SaveLog, args=('./robot.txt', hand, './hand.txt'))
        log_thread.daemon = True
        log_thread.start()
        threads.append(log_thread)
        
        # Enable robot arm
        print("使能机械臂...")
        dashboard.ClearError()
        dashboard.EnableRobot()
        dashboard.SpeedFactor(80)  # 80% speed
        dashboard.AccJ(80)         # 80% acceleration
        time.sleep(1)
        
        # Initialize L10 dexterous hand
        initialize_l10_hand(hand)
        
        # Initialize system components
        print("初始化控制组件...")
        teleoperator = IntegratedTeleop('/home/b600/wl/TeleVision/teleop/inspire_hand.yml')
        motion_generator = MotionGenerator()
        hand_retarget = HandRetarget()
        realsense = RealSense(teleoperator.resolution)
        
        # Create command managers
        command_manager = CommandManager()
        l10_command_manager = L10CommandManager()
        stop_event = threading.Event()
        
        # Start all control threads
        # Robot arm control thread
        robot_thread = threading.Thread(target=robot_control_loop, 
                                      args=(move, command_manager, stop_event))
        robot_thread.daemon = True
        robot_thread.start()
        threads.append(robot_thread)
        
        # L10 dexterous hand control thread
        l10_thread = threading.Thread(target=l10_control_loop,
                                     args=(hand, l10_command_manager, hand_retarget, stop_event))
        l10_thread.daemon = True
        l10_thread.start()
        threads.append(l10_thread)
        
        # Data processing thread
        data_thread = threading.Thread(target=data_processing_loop,
                                      args=(teleoperator, realsense, command_manager, 
                                           l10_command_manager, motion_generator, 
                                           hand_retarget, stop_event))
        data_thread.daemon = True
        data_thread.start()
        threads.append(data_thread)
        
        print("=" * 60)
        print("综合控制系统启动完成！")
        print("功能:")
        print("  • 手腕动作 → 控制机械臂运动")
        print("  • 手指动作 → 控制L10灵巧手")
        print("  • VisionPro遥操作 → 实时双重控制")
        print("=" * 60)
        print("按Ctrl+C停止控制...")
        
        # Main thread keep alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到停止信号，正在退出...")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal stop event
        print("设置停止信号...")
        try:
            stop_event.set()
        except:
            pass
        
        # Wait for threads to finish
        print("等待线程结束...")
        for thread in threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        print(f"线程 {thread.name} 未能正常结束")
            except Exception as e:
                print(f"等待线程结束时出错: {e}")
        
        # Cleanup device resources
        print("清理设备资源...")
        
        # 1. Cleanup RealSense
        if realsense:
            try:
                realsense.stop()
                print("RealSense已停止")
            except Exception as e:
                print(f"停止RealSense时出错: {e}")
        
        # 2. Cleanup L10 dexterous hand (reset to open state)
        if hand:
            try:
                print("将L10灵巧手复位到张开状态...")
                open_pose = [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
                hand.finger_move(pose=open_pose)
                time.sleep(1)
                print("L10灵巧手已复位")
            except Exception as e:
                print(f"L10灵巧手复位失败: {e}")
        
        # 3. Cleanup robot arm
        if dashboard and move and feed_four:
            try:
                disconnect_robot(dashboard, move, feed_four)
            except Exception as e:
                print(f"断开机械臂时出错: {e}")
        
        # 4. Cleanup TeleVision
        if teleoperator:
            try:
                teleoperator.cleanup()
            except Exception as e:
                print(f"清理TeleVision时出错: {e}")
        
        print("程序退出完成")
        
        # Force exit if cleanup did not finish
