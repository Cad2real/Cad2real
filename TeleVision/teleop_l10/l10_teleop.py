#!/usr/bin/env python3
import sys
import os
import time
import threading
import numpy as np
import cv2
import argparse
import traceback
from collections import deque

# Add SDK path
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "linker_hand_python_sdk")
sys.path.append(target_dir)

from LinkerHand.linker_hand_api import LinkerHandApi
from teleop_hand import VuerTeleop, RealSense
# Import the unified HandRetarget class
from hand_retargeting import HandRetarget

class SingleHandController:
    """
    Manage the state and control of a single L10 dexterous hand.
    """
    def __init__(self, hand_type, can_interface, hand_joint="L10"):
        """
        Initialize a single hand controller.
        
        Args:
            hand_type (str): Type of hand ("left" or "right").
            can_interface (str): CAN bus interface for this hand (e.g. "can0").
            hand_joint (str): Hand joint type ("L10").
        """
        print(f"Initializing L10 dexterous hand: {hand_type.upper()} on {can_interface}...")
        self.hand_type = hand_type
        self.hand = LinkerHandApi(hand_joint=hand_joint, hand_type=hand_type, can=can_interface)
        self.hand_retarget = HandRetarget(hand_type=self.hand_type)
        
        # Set hand speed
        self.hand.set_speed(speed=[120, 250, 250, 250, 250])
        
        # Control variables
        self.last_hand_angles = None
        self.last_l10_pose = None
        
        # Buffer for smoothing control signals
        self.control_buffer = deque(maxlen=5)
        self.buffer_size = 3
        
        print(f"L10 dexterous hand {hand_type.upper()} controller initialized.")

    def convert_angles_to_l10_pose(self, hand_angles):
        """Convert computed hand angles to an L10 pose vector."""
        if len(hand_angles) < 10:
            print(f"Warning: {self.hand_type} hand angle data length is insufficient: {len(hand_angles)}.")
            return [255.0] * 10  # Return open state

        try:
            l10_pose = hand_angles[:10].tolist() if hasattr(hand_angles, 'tolist') else list(hand_angles[:10])
            l10_pose = [max(0.0, min(255.0, float(val))) for val in l10_pose]
        except Exception as e:
            print(f"{self.hand_type} hand angle conversion error: {e}")
            l10_pose = [255.0] * 10  # Return open state on error
            
        return l10_pose

    def process_landmarks(self, landmarks):
        """Process landmarks to generate and send control commands."""
        try:
            hand_angles = self.hand_retarget.solve_fingers_angles(landmarks)
            
            # Check for significant change to reduce redundant commands
            if self.last_hand_angles is None or not np.allclose(hand_angles, self.last_hand_angles, atol=2.0):
                self.last_hand_angles = hand_angles.copy()
                l10_pose = self.convert_angles_to_l10_pose(hand_angles)
                self.control_buffer.append(l10_pose)
                
                if len(self.control_buffer) >= self.buffer_size:
                    # Average the buffer for smoother control
                    averaged_pose = np.mean(list(self.control_buffer), axis=0).tolist()
                    self.send_l10_control(averaged_pose)
                    self.last_l10_pose = averaged_pose
                    

        except Exception as e:
            print(f"{self.hand_type} hand gesture processing error: {e}")

    def send_l10_control(self, l10_pose):
        """Send a control command to the L10 hand."""
        try:
            # Avoid sending duplicate commands if pose has not changed
            if self.last_l10_pose is not None and np.allclose(l10_pose, self.last_l10_pose, atol=1.0):
                return
            
            self.hand.finger_move(pose=l10_pose)
            print(f"Sent {self.hand_type.upper()} hand command: {l10_pose}")
        except Exception as e:
            print(f"{self.hand_type} hand L10 control command failed: {e}")

    def initialize_hand(self):
        """Initialize the hand to a fully open state."""
        print(f"Initializing {self.hand_type.upper()} hand to open state...")
        try:
            open_pose = [255.0] * 10
            self.hand.finger_move(pose=open_pose)
            time.sleep(2)  # Wait for hand to move into position
            print(f"{self.hand_type.upper()} hand initialization complete.")
        except Exception as e:
            print(f"{self.hand_type.upper()} hand initialization failed: {e}")

    def reset_hand(self):
        """Reset the hand to a fully open state on exit."""
        print(f"Resetting {self.hand_type.upper()} hand to open state...")
        try:
            open_pose = [255.0] * 10
            self.hand.finger_move(pose=open_pose)
            time.sleep(1)
        except Exception as e:
            print(f"Hand reset failed for {self.hand_type} hand: {e}")


class L10TeleopManager:
    """
    Manage the entire teleoperation system, supporting single- or dual-hand modes.
    """
    def __init__(self, mode, config_file='inspire_hand.yml', can_interfaces={"left": "can0", "right": "can1"}):
        """
        Initialize the teleoperation system.
        
        Args:
            mode (str): Operation mode ("left", "right", or "dual").
            config_file (str): Path to the VisionPro config file.
            can_interfaces (dict): Mapping of hand types to CAN interfaces.
        """
        print(f"Initializing L10 teleoperation system in {mode.upper()} mode...")
        self.mode = mode
        
        # Initialize shared resources
        print("Initializing VisionPro teleoperation system...")
        self.teleoperator = VuerTeleop(config_file)
        self.realsense = RealSense(self.teleoperator.resolution)
        
        # Control parameters
        self.control_freq = 20  # Control frequency (Hz)
        self.dt = 1.0 / self.control_freq
        
        # Create controller instances based on selected mode, with fault tolerance
        self.hands = {}
        if self.mode in ['left', 'dual']:
            try:
                self.hands['left'] = SingleHandController('left', can_interfaces['left'])
            except Exception as e:
                print(f"!!!!!!!!!! Failed to initialize left hand, ignoring it: {e} !!!!!!!!!!")

        if self.mode in ['right', 'dual']:
            try:
                self.hands['right'] = SingleHandController('right', can_interfaces['right'])
            except Exception as e:
                print(f"!!!!!!!!!! Failed to initialize right hand, ignoring it: {e} !!!!!!!!!!")

        # Check that at least one hand was initialized successfully
        if not self.hands:
            print("Error: all hands failed to initialize, cannot run.")
            raise RuntimeError("Unable to initialize any dexterous hand controller.")

        # Control flags
        self.is_running = False
        self.control_enabled = False
        
        print("L10 teleoperation system initialized.")

    def teleop_control_thread(self):
        """Main control loop thread for capturing data and controlling hands."""
        print("VisionPro teleoperation control thread started...")
        frame_count = 0
        start_control_frame = 30  # Wait 30 frames for stabilization

        try:
            while self.is_running:
                # Obtain camera image and send it to VisionPro
                color_image = self.realsense.get_frame()
                if color_image is not None:
                    if color_image.shape[:2] != self.teleoperator.resolution_cropped:
                        color_image = cv2.resize(color_image, (self.teleoperator.resolution_cropped[1], self.teleoperator.resolution_cropped[0]))
                    
                    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    np.copyto(self.teleoperator.img_array[:, :self.teleoperator.resolution_cropped[1]], color_image_rgb)
                    np.copyto(self.teleoperator.img_array[:, self.teleoperator.resolution_cropped[1]:], color_image_rgb)
                
                # Get landmark data from VisionPro
                # Fix: according to error logs, step() actually returns 5 values, not 8
                _, _, _, right_landmarks, left_landmarks = self.teleoperator.step()
                
                frame_count += 1
                
                # Start control after stabilization period
                if frame_count >= start_control_frame:
                    if not self.control_enabled:
                        print("Control enabled. Starting L10 dexterous hand control...")
                        self.control_enabled = True
                    
                    # Process landmarks for each active hand
                    if 'left' in self.hands:
                        self.hands['left'].process_landmarks(left_landmarks)
                    if 'right' in self.hands:
                        self.hands['right'].process_landmarks(right_landmarks)

                # Explicitly update the vuer session at the end of the loop to resolve image refresh issues
                if hasattr(self.teleoperator, 'update'):
                    self.teleoperator.update()
                
                time.sleep(self.dt)
                
        except Exception as e:
            print(f"Teleoperation control thread error: {e}")
            traceback.print_exc()
        finally:
            print("VisionPro teleoperation control thread ended.")

    def run(self):
        """Start the entire teleoperation system."""
        print("Running L10 teleoperation system...")
        
        # Initialize all active hands
        for hand in self.hands.values():
            hand.initialize_hand()
        
        # Start the control thread
        self.is_running = True
        teleop_thread = threading.Thread(target=self.teleop_control_thread)
        teleop_thread.daemon = True
        teleop_thread.start()
        
        print("System is running. Press Ctrl+C to exit.")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the teleoperation system smoothly."""
        print("Stopping L10 teleoperation system...")
        self.is_running = False
        time.sleep(0.5)  # Allow the thread to finish the current loop
        
        if self.realsense:
            self.realsense.stop()
        
        # Reset all active hands to open state
        for hand in self.hands.values():
            hand.reset_hand()
        
        print("L10 teleoperation system stopped.")

def main():
    """主函数，用于解析参数并运行遥操作。"""
    parser = argparse.ArgumentParser(description="基于Vision Pro的L10灵巧手遥操作。")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['left', 'right', 'dual'], 
        default='dual',
        help="设置遥操作模式: 'left'仅左手, 'right'仅右手, 或 'dual'双手。"
    )
    args = parser.parse_args()

    try:
        # 如果CAN接口与默认值不同，可以在此配置
        can_config = {"left": "can0", "right": "can1"}
        
        teleop_manager = L10TeleopManager(
            mode=args.mode,
            can_interfaces=can_config
        )
        teleop_manager.run()
        
    except Exception as e:
        print(f"L10遥操作系统出现严重错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()