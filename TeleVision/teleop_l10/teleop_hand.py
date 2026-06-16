import numpy as np
import cv2

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import yaml
from multiprocessing import shared_memory, Queue,  Event 

from trans import compute_T_D_C
from motion_generation import MotionGenerator
import pyrealsense2 as rs

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # create shared memory space
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue() # for information communication
        toggle_streaming = Event() # for signal synchronization 
        # start vuer
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        # load URDF
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
        left_landmarks = self.tv.left_landmarks.copy()

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose_w=np.concatenate([right_wrist_mat[:3, 3] + np.array([0.2, 0.15, 0.55]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])

        return left_pose, right_pose, right_pose_w, right_landmarks, left_landmarks

class RealSense:
    def __init__(self, resolution=(1080, 1920)):
        self.resolution = resolution
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable color stream
        config.enable_stream(rs.stream.color, resolution[1], resolution[0], rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError("pyrealsense2 is not installed. Please install it with 'pip install pyrealsense2'")
        
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    
    def stop(self):
        self.pipeline.stop()

if __name__ == '__main__':
    print("初始化...")
    teleoperator = VuerTeleop('inspire_hand.yml')
    motion_generator = MotionGenerator()
    realsense = RealSense(teleoperator.resolution)
    
    key = 50
    i = 0
    t = 1000
    write = 1
    start = False
    last_right_pose_w = None
    last_joint_deg = [0.0] * 6
    last_hand_angles = None

    try:
        while True:
            # Get frame from RealSense camera
            color_image = realsense.get_frame()
            if color_image is not None:
                # Resize to match the expected resolution if needed
                if color_image.shape[:2] != teleoperator.resolution_cropped:
                    color_image = cv2.resize(color_image, (teleoperator.resolution_cropped[1], teleoperator.resolution_cropped[0]))
                
                # Copy the image to shared memory
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                np.copyto(teleoperator.img_array[:, :teleoperator.resolution_cropped[1]], color_image_rgb)  # 左眼
                np.copyto(teleoperator.img_array[:, teleoperator.resolution_cropped[1]:], color_image_rgb)
            
            head_rmat, left_pose, right_pose, left_qpos, right_qpos, right_pose_w, right_landmarks = teleoperator.step()

            i += 1
            if i >= t:
                start = True

            if i >= key and start:
                if write == 1:
                    print("开始控制：")
                
                i = 0

                if last_right_pose_w is None or not np.allclose(right_pose_w, last_right_pose_w, atol=1e-6):
                    T_D_C_result = compute_T_D_C(right_pose_w)
                    new_joint_deg = motion_generator.plan_motion(last_joint_deg, T_D_C_result)
                    
                    if new_joint_deg is not None:
                        last_joint_deg = new_joint_deg
                        new_joint_rad = np.deg2rad(new_joint_deg[:6])

                        print("robot_angles:\n", np.round(new_joint_rad, 4))
                        print('This is the '+str(write)+'\n')
                    else:
                        new_joint_deg = last_joint_deg

                    write += 1
                    last_right_pose_w = right_pose_w.copy()

    except KeyboardInterrupt:
        print("Exiting...")
        realsense.stop()
        