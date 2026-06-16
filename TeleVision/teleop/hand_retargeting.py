import numpy as np


def calculate_angle_between_vectors(v1, v2):
    # v1, v2为numpy数组，shape为(3,)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)
    return angle_radians


class HandRetarget:
    def __init__(self):
        # gripper parameters
        # 拇指和食指完全闭合的时候pinch_distance大概是8mm=0.008m
        # 这里设置的阈值是0.02m，即当拇指和食指的距离小于0.02m时，认为想要做pinch动作

        self.pinching_threshold = 0.02
        self.gripper_limits = (0.0, 90.0)

        # fingers parameters
        # 这些数据是需要根据人手大小不同来调的
        # 直接看手部关节数据，自由活动手部，看这些角度的上下限
        # 比如(40.0, 170.0)的意思是握拳的时候四根指头的角度是40，张开是170
        # 会被直接映射到因时机械手输入的0-1000之间
        self.four_fingers_limits = (40.0, 170.0)
        self.thumb_bending_limits = (15.0, 30.0)
        self.thumb_rotation_limits = (80.0, 150.0)

        # cache
        self.last_valid_left = None
        self.last_valid_right = None
        self.last_valid_left_pinch = None
        self.last_valid_right_pinch = None

    def compute_distance_between_points(self, finger_frames, index1, index2):
        """
        计算 landmarks 中 index1 和 index2 两个点之间的欧几里得距离。

        参数:
            landmarks (np.ndarray): 形状为 (N, 3) 的关键点数组。
            index1 (int): 第一个点的索引。
            index2 (int): 第二个点的索引。

        返回:
            float: 两点之间的距离。
        """
        point1 = finger_frames[index1]
        point2 = finger_frames[index2]
        distance = np.linalg.norm(point1 - point2)
        return distance


    def _get_point_angle(self, finger_frames, origin_idx, point1_idx, point2_idx):
        """
        计算 point1-origin-point2 的三维夹角，单位为角度（°）

        参数：
            right_landmarks: (25, 3) 的numpy数组，表示右手关键点坐标
            origin_idx: int，作为角点中心的关键点索引
            point1_idx: int，第一个关键点索引
            point2_idx: int，第二个关键点索引

        返回：
            float，三维夹角，单位为度
        """
        origin = finger_frames[origin_idx]
        point1 = finger_frames[point1_idx]
        point2 = finger_frames[point2_idx]

        v1 = point1 - origin
        v2 = point2 - origin

        angle_rad = calculate_angle_between_vectors(v1, v2)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _solve_four_fingers(self, finger_frames):
        # 顺序是 (little, ring, middle, index)
        four_angles = np.zeros(4)
        for i in range(4):
            # 6 to 9, 6 to 5 is index finger
            # plus 5 per finger
            angle = self._get_point_angle(
                finger_frames, 6+5*i, 5+5*i, 9+5*i)
            four_angles[3-i] = angle  # 倒着排

        # 这里两个值应该是人手打开和握拳的角度，映射到0到1000之间
        # 机械手的角度在19到176.7之间
        four_angles = np.clip(four_angles, *self.four_fingers_limits)
        four_angles = (four_angles - self.four_fingers_limits[0]) / (
            self.four_fingers_limits[1] - self.four_fingers_limits[0]) * 1000
        return four_angles
    
    def _solve_thumb(self, finger_frames, pinch_distance):
        # 在大多数情况下都是直接映射两个自由度
        bending_angle = self._get_point_angle(
            finger_frames, 1, 4, 6)
        rotation_angle = self._get_point_angle(
            finger_frames, 6, 3, 21)

        # bending
        # 人手角度在xx和xx之间，映射到0到1000，
        # 机械手值 -13.0deg 到 53.6deg
        bending_angle = np.clip(bending_angle, *self.thumb_bending_limits)
        bending_angle = (bending_angle - self.thumb_bending_limits[0]) / (
            self.thumb_bending_limits[1] - self.thumb_bending_limits[0]) * 1000

        # rotation
        # 人手角度在xx和xx之间，映射到0到1000，
        # 机械手值 90deg 到 165deg
        rotation_angle = np.clip(rotation_angle, *self.thumb_rotation_limits)
        rotation_angle = (rotation_angle - self.thumb_rotation_limits[0]) / (
            self.thumb_rotation_limits[1] - self.thumb_rotation_limits[0]) * 1000

        # 在pinch模式下例外
        # distance 0.01 到 0.04 之间，线性变换
        # bending_angle 400 到 1000 之间
        # 这里的参数都要根据人手大小来调，即使调的好，解决pinch问题也比较有限，无法捏住很细小的东西
        # 主要原因是六自由度的限制
        is_pinch_mode = pinch_distance < 0.04
        if is_pinch_mode:
            rotation_angle = 150
            pinch_distance = np.clip(pinch_distance, 0.01, 0.04)
            bending_angle = 800 * (pinch_distance - 0.01) / (0.04 - 0.01) + 400

        return bending_angle, rotation_angle
    
    def solve_fingers_angles(self, right_landmarks):
        # 如果末端执行器是因时六自由度灵巧手，用这个函数

        # 检查是否为空输入
        is_empty = np.all(right_landmarks[:3] == 0)

        # 初始化阶段，设置张开状态
        if self.last_valid_right is None:
            if is_empty:
                # 初始阶段且为空，则返回张开状态
                self.last_valid_right = np.zeros((25, 3))  # 占位
                self.last_valid_right_pinch = 0.05  # 大于 pinching 阈值
                return np.array([1000, 1000, 1000, 1000, 1000, 1000])
            else:
                # 初始阶段且不为空，记录当前值
                self.last_valid_right = right_landmarks
                self.last_valid_right_pinch = self.compute_distance_between_points(right_landmarks, 4, 9)

        # 控制阶段
        if is_empty:
            finger_frames = self.last_valid_right
            pinch_distance = self.last_valid_right_pinch
        else:
            finger_frames = right_landmarks
            pinch_distance = self.compute_distance_between_points(finger_frames, 4, 9)
            self.last_valid_right = finger_frames
            self.last_valid_right_pinch = pinch_distance

        right_four_fingers_angles = self._solve_four_fingers(finger_frames)
        right_thumb_angles = self._solve_thumb(finger_frames, pinch_distance)
        right_angles = np.concatenate((right_four_fingers_angles, right_thumb_angles))

        return right_angles