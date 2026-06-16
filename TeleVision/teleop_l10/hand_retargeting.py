import numpy as np


def calculate_angle_between_vectors(v1, v2):
    # v1 and v2 are numpy arrays with shape (3,)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)
    return angle_radians

def calculate_xzangle_between_vectors(v1, v2):
    # v1 and v2 are numpy arrays with shape (3,)
    v1_xz = np.array([v1[0], 0, v1[2]])
    v2_xz = np.array([v2[0], 0, v2[2]])
    norm_v1 = np.linalg.norm(v1_xz)
    norm_v2 = np.linalg.norm(v2_xz)
    cos_theta = np.dot(v1_xz, v2_xz) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


class HandRetarget:
    def __init__(self, hand_type='right'):
        # Set the hand type ('left' or 'right')
        self.hand_type = hand_type
        
        # Pinch threshold of 0.02m: if thumb and index distance is below this,
        # treat it as a pinch gesture.
        self.pinching_threshold = 0.02
        self.gripper_limits = (0.0, 90.0)

        # Human finger angle limits
        self.four_fingers_limits = (60.0, 170.0)
        self.index_finger_limits_sway = (11.0, 30.0)
        self.ring_finger_limits_sway = (10.0, 17.0)
        self.little_finger_limits_sway = (25.0, 50.0)
        self.thumb_bending_limits = (90.0, 176.0)
        self.thumb_rotation_limits = (80.0, 130.0)
        self.thumb_revolve_limits = (1.0, 25.0)

        self.last_revolve_angle = None
        self.revolve_angle_tolerance = 12.0

        # Shared cache variables
        self.last_valid_landmarks = None
        self.last_valid_pinch_distance = None

    def compute_distance_between_points(self, landmarks, index1, index2):
        """
        Compute the Euclidean distance between two points in landmarks.

        Args:
            landmarks (np.ndarray): Landmark array with shape (N, 3).
            index1 (int): Index of the first point.
            index2 (int): Index of the second point.

        Returns:
            float: Distance between the two points.
        """
        point1 = landmarks[index1]
        point2 = landmarks[index2]
        distance = np.linalg.norm(point1 - point2)
        return distance


    def _get_point_angle(self, landmarks, origin_idx, point1_idx, point2_idx):
        """
        Compute the 3D angle between point1-origin-point2 in degrees.

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.
            origin_idx: int index of the origin point.
            point1_idx: int index of the first point.
            point2_idx: int index of the second point.

        Returns:
            float: 3D angle in degrees.
        """
        origin = landmarks[origin_idx]
        point1 = landmarks[point1_idx]
        point2 = landmarks[point2_idx]

        v1 = point1 - origin
        v2 = point2 - origin

        angle_rad = calculate_angle_between_vectors(v1, v2)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def _get_point_xzangle(self, landmarks, origin_idx, point1_idx, point2_idx):
        """
        Compute the 2D angle between point1-origin-point2 projected onto the palm plane, in degrees.

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.
            origin_idx: int index of the origin point.
            point1_idx: int index of the first point.
            point2_idx: int index of the second point.

        Returns:
            float: Plane-projected angle in degrees.
        """
        origin = landmarks[origin_idx]
        point1 = landmarks[point1_idx]
        point2 = landmarks[point2_idx]

        v1 = point1 - origin
        v2 = point2 - origin

        angle_rad = calculate_xzangle_between_vectors(v1, v2)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _solve_four_fingers(self, landmarks, pinch_distances=None):
        """
        Compute bending control signals for the four fingers in the range [0, 255].

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.
            pinch_distances: optional dict containing thumb-to-finger distances.

        Returns:
            np.ndarray: Bending angles for the four fingers in [0, 255].
        """
        # Order is (index, middle, ring, little)
        four_angles = np.zeros(4)
        for i in range(4):
            angle = self._get_point_angle(
                landmarks, 6+5*i, 5+5*i, 9+5*i)
            four_angles[i] = angle 

        other_fingers_angles = four_angles[:3]
        other_fingers_angles = np.clip(other_fingers_angles, *self.four_fingers_limits)
        other_fingers_angles = (other_fingers_angles - self.four_fingers_limits[0]) / (
            self.four_fingers_limits[1] - self.four_fingers_limits[0]) * 255

        # --- Handle little finger separately (index 3) ---
        little_finger_angle_raw = four_angles[3]
        little_finger_final_angle = 0

        # Check if the little finger is in pinch state
        is_pinching_little = (pinch_distances is not None and
                            24 in pinch_distances and
                            pinch_distances[24] < 0.02)

        if is_pinching_little:
            # If pinching, map distance [0.01, 0.02] to bend values [120, 130]
            print("INFO: Detected Little Finger Pinch") # show state in console
            dist = np.clip(pinch_distances[24], 0.01, 0.02)
            # Linear interpolation formula
            little_finger_final_angle = 120 + ((dist - 0.01) / (0.02 - 0.01)) * (130 - 120)
        else:
            # If not pinching, use the standard angle mapping
            clipped_angle = np.clip(little_finger_angle_raw, *self.four_fingers_limits)
            little_finger_final_angle = (clipped_angle - self.four_fingers_limits[0]) / (
                self.four_fingers_limits[1] - self.four_fingers_limits[0]) * 255

        # --- Compose final results ---
        final_angles = np.zeros(4)
        final_angles[:3] = other_fingers_angles
        final_angles[3] = little_finger_final_angle

        return final_angles
    
    def _solve_three_fingers_sway(self, landmarks):
        """
        Compute sway control signals for three fingers in the range [0, 255].

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.

        Returns:
            np.ndarray: Sway angles for three fingers in [0, 255].
        """
        # Order is (index, ring, little)
        limits = [
            self.index_finger_limits_sway,   # three_angles_sway[0]
            self.ring_finger_limits_sway,    # three_angles_sway[1]
            self.little_finger_limits_sway   # three_angles_sway[2]
        ]
        three_angles_sway = np.zeros(3)
        finger_indices = [0,2,3]
        for i, finger_idx in enumerate(finger_indices):
            angle_sway = self._get_point_angle(
                landmarks, 11, 9+5*i, 14)
            three_angles_sway[i] = angle_sway 

        for i in range(3):
            min_val, max_val = limits[i]

            angle = np.clip(three_angles_sway[i], min_val, max_val)
            mapped = (angle - min_val) / (max_val - min_val) * 255

            three_angles_sway[i] = mapped

        return three_angles_sway
    
    def _solve_thumb(self, finger_frames, pinch_distance_all=None):
        """
        Compute thumb bending, sway, and rotation control signals in the range [0, 255].

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.
            pinch_distances: optional dict containing thumb-to-finger distances.

        Returns:
            tuple: Three values for thumb bending, rotation, and revolve in [0, 255].
        """
        bending_angle = self._get_point_angle(
            finger_frames, 2, 1, 4)
        rotation_angle = self._get_point_angle(
            finger_frames, 6, 3, 21)
        revolve_angle = self._get_point_xzangle(
            finger_frames, 20, 4, 21)

        bending_angle = np.clip(bending_angle, *self.thumb_bending_limits)
        bending_angle = (bending_angle - self.thumb_bending_limits[0]) / (
            self.thumb_bending_limits[1] - self.thumb_bending_limits[0]) * 255

        rotation_angle = np.clip(rotation_angle, *self.thumb_rotation_limits)
        rotation_angle = (rotation_angle - self.thumb_rotation_limits[0]) / (
            self.thumb_rotation_limits[1] - self.thumb_rotation_limits[0]) * 255

        if pinch_distance_all is None:
            pinch_distance_all = {
                9: self.compute_distance_between_points(finger_frames, 4, 9),
                14: self.compute_distance_between_points(finger_frames, 4, 14),
                19: self.compute_distance_between_points(finger_frames, 4, 19),
                24: self.compute_distance_between_points(finger_frames, 4, 24),
            }

        nearest_id, nearest_dist = min(pinch_distance_all.items(), key=lambda x: x[1])
        is_pinching = nearest_dist < 0.02

        if is_pinching:
            thumb_rot_map = {9: 8, 14: 0, 19: 0, 24: 0}
            thumb_rev_map = {9: 255, 14: 203, 19: 148, 24: 90}
            rotation_angle = thumb_rot_map[nearest_id]
            revolve_angle = thumb_rev_map[nearest_id]
            pinch_distance = np.clip(nearest_dist, 0.01, 0.02)
            bending_angle = 25 * (pinch_distance - 0.01) / (0.02 - 0.01) + 145
        else:
            revolve_angle = np.clip(revolve_angle, *self.thumb_revolve_limits)
            normalized_angle = (revolve_angle - self.thumb_revolve_limits[0]) / (
                self.thumb_revolve_limits[1] - self.thumb_revolve_limits[0])
            
            # Apply different logic based on hand type
            if self.hand_type == 'right':
                inverted_normalized_angle = 1.0 - normalized_angle
                revolve_angle = 90 + inverted_normalized_angle * 155
                if rotation_angle >= 5:
                    revolve_angle = 255
            else:  # left hand
                revolve_angle = 90 + normalized_angle * 110
                if rotation_angle >= 10:
                    revolve_angle = 255

        if self.last_revolve_angle is None:
            self.last_revolve_angle = revolve_angle

        if abs(revolve_angle - self.last_revolve_angle) > self.revolve_angle_tolerance:
            self.last_revolve_angle = revolve_angle
        else:
            revolve_angle = self.last_revolve_angle
            return bending_angle, rotation_angle, revolve_angle
    
    def solve_fingers_angles(self, landmarks):
        """
        Compute all dexterous hand control signals in the range [0, 255].

        Args:
            landmarks: numpy array with shape (25, 3) containing hand keypoint coordinates.

        Returns:
            np.ndarray: Array of all finger control values with length 10.
            (thumb bending, thumb rotation,
              index bending, middle bending, ring bending, little bending,
              index sway, ring sway, little sway,
              thumb revolve)
        """
        # Check for empty input
        is_empty = np.all(landmarks[:3] == 0)

        # Initialization phase: set open hand state
        if self.last_valid_landmarks is None:
            if is_empty:
                # Initial phase and empty input, return open hand state
                self.last_valid_landmarks = np.zeros((25, 3))  # placeholder
                self.last_valid_pinch_distance = 0.05  # above pinch threshold
                return np.full(10, 255)
            else:
                # Initial phase and valid input, store current values
                self.last_valid_landmarks = landmarks
                self.last_valid_pinch_distance = self.compute_distance_between_points(landmarks, 4, 9)

        # Control phase
        pinch_distances = None

        if is_empty:
            finger_frames = self.last_valid_landmarks
        else:
            finger_frames = landmarks
            pinch_distances = {
                9: self.compute_distance_between_points(finger_frames, 4, 9),
                14: self.compute_distance_between_points(finger_frames, 4, 14),
                19: self.compute_distance_between_points(finger_frames, 4, 19),
                24: self.compute_distance_between_points(finger_frames, 4, 24),
            }
            self.last_valid_landmarks = finger_frames
            self.last_valid_pinch_distance = pinch_distances[9]

        four_fingers_angles = self._solve_four_fingers(finger_frames, pinch_distances)
        three_fingers_angles_sway = self._solve_three_fingers_sway(finger_frames)
        thumb_bending, thumb_rotation, thumb_revolve = self._solve_thumb(finger_frames, pinch_distances)
        
        angles = np.array([
            thumb_bending,                       # thumb bending (1 value)
            thumb_rotation,                      # thumb rotation (1 value)
            *four_fingers_angles,                # four fingers bending (4 values)
            *three_fingers_angles_sway,          # three fingers sway (3 values)
            thumb_revolve                        # thumb revolve (1 value)
        ])

        return angles