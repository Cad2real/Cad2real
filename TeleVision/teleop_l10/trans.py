import numpy as np
from scipy.spatial.transform import Rotation as R

# Function: convert a quaternion and translation vector into a 4x4 transform matrix
def transform_matrix(translation, quaternion):

    # Create rotation matrix from quaternion
    rot_matrix = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
    
    # Construct 4x4 transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = translation
    return transform

# Function: extract quaternion and translation vector from a transform matrix
def extract_transform(matrix):

    # Extract rotation matrix
    rot_matrix = matrix[:3, :3]
    
    # Convert to quaternion
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()  # returns [qx, qy, qz, qw]
    
    # Extract translation vector
    translation = matrix[:3, 3]
    
    # Convert quaternion to [qw, qx, qy, qz] format
    return translation, [quat[3], quat[0], quat[1], quat[2]]

# 1. Pose of coordinate frame C relative to coordinate frame A
# trans.py

def compute_T_D_C(B_A):
    # Fixed transform from frame C to frame A
    R_A_C = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])
    r_A_C = R.from_matrix(R_A_C)
    quat_A_C = r_A_C.as_quat()
    quat_A_C = [quat_A_C[3], quat_A_C[0], quat_A_C[1], quat_A_C[2]]
    trans_A_C = [0, 0, 0]
    T_A_C = transform_matrix(trans_A_C, quat_A_C)

    # Split B_A into translation and quaternion
    trans_B_A = B_A[:3]
    quat_B_A = B_A[3:]
    T_B_A = transform_matrix(trans_B_A, quat_B_A)

    # D->B is a fixed rotation
    quat_D_B = [np.cos(np.radians(-90 / 2)), 0, 0, np.sin(np.radians(-90 / 2))]
    trans_D_B = [0, 0, 0]
    T_D_B = transform_matrix(trans_D_B, quat_D_B)

    # Combine transforms
    T_B_C = np.matmul(T_B_A, T_A_C)
    T_D_C = np.matmul(T_D_B, T_B_C)

    trans_D_C, quat_D_C = extract_transform(T_D_C)
    return np.concatenate([trans_D_C, quat_D_C])

