from __future__ import annotations

import numpy as np


NOVA2_PARAMS = [
    {"joint": "nova2joint1", "kp": 300.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova2joint2", "kp": 300.0, "ctrlrange": (-3.14, 3.14)},
    {"joint": "nova2joint3", "kp": 300.0, "ctrlrange": (-2.79, 2.79)},
    {"joint": "nova2joint4", "kp": 250.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova2joint5", "kp": 200.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova2joint6", "kp": 150.0, "ctrlrange": (-6.28, 6.28)},
]

NOVA5_PARAMS = [
    {"joint": "nova5joint1", "kp": 300.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova5joint2", "kp": 300.0, "ctrlrange": (-3.14, 3.14)},
    {"joint": "nova5joint3", "kp": 300.0, "ctrlrange": (-2.79, 2.79)},
    {"joint": "nova5joint4", "kp": 250.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova5joint5", "kp": 200.0, "ctrlrange": (-6.28, 6.28)},
    {"joint": "nova5joint6", "kp": 150.0, "ctrlrange": (-6.28, 6.28)},
]

RIGHT_HAND_PARAMS = [
    {"joint": "R_thumb_cmc_roll", "kp": 40.0, "ctrlrange": (0.0, 1.1339)},
    {"joint": "R_thumb_cmc_yaw", "kp": 40.0, "ctrlrange": (0.0, 1.9189)},
    {"joint": "R_thumb_cmc_pitch", "kp": 35.0, "ctrlrange": (0.0, 0.5146)},
    {"joint": "R_index_mcp_roll", "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "R_index_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_middle_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_ring_mcp_roll", "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "R_ring_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_pinky_mcp_roll", "kp": 25.0, "ctrlrange": (0.0, 0.3489)},
    {"joint": "R_pinky_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
]

LEFT_HAND_PARAMS = [
    {"joint": "L_thumb_cmc_roll", "kp": 40.0, "ctrlrange": (0.0, 1.1339)},
    {"joint": "L_thumb_cmc_yaw", "kp": 40.0, "ctrlrange": (0.0, 1.9189)},
    {"joint": "L_thumb_cmc_pitch", "kp": 35.0, "ctrlrange": (0.0, 0.5149)},
    {"joint": "L_index_mcp_roll", "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "L_index_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_middle_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_ring_mcp_roll", "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "L_ring_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_pinky_mcp_roll", "kp": 25.0, "ctrlrange": (0.0, 0.3489)},
    {"joint": "L_pinky_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
]


def params_to_arrays(group):
    names = [p["joint"] for p in group]
    kp = np.array([p["kp"] for p in group], dtype=np.float32)
    lo = np.array([p["ctrlrange"][0] for p in group], dtype=np.float32)
    hi = np.array([p["ctrlrange"][1] for p in group], dtype=np.float32)
    return names, kp, lo, hi


def resolve_dofs(entity, joint_names):
    idx, ok, missing = [], [], []
    for name in joint_names:
        try:
            joint = entity.get_joint(name)
            idx.append(joint.dof_idx_local)
            ok.append(name)
        except Exception:
            missing.append(name)
    return np.array(idx, dtype=int), ok, missing


def set_group_gains(entity, dof_idx, kp, kv_scale=2.0, kp_scale=1.0):
    if dof_idx.size == 0:
        return
    kp_eff = kp_scale * kp
    kv_eff = kv_scale * np.sqrt(kp_eff)
    entity.set_dofs_kp(kp_eff, dofs_idx_local=dof_idx)
    entity.set_dofs_kv(kv_eff, dofs_idx_local=dof_idx)

