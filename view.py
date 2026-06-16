
import numpy as np
import genesis as gs


NOVA2_PARAMS = [  # ... same as above ...
    {"joint": "nova2joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint2", "kp": 300.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova2joint3", "kp": 300.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova2joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova2joint6", "kp": 150.0, "ctrlrange": (-6.28,  6.28)},
]
NOVA5_PARAMS = [
    {"joint": "nova5joint1", "kp": 300.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint2", "kp": 300.0, "ctrlrange": (-3.14,  3.14)},
    {"joint": "nova5joint3", "kp": 300.0, "ctrlrange": (-2.79,  2.79)},
    {"joint": "nova5joint4", "kp": 250.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint5", "kp": 200.0, "ctrlrange": (-6.28,  6.28)},
    {"joint": "nova5joint6", "kp": 150.0, "ctrlrange": (-6.28,  6.28)},
]
RIGHT_HAND_PARAMS = [
    {"joint": "R_thumb_cmc_roll",   "kp": 40.0, "ctrlrange": (0.0, 1.1339)},
    {"joint": "R_thumb_cmc_yaw",    "kp": 40.0, "ctrlrange": (0.0, 1.9189)},
    {"joint": "R_thumb_cmc_pitch",  "kp": 35.0, "ctrlrange": (0.0, 0.5146)},
    {"joint": "R_index_mcp_roll",   "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "R_index_mcp_pitch",  "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_middle_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_ring_mcp_roll",    "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "R_ring_mcp_pitch",   "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "R_pinky_mcp_roll",   "kp": 25.0, "ctrlrange": (0.0, 0.3489)},
    {"joint": "R_pinky_mcp_pitch",  "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
]
LEFT_HAND_PARAMS = [
    {"joint": "L_thumb_cmc_roll",   "kp": 40.0, "ctrlrange": (0.0, 1.1339)},
    {"joint": "L_thumb_cmc_yaw",    "kp": 40.0, "ctrlrange": (0.0, 1.9189)},
    {"joint": "L_thumb_cmc_pitch",  "kp": 35.0, "ctrlrange": (0.0, 0.5149)},
    {"joint": "L_index_mcp_roll",   "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "L_index_mcp_pitch",  "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_middle_mcp_pitch", "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_ring_mcp_roll",    "kp": 30.0, "ctrlrange": (0.0, 0.2181)},
    {"joint": "L_ring_mcp_pitch",   "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
    {"joint": "L_pinky_mcp_roll",   "kp": 25.0, "ctrlrange": (0.0, 0.3489)},
    {"joint": "L_pinky_mcp_pitch",  "kp": 35.0, "ctrlrange": (0.0, 1.3607)},
]

# ---------------------------
# Helpers
# ---------------------------
def params_to_arrays(group):
    names = [p["joint"] for p in group]
    kp    = np.array([p["kp"] for p in group], dtype=np.float32)
    lo    = np.array([p["ctrlrange"][0] for p in group], dtype=np.float32)
    hi    = np.array([p["ctrlrange"][1] for p in group], dtype=np.float32)
    return names, kp, lo, hi

def resolve_dofs(entity, joint_names):
    """Return dof indices for names that exist; also list any missing names."""
    idx, ok, missing = [], [], []
    for n in joint_names:
        try:
            j = entity.get_joint(n)
            idx.append(j.dof_idx_local)
            ok.append(n)
        except Exception:
            missing.append(n)
    return np.array(idx, dtype=int), ok, missing

def set_group_gains(entity, dof_idx, kp, kv_scale=2.0, kp_scale=1.0):
    if dof_idx.size == 0:
        return
    kp_eff = kp_scale * kp
    kv_eff = kv_scale * np.sqrt(kp_eff)  # heuristic ~critically damped
    entity.set_dofs_kp(kp_eff, dofs_idx_local=dof_idx)
    entity.set_dofs_kv(kv_eff, dofs_idx_local=dof_idx)

def clip_range(q, lo, hi):
    return np.minimum(np.maximum(q, lo), hi)

def smoothstep_5th(u):
    """Min-jerk scalar s(u) in [0,1]."""
    return 10*u**3 - 15*u**4 + 6*u**5

# ---------------------------
# Build scene & load MJCF
# ---------------------------
MJCF_PATH = "/home/louhz/cad2real/robot_urdf/scene.xml"  # <--- set this to your MJCF with NOVA2/NOVA5 + hands

gs.init(backend=gs.gpu)  # or gs.cpu if needed
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.0, -3.5, 2.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30, max_FPS=60
    ),
    show_viewer=True,
)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file=MJCF_PATH))
scene.build()

# ---------------------------
# Resolve groups -> dof indices
# ---------------------------
n2_names, n2_kp, n2_lo, n2_hi = params_to_arrays(NOVA2_PARAMS)
n5_names, n5_kp, n5_lo, n5_hi = params_to_arrays(NOVA5_PARAMS)
rh_names, rh_kp, rh_lo, rh_hi = params_to_arrays(RIGHT_HAND_PARAMS)
lh_names, lh_kp, lh_lo, lh_hi = params_to_arrays(LEFT_HAND_PARAMS)

n2_idx, n2_ok, n2_missing = resolve_dofs(robot, n2_names)
n5_idx, n5_ok, n5_missing = resolve_dofs(robot, n5_names)
rh_idx, rh_ok, rh_missing = resolve_dofs(robot, rh_names)
lh_idx, lh_ok, lh_missing = resolve_dofs(robot, lh_names)

if n2_missing or n5_missing or rh_missing or lh_missing:
    print("[Warn] Missing joints:", {"nova2": n2_missing, "nova5": n5_missing,
                                     "right": rh_missing, "left": lh_missing})

# ---------------------------
# Apply gains (tune kp_scale if too soft)
# ---------------------------
set_group_gains(robot, n2_idx, n2_kp, kv_scale=2.0, kp_scale=1.0)
set_group_gains(robot, n5_idx, n5_kp, kv_scale=2.0, kp_scale=1.0)
set_group_gains(robot, rh_idx, rh_kp, kv_scale=2.0, kp_scale=1.0)
set_group_gains(robot, lh_idx, lh_kp, kv_scale=2.0, kp_scale=1.0)

# ---------------------------
# Define trajectory keyframes
# ---------------------------
# Arms: simple mirrored reach in joint space
n2_home = np.zeros_like(n2_kp)
n2_goal = np.array([0.6, -0.5, 0.8, -0.6, 0.4, 0.3], dtype=np.float32)

n5_home = np.zeros_like(n5_kp)
n5_goal = -n2_goal  # mirror

# Hands: open (all near lower bound) -> close (80% of range) -> open
rh_open = np.copy(rh_lo)
rh_close = rh_lo + 0.8 * (rh_hi - rh_lo)

lh_open = np.copy(lh_lo)
lh_close = lh_lo + 0.8 * (lh_hi - lh_lo)

# ---------------------------
# Run two min-jerk phases: home->goal, then goal->home
# ---------------------------
dt = 0.02
T_phase = 2.0  # seconds per phase
phases = [
    (n2_home, n2_goal, n5_home, n5_goal, rh_open, rh_close, lh_open, lh_close),
    (n2_goal, n2_home, n5_goal, n5_home, rh_close, rh_open, lh_close, lh_open),
]

for (a0, a1, b0, b1, r0, r1, l0, l1) in phases:
    steps = int(T_phase / dt)
    for k in range(steps):
        u = (k + 1) / steps
        s = smoothstep_5th(u)

        q_n2 = clip_range(a0 + s * (a1 - a0), n2_lo, n2_hi)
        q_n5 = clip_range(b0 + s * (b1 - b0), n5_lo, n5_hi)
        q_rh = clip_range(r0 + s * (r1 - r0), rh_lo, rh_hi)
        q_lh = clip_range(l0 + s * (l1 - l0), lh_lo, lh_hi)

        # Position control on each group (Genesis PD controller)
        robot.control_dofs_position(q_n2, n2_idx)
        robot.control_dofs_position(q_n5, n5_idx)
        robot.control_dofs_position(q_rh, rh_idx)
        robot.control_dofs_position(q_lh, lh_idx)

        # (Optional) inspect control / internal forces:
        # print("arm2 control force:", robot.get_dofs_control_force(n2_idx))

        scene.step()

print("Done.")