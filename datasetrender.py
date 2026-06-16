# fine tune the pretrained transformer in genesis




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



import os, ast, numpy as np
from pathlib import Path

# ---------------------------
# Helpers: load, resample, clamp, and shape alignment
# ---------------------------
def load_ros_txt_positions(txt_path, key="position"):
    """
    Parse a ROS-like text dump with repeated blocks:
      seq: N
      secs: <int>
      nsecs: <int>
      position: [ ... ]

    Returns:
      t   : (N,) float64 seconds, relative to first sample
      q   : (N, D) float64 positions
    """
    t_list, q_list = [], []
    secs, nsecs = None, None
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("secs:"):
                try: secs = int(s.split(":", 1)[1].strip())
                except: secs = None
            elif s.startswith("nsecs:"):
                try: nsecs = int(s.split(":", 1)[1].strip())
                except: nsecs = 0
            elif s.startswith(f"{key}:"):
                arr = ast.literal_eval(s.split(":", 1)[1].strip())
                q_list.append(np.asarray(arr, dtype=np.float64))
                if secs is not None:
                    t_list.append(float(secs) + float(nsecs or 0) * 1e-9)
                else:
                    # fallback to monotone counters if timestamps missing
                    t_list.append(len(t_list) * 1.0)

    if not q_list:
        raise RuntimeError(f"No '{key}:' entries found in {txt_path}")

    t = np.asarray(t_list, dtype=np.float64)
    t -= t[0]  # relative time
    q = np.vstack(q_list)  # (N, D)
    # Remove any duplicate timestamps that could break interpolation
    uniq, idx = np.unique(t, return_index=True)
    return uniq, q[idx]

def resample_trajectory(t_src, q_src, dt):
    """
    Linear resample to uniform grid [0, T] with step dt.
    """
    if t_src[-1] <= 0 or dt <= 0:
        return np.array([0.0]), q_src[:1]
    t_grid = np.arange(0.0, t_src[-1] + 1e-9, dt, dtype=np.float64)
    D = q_src.shape[1]
    q_grid = np.zeros((t_grid.size, D), dtype=np.float64)
    # per-dof 1D interpolation
    for j in range(D):
        q_grid[:, j] = np.interp(t_grid, t_src, q_src[:, j])
    return t_grid, q_grid

def align_dims(q, target_D):
    """
    Make q.shape[1] == target_D by truncating or zero-padding.
    """
    N, D = q.shape
    if D == target_D:
        return q
    out = np.zeros((N, target_D), dtype=q.dtype)
    m = min(D, target_D)
    out[:, :m] = q[:, :m]
    if D > target_D:
        print(f"[Warn] Traj has {D} columns, NOVA2 expects {target_D}. Truncating extras.")
    else:
        print(f"[Warn] Traj has {D} columns, NOVA2 expects {target_D}. Zero-padding the rest.")
    return out

def clamp_to_limits(q, lo, hi):
    """
    q: (N, D), lo/hi: (D,)
    """
    return np.minimum(np.maximum(q, lo[None, :]), hi[None, :])



from train import DrawDataset




if __name__ == "__main__":
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

    cam = scene.add_camera(
        res=(960, 640),
        pos=(2.8, -3.5, 2.3),
        lookat=(0.0, 0.0, 0.5),
        fov=35,
        GUI=False,  # no extra OpenCV window
    )

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

    indices = dict(n2=n2_idx, n5=n5_idx, rh=rh_idx, lh=lh_idx)
    bounds  = dict(n2=(n2_lo, n2_hi), n5=(n5_lo, n5_hi), rh=(rh_lo, rh_hi), lh=(lh_lo, lh_hi))

    dt_cmd   = 0.02   # command (control) update interval (your desired playback rate)
    dt_simul = 0.01   # your SimOptions(dt=0.01) above
    steps_per_cmd = max(1, int(round(dt_cmd / dt_simul)))

    # ----- Usage: replace your min-jerk block with this -----
    dt_sim = 0.02  # your integrator step (0.02 s above)


    # load the trajectory from the dataset and use genesis to deploy it on robot in simulation and save the rendering results

    # this is the nova2 trajectory

    nova2_path = 'exp_1/draw/draw_t5/nova2.txt'
    nova5_path = 'exp_1/draw/draw_t5/nova5.txt'
    left_path  = 'exp_1/draw/draw_t5/left.txt'
    right_path = 'exp_1/draw/draw_t5/right.txt'

    video_out = Path('renders') / 'draw_t5_all.mp4'
    video_out.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Playback timing
    #   - dt_simul must match your SimOptions(dt=...) above.
    #   - dt_cmd is how often you change the joint targets.
    # ---------------------------------------------------------
    dt_cmd   = 0.02   # command period
    dt_simul = 0.01   # <-- must match gs.options.SimOptions(dt=0.01) above
    steps_per_cmd = max(1, int(round(dt_cmd / dt_simul)))

    TARGET_STEPS = 1000  # total number of command steps for each group in this single video

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _down_or_upsample_to_N(t_src, q_src, N_target):
        """
        Return exactly N_target samples using time-aware interpolation if needed.
        - If q_src has >= N_target, pick N_target evenly spaced samples (includes first/last).
        - If q_src has <  N_target, linearly interpolate to N_target over [t_src[0], t_src[-1]].
        """
        N_src, D = q_src.shape
        if N_src == 0:
            raise RuntimeError("Empty trajectory")
        if N_src >= N_target:
            idx = np.linspace(0, N_src - 1, N_target, dtype=int)
            return q_src[idx]
        # upsample via interpolation
        t_grid = np.linspace(t_src[0], t_src[-1], N_target, dtype=np.float64)
        q_up = np.zeros((N_target, D), dtype=np.float64)
        for j in range(D):
            q_up[:, j] = np.interp(t_grid, t_src, q_src[:, j])
        return q_up

    def _prepare_group_cmd(txt_path, group_key, N_target=TARGET_STEPS):
        """
        Load a ROS-like log, produce exactly N_target waypoints,
        align to the group's DOF count, clamp to that group's limits.
        """
        t_src, q_src = load_ros_txt_positions(txt_path, key="position")  # returns relative seconds + (N,D)
        # to exactly N_target steps
        q = _down_or_upsample_to_N(t_src, q_src, N_target)

        # NEW: loaded results are in degrees -> convert to radians
        q = np.deg2rad(q).astype(q.dtype, copy=False)

        # align to group's dimensionality
        target_D = len(indices[group_key])
        if target_D == 0:
            print(f"[Skip] {group_key}: no resolved joints.")
            return None

        N = q.shape[0]
        D = q.shape[1]
        q_aligned = np.zeros((N, target_D), dtype=q.dtype)
        m = min(D, target_D)
        q_aligned[:, :m] = q[:, :m]
        if D > target_D:
            print(f"[Warn] {group_key}: trunc {D}->{target_D} columns")
        elif D < target_D:
            print(f"[Warn] {group_key}: pad {D}->{target_D} columns with zeros")

        # clamp to joint limits (ASSUMED radians)
        lo, hi = bounds[group_key]
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        q_aligned = np.minimum(np.maximum(q_aligned, lo[None, :]), hi[None, :])

        print(f"[traj:{group_key}] using {N} steps (target {N_target}) from file: {txt_path}")
        return q_aligned


    # ---------------------------------------------------------
    # Build all four command arrays (exactly 250 steps each)
    # ---------------------------------------------------------
    group_specs = [
        ("n2", nova2_path),
        ("n5", nova5_path),
        ("lh", left_path),
        ("rh", right_path),
    ]

    q_cmds = {}
    active_groups = []
    for key, path in group_specs:
        p = Path(path)
        if not p.exists():
            print(f"[Skip] {key}: file not found -> {path}")
            continue
        q = _prepare_group_cmd(path, key, TARGET_STEPS)
        if q is None:
            continue
        q_cmds[key] = q
        active_groups.append(key)

    if not active_groups:
        raise RuntimeError("No playable groups found. Check file paths and resolved DOFs.")

    # ---------------------------------------------------------
    # Move robot to the first sample for all active groups, then settle
    # ---------------------------------------------------------
    robot.zero_all_dofs_velocity()
    for key in active_groups:
        robot.set_dofs_position(q_cmds[key][0], indices[key])

    for _ in range(30):  # short settle
        scene.step()

    # ---------------------------------------------------------
    # Single recording covering all groups together
    # ---------------------------------------------------------
    cam.start_recording()

    for i in range(TARGET_STEPS):
        # set targets for all groups at this command step
        for key in active_groups:
            robot.control_dofs_position(q_cmds[key][i], indices[key])
        # advance physics and render in-between
        for _ in range(steps_per_cmd):
            scene.step()
            cam.render()

    # tail to let everything settle in the video
    for _ in range(int(0.5 / dt_simul)):
        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename=str(video_out), fps=int(round(1.0 / dt_simul)))
    print(f"[OK] Saved video to: {video_out.resolve()}")