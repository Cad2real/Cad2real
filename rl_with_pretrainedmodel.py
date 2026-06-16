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

# ---------------------------
# Build scene & load MJCF
# ---------------------------


# ---------------------------
# Define trajectory keyframes
# ---------------------------
# Arms: simple mirrored reach in joint space


# now infer this actions from the pretrained transformer and use reward to backward optimize this transformer policy:







from train import DrawDataset





root = "exp_1/draw"  # experiment folder containing draw_t1, draw_t2, ...
ds = DrawDataset(
        root=root,
        index_mode="stream",      # or "frame"
        decode_video=True,       # set False to skip decoding and only get frame indices
        video_backend_preference=None,  # "torchvision"|"opencv"|"read_video"|None
        transform=None,          # e.g., torchvision transforms expecting CxHxW FloatTensor
        align_mode="relative",   # "time" if you provide per-episode epoch0
        episode_time_offsets={   # only needed for align_mode="time"
            # "draw_t1": 1755682654.406556,  # secs + nsecs*1e-9 at video frame 0
        }
    )

from torch.utils.data import DataLoader
from train import PerEpisodeStreamBatchSampler

batch_sampler = PerEpisodeStreamBatchSampler(ds)
loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=0,
                    collate_fn=DrawDataset.collate_fn)

# Example usage (pseudo):

# can you load the trajectory of each arm and hand for the trajectory, and simulate it 





from helper import resample_traj, pad_to, unpack_sample, split_batch

def rollout_from_traj(q_n2, q_n5, q_rh, q_lh, bounds, indices, record_err=True):
        """
        Steps the scene using per-step joint position targets.
        All q_* are [T, D].
        """
        T = max(q_n2.shape[0], q_n5.shape[0], q_rh.shape[0], q_lh.shape[0])
        q_n2 = pad_to(q_n2, T); q_n5 = pad_to(q_n5, T); q_rh = pad_to(q_rh, T); q_lh = pad_to(q_lh, T)

        logs = dict(tracking_err=[]) if record_err else {}

        # Optional: reset between rollouts if your API supports it
        if hasattr(scene, 'reset'):
            scene.reset()

        for k in range(T):
            qn2 = clip_range(q_n2[k], bounds['n2'][0], bounds['n2'][1])
            qn5 = clip_range(q_n5[k], bounds['n5'][0], bounds['n5'][1])
            qrh = clip_range(q_rh[k], bounds['rh'][0], bounds['rh'][1])
            qlh = clip_range(q_lh[k], bounds['lh'][0], bounds['lh'][1])

            robot.control_dofs_position(qn2, indices['n2'])
            robot.control_dofs_position(qn5, indices['n5'])
            robot.control_dofs_position(qrh, indices['rh'])
            robot.control_dofs_position(qlh, indices['lh'])

            scene.step()

            # (optional) record basic joint tracking error
            if record_err and hasattr(robot, 'get_dofs_angle'):
                meas_n2 = robot.get_dofs_angle(indices['n2'])
                meas_n5 = robot.get_dofs_angle(indices['n5'])
                err = float(np.mean((meas_n2 - qn2)**2) + np.mean((meas_n5 - qn5)**2))
                logs['tracking_err'].append(err)

        return logs

def run_dataset_rollouts(loader, dt_sim, bounds, indices, match_dt=True):
        """
        Consume exactly ONE batch from `loader`, simulate each sample in that batch,
        and return the collected logs.
        """
        all_logs = []

        for batch_idx, batch in enumerate(loader):
            samples = split_batch(batch)

            # If you ever want only the first sample in that batch, uncomment:
            samples = samples[:1]

            for s in samples:

                # i want to load the dof checkpoint of right_position, left_position, nova2_position, nova5_position 

                # the datas are stored as the (t,n_dofs)
                q_n2, q_n5, q_rh, q_lh, dt_src = unpack_sample(s, dt_default=dt_sim)

                if match_dt:
                    # Preserve the original wall-clock duration per sample
                    T_src = q_n2.shape[0]  # if groups differ, resample each separately based on its own T
                    duration = T_src * dt_src
                    T_new = max(1, int(np.round(duration / dt_sim)))

                    q_n2 = resample_traj(q_n2, T_new)
                    q_n5 = resample_traj(q_n5, T_new)
                    q_rh = resample_traj(q_rh, T_new)
                    q_lh = resample_traj(q_lh, T_new)

                logs = rollout_from_traj(q_n2, q_n5, q_rh, q_lh, bounds, indices, record_err=True)
                all_logs.append(logs)

            # ---- stop after the first batch ----
            break

        return all_logs


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



    # ----- Usage: replace your min-jerk block with this -----
    dt_sim = 0.02  # your integrator step (0.02 s above)
    logs = run_dataset_rollouts(loader, dt_sim, bounds, indices, match_dt=True)
    print(f"Simulated {len(logs)} trajectory(ies) from the dataset.")



