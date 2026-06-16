from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad2real.config import PATHS
from cad2real.log_io import load_ros_txt_positions
from cad2real.robot_specs import (
    LEFT_HAND_PARAMS,
    NOVA2_PARAMS,
    NOVA5_PARAMS,
    RIGHT_HAND_PARAMS,
    params_to_arrays,
    resolve_dofs,
    set_group_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one collected episode in Genesis and save a render.")
    parser.add_argument("--episode", type=Path, default=PATHS.draw_dataset / "draw_t5")
    parser.add_argument("--scene", type=Path, default=PATHS.genesis_scene)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt-sim", type=float, default=0.01)
    parser.add_argument("--dt-cmd", type=float, default=0.02)
    parser.add_argument("--show-viewer", action="store_true")
    return parser.parse_args()


def resample_to_steps(t_src: np.ndarray, q_src: np.ndarray, n_steps: int) -> np.ndarray:
    if q_src.shape[0] == 0:
        raise RuntimeError("Empty trajectory")
    if q_src.shape[0] >= n_steps:
        idx = np.linspace(0, q_src.shape[0] - 1, n_steps, dtype=int)
        return q_src[idx]

    t_grid = np.linspace(t_src[0], t_src[-1], n_steps, dtype=np.float64)
    q_out = np.zeros((n_steps, q_src.shape[1]), dtype=np.float64)
    for j in range(q_src.shape[1]):
        q_out[:, j] = np.interp(t_grid, t_src, q_src[:, j])
    return q_out


def prepare_command(path: Path, group_key: str, n_steps: int, indices: dict, bounds: dict):
    if not path.exists():
        print(f"[skip] {group_key}: missing {path}")
        return None

    t_src, q_src = load_ros_txt_positions(str(path), key="position")
    q = np.deg2rad(resample_to_steps(t_src, q_src, n_steps))
    target_d = len(indices[group_key])
    if target_d == 0:
        print(f"[skip] {group_key}: no resolved DOFs")
        return None

    q_aligned = np.zeros((q.shape[0], target_d), dtype=q.dtype)
    n = min(q.shape[1], target_d)
    q_aligned[:, :n] = q[:, :n]
    lo, hi = bounds[group_key]
    q_aligned = np.minimum(np.maximum(q_aligned, lo[None, :]), hi[None, :])
    print(f"[traj] {group_key}: {path} -> {q_aligned.shape}")
    return q_aligned


def main() -> None:
    args = parse_args()

    import genesis as gs

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt_sim),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(args.scene)))
    cam = scene.add_camera(res=(960, 640), pos=(2.8, -3.5, 2.3), lookat=(0.0, 0.0, 0.5), fov=35, GUI=False)
    scene.build()

    groups = {
        "n2": NOVA2_PARAMS,
        "n5": NOVA5_PARAMS,
        "lh": LEFT_HAND_PARAMS,
        "rh": RIGHT_HAND_PARAMS,
    }
    indices = {}
    bounds = {}
    for key, params in groups.items():
        names, kp, lo, hi = params_to_arrays(params)
        dof_idx, _, missing = resolve_dofs(robot, names)
        if missing:
            print(f"[warn] {key} missing joints: {missing}")
        set_group_gains(robot, dof_idx, kp)
        indices[key] = dof_idx
        bounds[key] = (lo, hi)

    episode = args.episode
    commands = {
        "n2": prepare_command(episode / "nova2.txt", "n2", args.steps, indices, bounds),
        "n5": prepare_command(episode / "nova5.txt", "n5", args.steps, indices, bounds),
        "lh": prepare_command(episode / "left.txt", "lh", args.steps, indices, bounds),
        "rh": prepare_command(episode / "right.txt", "rh", args.steps, indices, bounds),
    }
    commands = {k: v for k, v in commands.items() if v is not None}
    if not commands:
        raise RuntimeError(f"No playable streams found in {episode}")

    robot.zero_all_dofs_velocity()
    for key, q in commands.items():
        robot.set_dofs_position(q[0], indices[key])
    for _ in range(30):
        scene.step()

    out = args.output or (PATHS.renders / f"{episode.name}_genesis.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    steps_per_cmd = max(1, int(round(args.dt_cmd / args.dt_sim)))
    cam.start_recording()
    for i in range(args.steps):
        for key, q in commands.items():
            robot.control_dofs_position(q[i], indices[key])
        for _ in range(steps_per_cmd):
            scene.step()
            cam.render()
    for _ in range(int(0.5 / args.dt_sim)):
        scene.step()
        cam.render()
    cam.stop_recording(save_to_filename=str(out), fps=int(round(1.0 / args.dt_sim)))
    print(f"[ok] saved {out.resolve()}")


if __name__ == "__main__":
    main()
