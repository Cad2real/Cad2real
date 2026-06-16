# CAD2Real

**CAD-Grounded Skill Primitives and Vision-Language Planning for Precision Assembly**

<p align="center">
  <a href="assets/"><b>Video</b></a>
</p>

> The demo video is provided in the `assets/` directory. For best compatibility, we recommend opening it with VLC media player.

CAD2Real is a hierarchical robot learning framework for precision assembly. It uses CAD assets as a lightweight geometric scaffold, learns reusable manipulation primitives from teleoperated demonstrations, and composes those primitives with vision-language planning for long-horizon, contact-rich assembly.

This repository provides the code and assets for the open-source implementation: teleoperation utilities, robot control scripts, collected episode format, dataset inspection, Genesis replay, and policy training.

## Method Overview

CAD2Real separates semantic planning from geometry-grounded execution:

1. **Asset Preparation**
   CAD models, meshes, URDF/MJCF files, and object metadata provide geometry for simulation, replay, pose-aware planning, and primitive parameterization.

2. **Skill Learning**
   Human demonstrations and teleoperated rollouts are converted into timestamped arm/hand trajectories. Primitive policies are trained from these trajectories, with simulation replay used to expand coverage over pose, contact, and execution variation.

3. **Autonomous Deployment**
   A vision-language planner selects symbolic assembly steps from the current scene and task goal. These steps are grounded into parameterized primitives such as reach, grasp, align, insert, and release, then executed by robot and hand controllers.

At runtime, the system is intended to use RGB-D observations, CAD-anchored object poses, proprioception, and execution history to decide what primitive to run next and how to parameterize it.


## Requirements

Recommended platform:

- Ubuntu/Linux
- Python 3.8-3.10 for the full teleoperation stack
- Python 3.10+ for the top-level dataset, replay, and training scripts

Core Python dependencies:

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `opencv-contrib-python`
- `tqdm`

Optional dependencies:

- `genesis-world` for simulation replay
- packages in `TeleVision/requirements.txt` for teleoperation
- ZED SDK and ZED Python API for ZED camera streaming
- RGB-D camera SDKs, CAN drivers, robot SDKs, and hand SDKs for real hardware
- external perception/planning modules if reproducing the full paper system

## Installation

Clone the repository and enter the project root:

```bash
git clone <repo-url>
cd Cad2real
```

Create a Python environment:

```bash
conda create -n cad2real python=3.10
conda activate cad2real
```

Install the core dependencies:

```bash
pip install torch torchvision numpy opencv-python opencv-contrib-python tqdm
```

Install Genesis for simulation replay:

```bash
pip install genesis-world
```

Install teleoperation dependencies if you use `TeleVision/`:

```bash
pip install -r TeleVision/requirements.txt
cd TeleVision/act/detr
pip install -e .
cd ../../..
```

If the installed PyTorch build does not match your CUDA version, reinstall `torch` and `torchvision` using the official command for your machine.

## Dataset Format

The default task dataset root is `dataset/draw`. Each episode directory is named `draw_t<N>` and contains synchronized video, event logs, and timestamped robot streams:

```text
dataset/draw/draw_t1/
├── draw_t1.mp4      # synchronized video
├── ex_log.txt       # task/event log
├── left.txt         # left hand timestamped positions
├── right.txt        # right hand timestamped positions
├── nova2.txt        # NOVA2 timestamped joint positions
└── nova5.txt        # NOVA5 timestamped joint positions
```

The stream files use repeated ROS-style blocks:

```text
seq: 0
secs: 1755682651
nsecs: 678684949
position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

`ex_log.txt` stores task steps such as `t_1`, object poses `D1`/`D2`, and a `pose_list` command sequence. Actor IDs are parsed as:

```text
-1: skip
 0: stop
 1: nova2
 2: nova5
 3: LeftHand
 4: RightHand
 5: thread/synchronization marker
```

This format corresponds to the primitive-learning stage of CAD2Real: demonstrations are represented as aligned object state, arm state, hand state, and high-level event/action annotations.

## Quick Start

Check whether the dataset episodes are complete:

```bash
python scripts/inspect_dataset.py --root dataset/draw
```

Write a JSON manifest:

```bash
python scripts/inspect_dataset.py \
  --root dataset/draw \
  --write-manifest dataset/draw/manifest.json
```

Replay one episode in Genesis:

```bash
python scripts/run_genesis_replay.py \
  --episode dataset/draw/draw_t5 \
  --backend gpu \
  --steps 1000
```

Use CPU replay if GPU initialization is unavailable:

```bash
python scripts/run_genesis_replay.py \
  --episode dataset/draw/draw_t5 \
  --backend cpu
```

Show the Genesis viewer while replaying:

```bash
python scripts/run_genesis_replay.py \
  --episode dataset/draw/draw_t5 \
  --show-viewer
```

Rendered videos are written to `renders/` by default.

## Training

Train the multimodal action policy from collected episodes:

```bash
python train.py \
  --data-root dataset/draw \
  --index-mode event \
  --epochs 100 \
  --batch-size 10 \
  --lr 3e-4 \
  --checkpoint-dir checkpoints
```

Available indexing modes:

- `event`: one sample per parsed `ex_log.txt` task step
- `frame`: one sample per video frame, mapped to the nearest task step
- `stream`: one sample per aligned trajectory stream index

By default, training uses object poses and four robot streams without decoding video. To include image tensors:

```bash
python train.py \
  --data-root dataset/draw \
  --index-mode event \
  --decode-video \
  --use-images \
  --video-backend opencv
```

Checkpoints are saved as:

```text
checkpoints/policy_epoch_XXXX.pt
```

Each checkpoint contains model weights, optimizer state, task vocabulary, and training arguments.

## Simulation Replay

`scripts/run_genesis_replay.py` loads the MJCF scene and recorded trajectories:

- `robot_urdf/scene.xml`
- `nova2.txt`
- `nova5.txt`
- `left.txt`
- `right.txt`

The script resolves DOFs by joint name, applies configured gains, resamples each stream to the requested replay length, clamps commands to joint limits, and records a camera video.

Useful options:

```bash
python scripts/run_genesis_replay.py --help
```

Common flags:

- `--scene`: path to MJCF scene
- `--output`: output video path
- `--backend`: `gpu` or `cpu`
- `--steps`: replay command steps
- `--dt-sim`: simulation step size
- `--dt-cmd`: command interval
- `--show-viewer`: open Genesis viewer

## CAD-Grounded Deployment Notes

The full CAD2Real paper pipeline uses CAD assets beyond offline replay:

- CAD geometry defines simulation assets and robot collision geometry.
- RGB-D observations are aligned to known meshes for CAD-anchored 6-DoF object state.
- A vision-language planner receives scene observations, CAD-derived metadata, and execution history.
- Planner outputs are converted into structured primitive calls such as `SKILL(part_id, pose, params)`.
- Primitive policies and motion planners execute the requested action and then re-estimate the scene state.

The current repository contains the lower-level data, replay, training, teleoperation, and robot-control pieces. External perception and VLM planning modules should be connected through the same episode/state/primitive interfaces.

## Teleoperation

The `TeleVision/` directory contains the teleoperation stack used for collecting hand and arm demonstrations. A typical setup is:

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r TeleVision/requirements.txt
cd TeleVision/act/detr
pip install -e .
cd ../../..
```

For local WebXR streaming, HTTPS certificates may be required by the browser/device. For ZED camera use, install the ZED SDK and its Python API. See `TeleVision/teleop/` and `TeleVision/teleop_l10/` for teleoperation entry points.

## Real Robot Control

Real hardware scripts live in `robot_control/`. Before running them, verify all hardware addresses and device names in the corresponding script/config file.

Values that commonly need local changes:

- robot controller IPs such as `192.168.5.1` and `192.168.5.2`
- CAN interfaces such as `can0` and `can1`
- serial devices such as `/dev/ttyUSB0`
- camera IDs and RealSense serial numbers
- task command sequences in `robot_control/config.py`

Run real robot scripts only after confirming workspace clearance, emergency stop behavior, joint limits, and device connectivity.

Example fixed-sequence ACT-style recording entry point:

```bash
python robot_control/record_episodes.py \
  --episode_idx 0 \
  --dataset_dir act_datasets
```

Example dual-arm/hand command entry point:

```bash
python robot_control/main.py
```

## Configuration

Project paths are centralized in `cad2real/config.py`:

```python
PATHS.draw_dataset   # dataset/draw
PATHS.genesis_scene  # robot_urdf/scene.xml
PATHS.renders        # renders
```

Replay joint names, gains, and control ranges are centralized in `cad2real/robot_specs.py`. Update that file if your MJCF/URDF joint names or joint limits differ.

