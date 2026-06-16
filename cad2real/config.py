from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    dataset_root: Path = PROJECT_ROOT / "dataset"
    draw_dataset: Path = PROJECT_ROOT / "dataset" / "draw"
    robot_urdf: Path = PROJECT_ROOT / "robot_urdf"
    genesis_scene: Path = PROJECT_ROOT / "robot_urdf" / "scene.xml"
    renders: Path = PROJECT_ROOT / "renders"
    television: Path = PROJECT_ROOT / "TeleVision"
    teleop: Path = PROJECT_ROOT / "TeleVision" / "teleop"
    robot_control: Path = PROJECT_ROOT / "robot_control"


PATHS = ProjectPaths()

