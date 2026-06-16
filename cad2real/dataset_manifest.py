from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_EPISODE_FILES = ("left.txt", "right.txt", "nova2.txt", "nova5.txt")
EX_LOG_ALIASES = ("ex_log.txt", "ex_log .txt")


@dataclass
class EpisodeManifest:
    name: str
    path: str
    video: Optional[str]
    ex_log: Optional[str]
    streams: Dict[str, Optional[str]]
    missing: List[str]


def discover_episodes(dataset_root: Path) -> List[Path]:
    return sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("draw_t"))


def build_manifest(dataset_root: Path) -> List[EpisodeManifest]:
    dataset_root = dataset_root.resolve()
    episodes: List[EpisodeManifest] = []
    for episode_dir in discover_episodes(dataset_root):
        missing: List[str] = []
        video = episode_dir / f"{episode_dir.name}.mp4"
        ex_log = next((episode_dir / name for name in EX_LOG_ALIASES if (episode_dir / name).exists()), episode_dir / "ex_log.txt")
        if not video.exists():
            missing.append(video.name)
        if not ex_log.exists():
            missing.append("ex_log.txt")

        streams: Dict[str, Optional[str]] = {}
        for filename in REQUIRED_EPISODE_FILES:
            path = episode_dir / filename
            stream_name = filename[:-4] if filename.endswith(".txt") else filename
            streams[stream_name] = str(path) if path.exists() else None
            if not path.exists():
                missing.append(filename)

        episodes.append(
            EpisodeManifest(
                name=episode_dir.name,
                path=str(episode_dir),
                video=str(video) if video.exists() else None,
                ex_log=str(ex_log) if ex_log.exists() else None,
                streams=streams,
                missing=missing,
            )
        )
    return episodes


def manifest_to_dict(episodes: Iterable[EpisodeManifest]) -> dict:
    episode_list = [asdict(ep) for ep in episodes]
    return {
        "num_episodes": len(episode_list),
        "num_complete": sum(1 for ep in episode_list if not ep["missing"]),
        "episodes": episode_list,
    }


def write_manifest(dataset_root: Path, output_path: Path) -> dict:
    manifest = manifest_to_dict(build_manifest(dataset_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
