from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad2real.config import PATHS
from cad2real.dataset_manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a cad2real dataset folder.")
    parser.add_argument("--root", type=Path, default=PATHS.draw_dataset, help="Dataset task folder, e.g. dataset/draw.")
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <root>/manifest.json when passed without a value is not supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = build_manifest(args.root)
    complete = [ep for ep in episodes if not ep.missing]

    print(f"Dataset root: {args.root.resolve()}")
    print(f"Episodes: {len(episodes)} total, {len(complete)} complete")
    for ep in episodes:
        status = "ok" if not ep.missing else "missing " + ", ".join(ep.missing)
        print(f"- {ep.name}: {status}")

    if args.write_manifest:
        manifest = write_manifest(args.root, args.write_manifest)
        print(f"Wrote manifest: {args.write_manifest.resolve()}")
        print(f"Complete episodes: {manifest['num_complete']}/{manifest['num_episodes']}")


if __name__ == "__main__":
    main()
