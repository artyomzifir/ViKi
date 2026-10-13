"""Replay every available isolated cube plan under the same physics settings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.replay import replay


def batch_replay(episodes_root: Path, model_root: Path,
                 time_scale: float, grasp_pressure_offset_rad: float) -> dict:
    rows = []
    for episode in sorted(episodes_root.iterdir()):
        if not episode.is_dir() or not (episode / "plan.h5").is_file():
            continue
        try:
            rows.append(replay(episode, model_root, time_scale,
                               grasp_pressure_offset_rad))
        except Exception as error:
            rows.append({"episode": episode.name, "error": repr(error)})
    return {
        "episodes_root": str(episodes_root),
        "plan_count": len(rows),
        "outcomes": dict(Counter(
            "error" if "error" in row else
            "success" if row["released_stack_success"] else "failure"
            for row in rows
        )),
        "episodes": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--grasp-pressure-offset-rad", type=float, default=0.10)
    arguments = parser.parse_args()
    print(json.dumps(batch_replay(
        arguments.episodes_root, arguments.models, arguments.time_scale,
        arguments.grasp_pressure_offset_rad,
    ), indent=2))
