"""Capture simulator-executed state/action pairs from accepted cube plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.batch_viewer import accepted_episodes
from experiments.replay import replay


def collect(episodes_root: Path, output: Path, model_root: Path,
            time_scale: float = 3.0, pressure_rad: float = 0.10,
            control_hz: float = 50.0) -> dict:
    episodes = accepted_episodes(episodes_root / "status.jsonl", episodes_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, episode in enumerate(episodes, start=1):
        print(f"Capturing {index}/{len(episodes)}: {episode.name}", flush=True)
        destination = output / "rollouts" / f"{episode.name}.npz"
        destination.unlink(missing_ok=True)
        result = replay(episode, model_root, time_scale, pressure_rad,
                        capture_path=destination, control_hz=control_hz)
        rows.append({"episode": episode.name,
                     "released_stack_success": bool(result["released_stack_success"]),
                     "rollout": str(destination) if result["released_stack_success"] else None,
                     "red_robot_contacts": sum(result["support_robot_contact_counts"]["red"].values()),
                     "green_robot_contacts": sum(result["support_robot_contact_counts"]["green"].values())})
        print(json.dumps(rows[-1]), flush=True)
    manifest = {
        "source": str(episodes_root),
        "time_scale": time_scale,
        "grasp_pressure_offset_rad": pressure_rad,
        "control_hz": control_hz,
        "candidate_count": len(episodes),
        "successful_count": sum(row["released_stack_success"] for row in rows),
        "episodes": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path,
                        default=Path("data/experiments/recovered-retarget"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    arguments = parser.parse_args()
    result = collect(arguments.episodes, arguments.output, arguments.models)
    print(f"Captured {result['successful_count']}/{result['candidate_count']} successful rollouts.")
