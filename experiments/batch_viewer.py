"""Play quality-approved recovered cube episodes in chronological order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def accepted_episodes(status_path: Path, episodes_root: Path) -> list[Path]:
    latest = {}
    for line in status_path.read_text().splitlines():
        event = json.loads(line)
        latest[event["episode"]] = event["status"]
    episodes = [episodes_root / name for name, status in sorted(latest.items())
                if status == "accepted"]
    missing = [episode.name for episode in episodes
               if not (episode / "plan.h5").is_file()]
    if missing:
        raise FileNotFoundError(f"accepted episodes lack plans: {missing}")
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path,
                        default=Path("data/experiments/recovered-retarget"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--gap-s", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int, default=22)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--grasp-pressure-offset-rad", type=float, default=0.10)
    arguments = parser.parse_args()
    if arguments.gap_s < 0 or arguments.expected_count <= 0:
        parser.error("gap must be nonnegative and expected count must be positive")
    episodes = accepted_episodes(arguments.episodes / "status.jsonl", arguments.episodes)
    if len(episodes) != arguments.expected_count:
        parser.error(f"expected {arguments.expected_count} accepted episodes, found {len(episodes)}")

    from experiments.viewer import watch

    for index, episode in enumerate(episodes, start=1):
        print(f"Playing {index}/{len(episodes)}: {episode.name}", flush=True)
        result = watch(episode, arguments.models, arguments.speed,
                       arguments.time_scale,
                       grasp_pressure_offset_rad=arguments.grasp_pressure_offset_rad,
                       start_delay_s=0.0, keep_open=False,
                       verify_result=False, label=f"{index}/{len(episodes)}")
        if result["playback_incomplete"]:
            print("Playback stopped when the MuJoCo window closed.", flush=True)
            return
        if index < len(episodes):
            time.sleep(arguments.gap_s)
    print(f"Finished {len(episodes)} episodes.", flush=True)


if __name__ == "__main__":
    main()
