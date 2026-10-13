"""Show diverse, physics-validated object-relative cube transfers in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def diverse_successes(rows: list[dict], count: int) -> list[dict]:
    successful = [row for row in rows if row["released_stack_success"]]
    if not successful or count < 1:
        return []
    offsets = np.asarray([row["red_offset_xy_m"] + row["green_offset_xy_m"]
                          for row in successful])
    chosen = [int(np.argmax(np.linalg.norm(offsets, axis=1)))]
    remaining = set(range(len(successful))) - set(chosen)
    while remaining and len(chosen) < count:
        next_index = max(sorted(remaining), key=lambda index: min(
            np.linalg.norm(offsets[index] - offsets[previous]) for previous in chosen))
        chosen.append(next_index)
        remaining.remove(next_index)
    return [successful[index] for index in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/experiments/bc-v1/augmented/manifest.json"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--gap-s", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.max_scenes < 1 or arguments.gap_s < 0:
        parser.error("max scenes must be positive and gap nonnegative")
    report = json.loads(arguments.manifest.read_text())
    scenes = diverse_successes(report["candidates"], arguments.max_scenes)
    if not scenes:
        parser.error("no physics-validated augmented scenes in manifest")

    from experiments.viewer import watch

    for position, scene in enumerate(scenes, start=1):
        red = np.asarray(scene["red_offset_xy_m"])
        green = np.asarray(scene["green_offset_xy_m"])
        plan = arguments.manifest.parent / scene["plan"]
        if not plan.is_file():
            raise FileNotFoundError(plan)
        label = (f"{position}/{len(scenes)}  red ({red[0]*1000:+.0f}, {red[1]*1000:+.0f}) mm"
                 f"  green ({green[0]*1000:+.0f}, {green[1]*1000:+.0f}) mm")
        print(f"Playing candidate {scene['candidate']}: {label}", flush=True)
        result = watch(arguments.source, arguments.models, arguments.speed, 3.0,
                       grasp_pressure_offset_rad=0.10,
                       start_delay_s=0.0, keep_open=False, verify_result=False,
                       label=label, command_plan_path=plan,
                       red_offset_xy=red, green_offset_xy=green,
                       hold_commands=True)
        if result["playback_incomplete"]:
            print("Playback stopped when the MuJoCo window closed.", flush=True)
            return
        if position < len(scenes):
            time.sleep(arguments.gap_s)
    print(f"Finished {len(scenes)} augmented scenes.", flush=True)


if __name__ == "__main__":
    main()
