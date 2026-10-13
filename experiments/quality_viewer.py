"""Show before/after physics-approved cube-placement augmentations in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def improved_pairs(report: dict, count: int) -> list[tuple[dict, dict, dict]]:
    pairs = []
    for episode in report["episodes"]:
        baseline = next(row for row in episode["variants"]
                        if row["variant"] == "baseline")
        selected = next(row for row in episode["variants"]
                        if row["variant"] == episode["best_variant"])
        if (baseline["quality_penalty"] is not None
                and selected["quality_penalty"] is not None
                and selected["quality_penalty"] < baseline["quality_penalty"]):
            pairs.append((episode, baseline, selected))
    pairs.sort(key=lambda pair: pair[1]["quality_penalty"]
               - pair[2]["quality_penalty"], reverse=True)
    return pairs[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/experiments/bc-v1/quality-batch-001/manifest.json"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--gap-s", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.max_pairs < 1 or arguments.gap_s < 0:
        parser.error("max pairs must be positive and gap nonnegative")
    pairs = improved_pairs(json.loads(arguments.manifest.read_text()),
                           arguments.max_pairs)
    if not pairs:
        parser.error("no improved, physically successful plan pairs")

    from experiments.viewer import watch

    for index, (episode, baseline, selected) in enumerate(pairs, start=1):
        red = np.asarray(episode["red_offset_xy_m"])
        green = np.asarray(episode["green_offset_xy_m"])
        for stage, variant in (("BEFORE", baseline), ("AFTER", selected)):
            label = (f"{stage} {index}/{len(pairs)} candidate {episode['candidate']}"
                     f"  center {variant['center_error_mm']:.1f} mm"
                     f"  rotation {variant['orientation_error_deg']:.1f} deg")
            print(f"Playing {label}", flush=True)
            result = watch(arguments.source, arguments.models, 1.0, 3.0,
                           grasp_pressure_offset_rad=0.10, start_delay_s=0.0,
                           keep_open=False, verify_result=False, label=label,
                           command_plan_path=arguments.manifest.parent / variant["plan"],
                           red_offset_xy=red, green_offset_xy=green,
                           hold_commands=True)
            if result["playback_incomplete"]:
                return
            time.sleep(arguments.gap_s)


if __name__ == "__main__":
    main()
