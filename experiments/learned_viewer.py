"""Play held-out successful learned-policy layouts in the native MuJoCo GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def selected_test_rows(report: dict, count: int) -> list[dict]:
    rows = [row for row in report["layouts"]
            if row["split"] == "test" and row["learned"]["released_stack_success"] is True]
    rows.sort(key=lambda row: (
        row["fixed_template"]["released_stack_success"] is True,
        -np.linalg.norm(row["red_offset_xy_m"] + row["green_offset_xy_m"])))
    return rows[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("data/experiments/bc-v1/layout_bc_v1/evaluation.json"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/experiments/bc-v1/layout_bc_v1/layout_bc_v1.pt"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--gap-s", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.max_scenes < 1 or arguments.gap_s < 0:
        parser.error("max scenes must be positive and gap nonnegative")
    rows = selected_test_rows(json.loads(arguments.results.read_text()),
                              arguments.max_scenes)
    if not rows:
        parser.error("no physically successful held-out learned-policy scenes")

    from experiments.layout_bc import LayoutBCPolicy
    from experiments.viewer import watch

    for index, row in enumerate(rows, start=1):
        red = np.asarray(row["red_offset_xy_m"])
        green = np.asarray(row["green_offset_xy_m"])
        label = (f"LEARNED {index}/{len(rows)}  {row['layout_id']}"
                 f"  red ({red[0]*1000:+.0f}, {red[1]*1000:+.0f}) mm"
                 f"  green ({green[0]*1000:+.0f}, {green[1]*1000:+.0f}) mm")
        print(f"Playing {label}", flush=True)
        result = watch(arguments.source, arguments.models, 1.0, 3.0,
                       grasp_pressure_offset_rad=0.10,
                       start_delay_s=0.0, keep_open=False, verify_result=False,
                       label=label, red_offset_xy=red, green_offset_xy=green,
                       hold_commands=True,
                       policy=LayoutBCPolicy(arguments.checkpoint,
                                             arguments.source / "plan.h5"))
        if result["playback_incomplete"]:
            return
        if index < len(rows):
            time.sleep(arguments.gap_s)


if __name__ == "__main__":
    main()
