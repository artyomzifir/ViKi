"""Held-out placement perturbations for the fixed-template baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.replay import replay


def evaluate(source: Path, output: Path, model_root: Path,
             count: int = 25, offset_mm: float = 20.0, seed: int = 2026) -> dict:
    if count <= 0 or not 0 < offset_mm <= 50:
        raise ValueError("count must be positive and offset range must be within 50 mm")
    random = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        red = random.uniform(-offset_mm, offset_mm, 2) / 1000
        green = random.uniform(-offset_mm, offset_mm, 2) / 1000
        result = replay(source, model_root, 3.0, 0.10,
                        red_offset_xy=red, green_offset_xy=green,
                        hold_commands=True)
        rows.append(result)
        print(json.dumps({"trial": index, "success": result["released_stack_success"],
                          "red_mm": (red * 1000).round(1).tolist(),
                          "green_mm": (green * 1000).round(1).tolist()}), flush=True)
    report = {
        "source_episode": source.name,
        "seed": seed,
        "offset_mm_per_axis": offset_mm,
        "count": count,
        "success_count": sum(row["released_stack_success"] is True for row in rows),
        "episodes": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"template_perturbation_{int(offset_mm)}mm.json").write_text(
        json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--offset-mm", type=float, default=20)
    arguments = parser.parse_args()
    result = evaluate(arguments.source, arguments.output, arguments.models,
                      arguments.count, arguments.offset_mm)
    print(f"Perturbed template success: {result['success_count']}/{result['count']}")
