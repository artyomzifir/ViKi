"""Test one successful recorded command sequence on other cube scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.replay import replay


def compare(episodes_root: Path, output: Path, model_root: Path,
            source_episode: str) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    names = [row["episode"] for row in manifest["episodes"]]
    successful_sources = {row["episode"] for row in manifest["episodes"]
                          if row["released_stack_success"]}
    if source_episode not in successful_sources:
        raise ValueError("template source must be a physically successful episode")
    source_plan = episodes_root / source_episode / "plan.h5"
    rows = []
    for name in names:
        print(f"Template {source_episode} -> {name}", flush=True)
        result = replay(episodes_root / name, model_root, 3.0, 0.10,
                        command_plan_path=source_plan, hold_commands=True)
        rows.append(result)
        print(json.dumps({"episode": name,
                          "success": result["released_stack_success"],
                          "invalid_reason": result["invalid_reason"]}), flush=True)
    report = {"source_episode": source_episode, "episode_count": len(rows),
              "success_count": sum(row["released_stack_success"] is True for row in rows),
              "episodes": rows}
    (output / "template_baseline.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path,
                        default=Path("data/experiments/recovered-retarget"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--source", default="2026-10-06_11-01-57")
    arguments = parser.parse_args()
    result = compare(arguments.episodes, arguments.output,
                     arguments.models, arguments.source)
    print(f"Template success: {result['success_count']}/{result['episode_count']}")
