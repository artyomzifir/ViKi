"""Select physics-successful cube plans by support-contact and placement penalty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    ("baseline", (0.0, 0.0), 0.0),
    ("place_5_0", (0.005, 0.0), 0.0),
    ("place_5_5", (0.005, 0.005), 0.0),
    ("place_5_5_yaw_5", (0.005, 0.005), 5.0),
)


def best_variant(rows: list[dict]) -> dict | None:
    successful = [row for row in rows if row["quality_penalty"] is not None]
    return min(successful, key=lambda row: row["quality_penalty"]) if successful else None


def generate_quality_batch(source: Path, model_root: Path, manifest_path: Path,
                           output: Path) -> dict:
    from experiments.quality import quality_penalty
    from experiments.quality_search import perturbed_plan
    from experiments.replay import replay
    from experiments.scene import load_scene

    manifest = json.loads(manifest_path.read_text())
    object_model = (source / "intermediates/segmentation/"
                    "sam2.1_hiera_small/object_models.npz")
    episodes = []
    for candidate in manifest["candidates"]:
        if candidate["released_stack_success"] is not True:
            continue
        candidate_id = candidate["candidate"]
        source_plan = manifest_path.parent / candidate["plan"]
        red = np.asarray(candidate["red_offset_xy_m"])
        green = np.asarray(candidate["green_offset_xy_m"])
        model = load_scene(source / "plan.h5", object_model, model_root, red, green)
        variants = []
        for name, placement, yaw in VARIANTS:
            destination = output / "plans" / f"candidate_{candidate_id:03d}_{name}.h5"
            row = {"variant": name, "plan": str(destination.relative_to(output))}
            try:
                row["augmentation"] = perturbed_plan(
                    source_plan, destination, model, np.asarray(placement), yaw, 0)
                result = replay(source, model_root, 3.0, 0.10,
                                command_plan_path=destination,
                                red_offset_xy=red, green_offset_xy=green,
                                hold_commands=True)
                row["released_stack_success"] = result["released_stack_success"]
                row["quality_penalty"] = quality_penalty(result)
                row["red_contact_impulse_Ns"] = result["support_robot_normal_impulse_Ns"]["red"]
                row["green_contact_impulse_Ns"] = result["support_robot_normal_impulse_Ns"]["green"]
                row["center_error_mm"] = result["final_blue_green_center_error_m"] * 1000
                row["upright_error_deg"] = result["final_blue_upright_error_deg"]
                row["orientation_error_deg"] = result["final_blue_green_rotation_error_deg"]
            except ValueError as error:
                row["released_stack_success"] = False
                row["quality_penalty"] = None
                row["error"] = str(error)
            variants.append(row)
        selected = best_variant(variants)
        episode = {"candidate": candidate_id,
                   "red_offset_xy_m": red.tolist(),
                   "green_offset_xy_m": green.tolist(),
                   "best_variant": selected["variant"] if selected else None,
                   "best_plan": selected["plan"] if selected else None,
                   "variants": variants}
        episodes.append(episode)
        print(json.dumps({"candidate": candidate_id,
                          "best_variant": episode["best_variant"],
                          "baseline_penalty": variants[0]["quality_penalty"],
                          "best_penalty": selected["quality_penalty"] if selected else None}),
              flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps({
            "source_episode": source.name,
            "source_manifest": str(manifest_path),
            "episode_count": len(episodes), "episodes": episodes,
        }, indent=2))
    return {"source_episode": source.name,
            "source_manifest": str(manifest_path),
            "episode_count": len(episodes), "episodes": episodes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/app/output/augmented-test-50mm/manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("/app/output/quality-batch-001"))
    arguments = parser.parse_args()
    result = generate_quality_batch(arguments.source, arguments.models,
                                    arguments.manifest, arguments.output)
    print(f"Quality-ranked {result['episode_count']} accepted layouts.")
