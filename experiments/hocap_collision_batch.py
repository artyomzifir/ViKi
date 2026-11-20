"""Re-run the four G10 object-conditioned clips with scene collision barriers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py

WINDOWS = ((2, 0, 190), (3, 185, 420), (4, 420, 600), (1, 635, 831))
DEFAULT_PLANS = {
    2: Path("data/_retarget_diagnostics/hocap_g10_full_object_2/plan.h5"),
    3: Path("data/_retarget_diagnostics/hocap_g10_full_object_3/plan.h5"),
    4: Path("data/_retarget_diagnostics/hocap_g10_4_object_pilot/plan.h5"),
    1: Path("data/_retarget_diagnostics/hocap_g10_full_object_1/plan.h5"),
}
from viki.contracts import Episode
from viki.retarget.run import retarget_episode


def _options(plan_path: Path, track_path: Path, object_id: int) -> dict:
    with h5py.File(plan_path) as plan:
        archived = json.loads(plan["config_json"][()].decode())
    keys = (
        "robot", "reference_policy", "base_position", "base_rpy_deg",
        "table_plane_z_m", "hand_to_ee_translation", "hand_to_ee_rpy_deg",
        "gripper", "target_position_anchor", "pose_source", "object_grasp",
        "object_cube_side_mm", "object_contact_overlap_mm",
        "object_insertion_depth_mm", "approach_sec", "velocity_limit_scale",
        "collision_pairs", "collision_min_distance_m", "sequential_baseline",
    )
    options = {key: archived[key] for key in keys if key in archived}
    options.update(archived["solver"])
    options.update({f"w_{key}": value for key, value in archived["weights"].items()})
    adapter = archived["adapter"]
    options.update({
        "adapter_translation_mm": [1000 * value for value in adapter["translation_m"]],
        "adapter_rpy_deg": adapter["rpy_deg"],
        "adapter_radius_mm": 1000 * adapter["radius_m"],
        "object_track_path": str(track_path),
        "object_id": object_id,
    })
    return options


def run(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for object_id, first, last in WINDOWS:
        source = DEFAULT_PLANS[object_id].parent
        destination = output_root / f"G10_{object_id}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("cln.npz", "meta.json"):
            target = destination / name
            if not target.exists():
                if name == "cln.npz":
                    os.link(source / name, target)
                else:
                    shutil.copy2(source / name, target)
        episode = Episode(destination)
        options = _options(source / "plan.h5", source / "object_track.npz", object_id)
        started = time.monotonic()
        print(f"G10_{object_id}: retarget {last - first} frames", flush=True)

        def progress(**fields):
            if fields.get("stage") == "retarget_solve":
                print(
                    f"G10_{object_id}: iteration {fields['frame']}/{fields['total']}, "
                    f"objective={fields['objective']:.5g}, "
                    f"step={fields['max_step']:.5g}",
                    flush=True,
                )

        try:
            retarget_episode(episode, options=options, report=progress)
            with h5py.File(episode.plan_h5) as plan:
                metrics = json.loads(plan["metrics_json"][()].decode())
            results[str(object_id)] = {
                "status": "complete", "plan": str(episode.plan_h5),
                "position_rmse_mm": metrics["position_rmse_mm"],
                "collision": metrics["object_collision"],
                "elapsed_s": time.monotonic() - started,
            }
        except Exception as exc:
            results[str(object_id)] = {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.monotonic() - started,
            }
        (output_root / "batch_status.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"G10_{object_id}: {results[str(object_id)]}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    statuses = run(arguments.output_root)
    if any(result["status"] != "complete" for result in statuses.values()):
        raise SystemExit(1)
