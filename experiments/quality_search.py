"""Physics-rank small placement and jaw-timing variants of one accepted plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transfer_fraction(approach_count: int, frame_count: int,
                      phase_bounds: list[int]) -> np.ndarray:
    _, carry_start, release_start, _ = phase_bounds
    if not 0 <= carry_start < release_start < frame_count:
        raise ValueError("invalid transfer phase bounds")
    demonstration = np.clip((np.arange(frame_count) - carry_start)
                            / (release_start - carry_start), 0.0, 1.0)
    return np.r_[np.zeros(approach_count), demonstration]


def shifted_closing(drive: np.ndarray, grasp_start: int,
                    close_complete: int, shift_frames: int) -> np.ndarray:
    if not 0 <= grasp_start < close_complete < len(drive):
        raise ValueError("invalid gripper closure bounds")
    result = np.asarray(drive, float).copy()
    frames = np.arange(grasp_start, close_complete + 1)
    result[frames] = np.interp(frames + shift_frames, frames, drive[frames])
    return result


def perturbed_plan(source_plan: Path, destination: Path, model,
                   placement_xy_m: np.ndarray, yaw_deg: float,
                   close_shift_frames: int) -> dict:
    import h5py
    import mujoco
    from scipy.spatial.transform import Rotation

    from experiments.augment_cube import solve_translated_path, source_tool_poses

    with h5py.File(source_plan) as archive:
        approach = np.asarray(archive["q_approach"], float)
        demonstration = np.asarray(archive["q"], float)
        drive = np.asarray(archive["gripper_joint_position"], float)
        metrics = json.loads(archive["metrics_json"][()].decode())
    phases = metrics["object_grasp"]
    fraction = transfer_fraction(len(approach), len(demonstration),
                                 phases["phase_bounds"])
    original = np.concatenate((approach, demonstration))
    position_shift = np.r_[np.asarray(placement_xy_m, float), 0.0]
    if position_shift.shape != (3,) or not np.isfinite(position_shift).all():
        raise ValueError("placement shift must be finite XY")
    quality = {}
    if np.any(position_shift) or yaw_deg:
        positions, rotations = source_tool_poses(model, mujoco.MjData(model), original)
        yaw_rotations = Rotation.from_euler("z", fraction * yaw_deg, degrees=True).as_matrix()
        translated, quality = solve_translated_path(
            model, original, positions + fraction[:, None] * position_shift,
            yaw_rotations @ rotations)
    else:
        translated = original
    shifted_drive = shifted_closing(
        drive, phases["grasp_start"], phases["close_complete"], close_shift_frames)
    augmentation = {
        "placement_offset_xy_m": position_shift[:2].tolist(),
        "placement_yaw_deg": yaw_deg,
        "close_shift_frames": close_shift_frames,
        "ik": quality,
    }
    metrics["quality_augmentation"] = augmentation
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_plan) as archive, h5py.File(destination, "w") as output:
        for key in archive:
            if key != "metrics_json":
                archive.copy(key, output)
        output["q_approach"][:] = translated[:len(approach)]
        output["q"][:] = translated[len(approach):]
        output["gripper_joint_position"][:] = shifted_drive
        output.create_dataset("metrics_json", data=json.dumps(metrics).encode())
    return augmentation


def search(source: Path, model_root: Path, manifest_path: Path,
           candidate_id: int, output: Path) -> dict:
    from experiments.quality import quality_penalty
    from experiments.replay import replay
    from experiments.scene import load_scene

    manifest = json.loads(manifest_path.read_text())
    candidate = next(row for row in manifest["candidates"]
                     if row["candidate"] == candidate_id)
    if candidate["released_stack_success"] is not True:
        raise ValueError("quality search requires a physically accepted base candidate")
    original_plan = manifest_path.parent / candidate["plan"]
    red = np.asarray(candidate["red_offset_xy_m"])
    green = np.asarray(candidate["green_offset_xy_m"])
    object_model = (source / "intermediates/segmentation/"
                    "sam2.1_hiera_small/object_models.npz")
    model = load_scene(source / "plan.h5", object_model, model_root, red, green)
    variants = [("baseline", np.zeros(2), 0.0, 0)]
    variants += [(f"place_{east_mm}_{north_mm}",
                  np.array([east_mm, north_mm]) / 1000, 0.0, 0)
                 for east_mm in (0, 5, 10) for north_mm in (0, 5, 10)
                 if east_mm or north_mm]
    variants += [(f"yaw_{yaw_deg}", np.zeros(2), float(yaw_deg), 0)
                 for yaw_deg in (-20, -10, -5, 2, 5, 10, 20)]
    variants += [(f"place_5_5_yaw_{yaw_deg}", np.array([0.005, 0.005]),
                  float(yaw_deg), 0) for yaw_deg in (2, 5)]
    variants += [(f"close_{shift}", np.zeros(2), 0.0, shift)
                 for shift in (-4, -2, 2, 4)]
    rows = []
    for name, placement, yaw, shift in variants:
        destination = output / "plans" / f"{name}.h5"
        row = {"variant": name, "plan": str(destination.relative_to(output))}
        try:
            row["augmentation"] = perturbed_plan(original_plan, destination, model,
                                                   placement, yaw, shift)
            result = replay(source, model_root, 3.0, 0.10,
                            command_plan_path=destination,
                            red_offset_xy=red, green_offset_xy=green,
                            hold_commands=True)
            row["released_stack_success"] = result["released_stack_success"]
            row["quality_penalty"] = quality_penalty(result)
            row["red_contact_duration_s"] = result["support_robot_contact_duration_s"]["red"]
            row["red_contact_impulse_Ns"] = result["support_robot_normal_impulse_Ns"]["red"]
            row["green_contact_duration_s"] = result["support_robot_contact_duration_s"]["green"]
            row["center_error_mm"] = result["final_blue_green_center_error_m"] * 1000
            row["upright_error_deg"] = result["final_blue_upright_error_deg"]
            row["orientation_error_deg"] = result["final_blue_green_rotation_error_deg"]
        except ValueError as error:
            row["released_stack_success"] = False
            row["quality_penalty"] = None
            row["error"] = str(error)
        rows.append(row)
        print(json.dumps(row), flush=True)
    successful = [row for row in rows if row["quality_penalty"] is not None]
    report = {"source_episode": source.name, "candidate": candidate_id,
              "red_offset_xy_m": red.tolist(), "green_offset_xy_m": green.tolist(),
              "best_variant": min(successful, key=lambda row: row["quality_penalty"])["variant"]
              if successful else None,
              "variants": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/app/output/augmented-test-50mm/manifest.json"))
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("/app/output/quality-search-001"))
    arguments = parser.parse_args()
    result = search(arguments.source, arguments.models, arguments.manifest,
                    arguments.candidate, arguments.output)
    print(json.dumps({"best_variant": result["best_variant"],
                      "candidate": result["candidate"]}, indent=2))
