"""Generate object-relative UR10 plan candidates for shifted cube layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def path_offsets(approach_count: int, frame_count: int,
                 phase_bounds: list[int], red_xy: np.ndarray,
                 green_xy: np.ndarray) -> np.ndarray:
    _, carry_start, release_start, _ = phase_bounds
    if not 0 < carry_start < release_start < frame_count or approach_count < 2:
        raise ValueError("invalid approach or cube transfer phase bounds")
    red = np.r_[red_xy, 0.0]
    green = np.r_[green_xy, 0.0]
    approach = np.linspace(0.0, 1.0, approach_count)[:, None] * red
    fraction = np.clip(
        (np.arange(frame_count) - carry_start) / (release_start - carry_start),
        0.0, 1.0,
    )[:, None]
    transfer = (1.0 - fraction) * red + fraction * green
    return np.concatenate((approach, transfer))


def source_tool_poses(model, data, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    from experiments.scene import ARM_JOINTS

    qpos = np.asarray([model.jnt_qposadr[model.joint(name).id]
                       for name in ARM_JOINTS], dtype=int)
    body = model.body("wrist_3_link").id
    positions = []
    rotations = []
    for joints in path:
        data.qpos[qpos] = joints
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body].copy())
        rotations.append(data.xmat[body].reshape(3, 3).copy())
    return np.asarray(positions), np.asarray(rotations)


def solve_translated_path(model, original: np.ndarray, target_positions: np.ndarray,
                          target_rotations: np.ndarray) -> tuple[np.ndarray, dict]:
    import mujoco
    from scipy.spatial.transform import Rotation

    from experiments.scene import ARM_JOINTS

    data = mujoco.MjData(model)
    body = model.body("wrist_3_link").id
    qpos = np.asarray([model.jnt_qposadr[model.joint(name).id]
                       for name in ARM_JOINTS], dtype=int)
    dof = np.asarray([model.jnt_dofadr[model.joint(name).id]
                      for name in ARM_JOINTS], dtype=int)
    limits = np.asarray([model.jnt_range[model.joint(name).id]
                         for name in ARM_JOINTS], dtype=float)
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    translated = np.empty_like(original)
    previous_correction = np.zeros(6)
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_correction = 0.0
    for index, source_joints in enumerate(original):
        joints = np.clip(source_joints + previous_correction,
                         limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        for _ in range(14):
            data.qpos[qpos] = joints
            mujoco.mj_forward(model, data)
            position_error = target_positions[index] - data.xpos[body]
            rotation_error = Rotation.from_matrix(
                target_rotations[index] @ data.xmat[body].reshape(3, 3).T
            ).as_rotvec()
            if (np.linalg.norm(position_error) < 0.0008
                    and np.linalg.norm(rotation_error) < np.deg2rad(0.5)):
                break
            mujoco.mj_jacBody(model, data, jacobian_position,
                              jacobian_rotation, body)
            jacobian = np.vstack((jacobian_position[:, dof],
                                  0.15 * jacobian_rotation[:, dof]))
            error = np.r_[position_error, 0.15 * rotation_error]
            damping = 0.003
            correction = source_joints - joints
            step = np.linalg.solve(
                jacobian.T @ jacobian + damping**2 * np.eye(6),
                jacobian.T @ error + damping**2 * correction,
            )
            joints = np.clip(joints + np.clip(step, -0.12, 0.12),
                             limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        data.qpos[qpos] = joints
        mujoco.mj_forward(model, data)
        position_error = np.linalg.norm(target_positions[index] - data.xpos[body])
        rotation_error = np.linalg.norm(Rotation.from_matrix(
            target_rotations[index] @ data.xmat[body].reshape(3, 3).T
        ).as_rotvec())
        max_position_error = max(max_position_error, float(position_error))
        max_orientation_error = max(max_orientation_error, float(rotation_error))
        max_correction = max(max_correction,
                             float(np.max(np.abs(joints - source_joints))))
        if position_error > 0.003 or rotation_error > np.deg2rad(3):
            raise ValueError(f"wrist IK misses frame {index}: "
                             f"{position_error * 1000:.1f} mm, "
                             f"{np.rad2deg(rotation_error):.1f} deg")
        translated[index] = joints
        previous_correction = joints - source_joints
    return translated, {
        "max_wrist_position_error_mm": max_position_error * 1000,
        "max_wrist_orientation_error_deg": np.rad2deg(max_orientation_error),
        "max_joint_correction_rad": max_correction,
    }


def generate(source: Path, destination: Path, model_root: Path,
             red_offset_xy: np.ndarray, green_offset_xy: np.ndarray,
             model=None) -> dict:
    import h5py
    import mujoco

    from experiments.scene import load_scene

    source_plan = source / "plan.h5"
    with h5py.File(source_plan) as plan:
        approach = np.asarray(plan["q_approach"], float)
        demonstration = np.asarray(plan["q"], float)
        metrics = json.loads(plan["metrics_json"][()].decode())
    if model is None:
        object_model = (source / "intermediates/segmentation/"
                        "sam2.1_hiera_small/object_models.npz")
        model = load_scene(source_plan, object_model, model_root)
    original = np.concatenate((approach, demonstration))
    offsets = path_offsets(len(approach), len(demonstration),
                           metrics["object_grasp"]["phase_bounds"],
                           red_offset_xy, green_offset_xy)
    data = mujoco.MjData(model)
    source_positions, source_rotations = source_tool_poses(model, data, original)
    translated, quality = solve_translated_path(
        model, original, source_positions + offsets, source_rotations)
    metrics["augmentation"] = {
        "source_episode": source.name,
        "red_offset_xy_m": red_offset_xy.tolist(),
        "green_offset_xy_m": green_offset_xy.tolist(),
        **quality,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_plan) as source_archive, h5py.File(destination, "w") as output:
        for key in ("q_approach", "q", "gripper_joint_position", "timestamps", "dt"):
            source_archive.copy(key, output)
        output["q_approach"][:] = translated[:len(approach)]
        output["q"][:] = translated[len(approach):]
        output.create_dataset("metrics_json", data=json.dumps(metrics).encode())
    return quality


def search(source: Path, output: Path, model_root: Path,
           candidate_count: int, offset_mm: float, seed: int) -> dict:
    from experiments.quality import quality_penalty
    from experiments.replay import replay
    from experiments.scene import load_scene

    if candidate_count < 1 or not 0 < offset_mm <= 50:
        raise ValueError("candidate count must be positive and offset range <= 50 mm")
    random = np.random.default_rng(seed)
    object_model = (source / "intermediates/segmentation/"
                    "sam2.1_hiera_small/object_models.npz")
    model = load_scene(source / "plan.h5", object_model, model_root)
    rows = []
    for index in range(candidate_count):
        red = random.uniform(-offset_mm, offset_mm, 2) / 1000
        green = random.uniform(-offset_mm, offset_mm, 2) / 1000
        destination = output / "plans" / f"candidate_{index:03d}.h5"
        row = {"candidate": index, "red_offset_xy_m": red.tolist(),
               "green_offset_xy_m": green.tolist(),
               "plan": str(destination.relative_to(output))}
        try:
            row["ik"] = generate(source, destination, model_root, red, green, model)
            result = replay(source, model_root, 3.0, 0.10,
                            command_plan_path=destination,
                            red_offset_xy=red, green_offset_xy=green,
                            hold_commands=True)
            row["released_stack_success"] = bool(result["released_stack_success"])
            row["quality_penalty"] = quality_penalty(result)
            row["red_contact_impulse_Ns"] = result["support_robot_normal_impulse_Ns"]["red"]
            row["green_contact_impulse_Ns"] = result["support_robot_normal_impulse_Ns"]["green"]
            row["center_error_mm"] = result["final_blue_green_center_error_m"] * 1000
            row["orientation_error_deg"] = result["final_blue_green_rotation_error_deg"]
            row["max_airborne_grasp_slip_mm"] = result["max_airborne_grasp_slip_m"] * 1000
            row["red_robot_contacts"] = sum(result["support_robot_contact_counts"]["red"].values())
            row["green_robot_contacts"] = sum(result["support_robot_contact_counts"]["green"].values())
        except ValueError as error:
            row["released_stack_success"] = False
            row["quality_penalty"] = None
            row["error"] = str(error)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "source_episode": source.name,
        "candidate_count": candidate_count,
        "offset_mm_per_axis": offset_mm,
        "seed": seed,
        "success_count": sum(row["released_stack_success"] for row in rows),
        "candidates": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--output", type=Path, default=Path("/app/output/augmented"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--offset-mm", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    result = search(arguments.source, arguments.output, arguments.models,
                    arguments.count, arguments.offset_mm, arguments.seed)
    print(f"Physics-valid variants: {result['success_count']}/{result['candidate_count']}")
