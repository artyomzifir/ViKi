"""Pure geometric placement metrics for the cube-transfer simulator."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np


def cube_symmetries() -> tuple[np.ndarray, ...]:
    rotations = []
    for order in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            rotation[np.arange(3), order] = signs
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return tuple(rotations)


CUBE_SYMMETRIES = cube_symmetries()


def cube_pose_errors(blue_xyz: np.ndarray, green_xyz: np.ndarray,
                     blue_rotation: np.ndarray, green_rotation: np.ndarray) -> dict:
    blue_xyz = np.asarray(blue_xyz, float)
    green_xyz = np.asarray(green_xyz, float)
    blue_rotation = np.asarray(blue_rotation, float)
    green_rotation = np.asarray(green_rotation, float)
    if (blue_xyz.shape != (3,) or green_xyz.shape != (3,)
            or blue_rotation.shape != (3, 3) or green_rotation.shape != (3, 3)
            or not all(np.isfinite(value).all() for value in
                       (blue_xyz, green_xyz, blue_rotation, green_rotation))):
        raise ValueError("cube pose metrics need finite 3-D positions and rotations")
    relative = green_rotation.T @ blue_rotation
    relative_errors = [np.arccos(np.clip((np.trace(relative @ symmetry) - 1) / 2,
                                           -1.0, 1.0)) for symmetry in CUBE_SYMMETRIES]
    upright_cosine = np.max(np.abs(blue_rotation[2, :]))
    return {
        "final_blue_green_center_error_m": float(np.linalg.norm(
            blue_xyz[:2] - green_xyz[:2])),
        "final_blue_green_height_error_m": float(abs(
            blue_xyz[2] - green_xyz[2] - 0.04)),
        "final_blue_upright_error_deg": float(np.rad2deg(
            np.arccos(np.clip(upright_cosine, -1.0, 1.0)))),
        "final_blue_green_rotation_error_deg": float(np.rad2deg(min(relative_errors))),
    }


def quality_penalty(result: dict) -> float | None:
    if result["released_stack_success"] is not True:
        return None
    impulse = result["support_robot_normal_impulse_Ns"]
    return float(
        2.0 * (impulse["red"] + impulse["green"])
        + result["final_blue_green_center_error_m"] / 0.005
        + result["final_blue_upright_error_deg"] / 5.0
        + result["final_blue_green_rotation_error_deg"] / 10.0
    )
