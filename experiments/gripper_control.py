"""Simulator-only closing pressure layered over a geometric gripper plan."""

from __future__ import annotations

import numpy as np


def pressure_commands(drive: np.ndarray, object_grasp: dict | None,
                      offset_rad: float, fully_close: bool = False) -> np.ndarray:
    command = np.asarray(drive, dtype=float)
    if command.ndim != 1 or not np.isfinite(command).all():
        raise ValueError("gripper drive must be a finite trajectory")
    if not np.isfinite(offset_rad) or not 0 <= offset_rad <= 0.2:
        raise ValueError("pressure offset must be between 0 and 0.2 rad")
    if fully_close and offset_rad != 0:
        raise ValueError("full closing and pressure offset are mutually exclusive")
    if offset_rad == 0 and not fully_close:
        return command.copy()
    supported_modes = {"experimental_cube_face_transfer_v2",
                       "experimental_cube_face_transfer_v3",
                       "experimental_tracked_side_grasp_v1"}
    if not object_grasp or object_grasp.get("mode") not in supported_modes:
        raise ValueError("pressure control requires a supported phased grasp plan")
    grasp_start = int(object_grasp["grasp_start"])
    close_complete = int(object_grasp["close_complete"])
    _, carry_start, release_start, open_end = object_grasp["phase_bounds"]
    if not 0 <= grasp_start < close_complete < carry_start < release_start < open_end < len(command):
        raise ValueError("invalid grasp phase bounds")
    release_end = min(open_end, release_start + 15)
    if fully_close:
        closed = command.copy()
        closed[grasp_start:close_complete + 1] = np.linspace(
            command[grasp_start], 0.8, close_complete - grasp_start + 1,
        )
        closed[close_complete:release_start + 1] = 0.8
        closed[release_start:release_end + 1] = np.linspace(
            0.8, command[release_end], release_end - release_start + 1,
        )
        return closed
    pressure = np.zeros_like(command)
    pressure[grasp_start:close_complete + 1] = np.linspace(
        0.0, offset_rad, close_complete - grasp_start + 1)
    pressure[close_complete:release_start] = offset_rad
    pressure[release_start:release_end + 1] = np.linspace(
        offset_rad, 0.0, release_end - release_start + 1)
    return np.clip(command + pressure, 0.0, 0.8)
