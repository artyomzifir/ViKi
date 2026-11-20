"""Simulation-only pause and bilateral-force gate before object lift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def insert_grasp_dwell(
    command_times: np.ndarray,
    command_joints: np.ndarray,
    command_drive: np.ndarray,
    close_frame: int,
    frame_start: int,
    dwell_s: float,
    hold_drive: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not 0 < dwell_s <= 2 or not np.isfinite(dwell_s):
        raise ValueError("grasp dwell must be finite and in (0, 2] seconds")
    command_index = 1 + close_frame - frame_start
    if not 1 <= command_index < len(command_times) - 1:
        raise ValueError("grasp close frame is outside the replay window")
    if hold_drive is not None:
        if not np.isfinite(hold_drive) or not 0 <= hold_drive <= 0.8:
            raise ValueError("hold drive must be finite and in [0, 0.8] radians")
        command_drive = command_drive.copy()
        command_drive[command_index] = hold_drive
    dwell_start = float(command_times[command_index])
    dwell_end = dwell_start + dwell_s
    command_times = np.insert(command_times, command_index + 1, dwell_end)
    command_times[command_index + 2:] += dwell_s
    command_joints = np.insert(
        command_joints, command_index + 1, command_joints[command_index], axis=0,
    )
    command_drive = np.insert(
        command_drive, command_index + 1, command_drive[command_index],
    )
    return command_times, command_joints, command_drive, dwell_start, dwell_end


@dataclass
class BilateralForceGate:
    minimum_force_n: float
    hold_s: float
    stable_s: float = 0.0

    def __post_init__(self) -> None:
        if (not np.isfinite(self.minimum_force_n) or self.minimum_force_n <= 0
                or not np.isfinite(self.hold_s) or self.hold_s <= 0):
            raise ValueError("force and hold duration must be positive and finite")

    def observe(self, left_n: float, right_n: float, dt: float) -> bool:
        if (left_n >= self.minimum_force_n
                and right_n >= self.minimum_force_n):
            self.stable_s += dt
        else:
            self.stable_s = 0.0
        return self.stable_s >= self.hold_s
