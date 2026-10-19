"""Simulation-only contact-gated attachment of a free object to a robot wrist."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ContactAttachment:
    model: mujoco.MjModel
    data: mujoco.MjData
    weld_name: str
    wrist_name: str
    object_name: str
    minimum_force_n: float = 0.2
    hold_s: float = 0.08
    release_opening_rad: float = 0.12
    attach_on_any_pad: bool = False

    def __post_init__(self) -> None:
        thresholds = (self.minimum_force_n, self.hold_s, self.release_opening_rad)
        if any(not np.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("attachment thresholds must be positive")
        self.weld_id = self.model.equality(self.weld_name).id
        self.wrist_id = self.model.body(self.wrist_name).id
        self.object_id = self.model.body(self.object_name).id
        self.reset()

    def reset(self) -> None:
        self.data.eq_active[self.weld_id] = False
        self.stable_s = 0.0
        self.attached = False
        self.released = False
        self.attach_time_s = None
        self.release_time_s = None
        self.attach_forces_n = None
        self.attach_pose_jump_m = None
        self.closing_command_rad = None
        self.closing_position_rad = None

    def release(self) -> None:
        if not self.attached:
            return
        self.data.eq_active[self.weld_id] = False
        self.attached = False
        self.released = True
        self.release_time_s = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)

    def observe(
        self,
        forces_n: dict[str, float],
        pad_positions: list[np.ndarray],
        non_pad_contact: bool,
        finger_command_rad: float,
        finger_position_rad: float,
        dt: float,
    ) -> None:
        if self.attached:
            opening = (
                finger_command_rad <= self.closing_command_rad - self.release_opening_rad
                and finger_position_rad <= self.closing_position_rad - self.release_opening_rad
            )
            if opening:
                self.release()
            return
        if self.released:
            return
        left, right = np.asarray(pad_positions, dtype=float)
        axis = right - left
        gap_squared = float(np.dot(axis, axis))
        object_center = self.data.xpos[self.object_id]
        between_pads = (
            gap_squared > 1e-8
            and 0.05 <= float(np.dot(object_center - left, axis) / gap_squared) <= 0.95
        )
        valid_contact = (
            max(forces_n.values()) > 0.0 if self.attach_on_any_pad else
            forces_n["left"] >= self.minimum_force_n
            and forces_n["right"] >= self.minimum_force_n
            and between_pads
            and not non_pad_contact
            and finger_command_rad >= 0.25
        )
        self.stable_s = self.stable_s + dt if valid_contact else 0.0
        if self.stable_s < self.hold_s:
            return
        wrist_rotation = self.data.xmat[self.wrist_id].reshape(3, 3)
        object_rotation = self.data.xmat[self.object_id].reshape(3, 3)
        relative_position = wrist_rotation.T @ (
            object_center - self.data.xpos[self.wrist_id]
        )
        relative_rotation = wrist_rotation.T @ object_rotation
        self.model.eq_data[self.weld_id, 3:6] = relative_position
        self.model.eq_data[self.weld_id, 6:10] = Rotation.from_matrix(
            relative_rotation,
        ).as_quat(scalar_first=True)
        before = object_center.copy()
        self.data.eq_active[self.weld_id] = True
        self.attached = True
        self.attach_time_s = float(self.data.time)
        self.attach_forces_n = dict(forces_n)
        self.closing_command_rad = float(finger_command_rad)
        self.closing_position_rad = float(finger_position_rad)
        mujoco.mj_forward(self.model, self.data)
        self.attach_pose_jump_m = float(np.linalg.norm(
            self.data.xpos[self.object_id] - before,
        ))
