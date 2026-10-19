"""Tracked-object collision constraints for offline object-conditioned IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget.frames import base_rotation, calibration_to_robot


@dataclass(frozen=True)
class ObjectBox:
    identifier: int
    size: np.ndarray
    center_local: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray


def load_object_boxes(
    path: Path,
    selected_id: int,
    frame_count: int,
    base_position: tuple[float, float, float],
    base_rpy_deg: tuple[float, float, float],
) -> tuple[ObjectBox, tuple[ObjectBox, ...]]:
    """Fit enclosing oriented boxes to canonical clouds in robot coordinates."""
    with np.load(path, allow_pickle=False) as models:
        if str(models["schema"]) != OBJECT_MODEL_SCHEMA:
            raise ValueError("unsupported object-model schema")
        identifiers = np.asarray(models["object_ids"], dtype=int)
        offsets = np.asarray(models["model_offsets"], dtype=int)
        surface = np.asarray(models["model_xyz_object"], dtype=float)
        positions = np.asarray(models["translation_world"], dtype=float)
        rotations = np.asarray(models["rotation_world_object"], dtype=float)
        confidence = np.asarray(models["track_confidence"], dtype=float)
    count = len(identifiers)
    if (count == 0 or len(set(identifiers)) != count
            or np.count_nonzero(identifiers == selected_id) != 1
            or offsets.shape != (count + 1,) or offsets[0] != 0
            or offsets[-1] != len(surface) or np.any(np.diff(offsets) < 100)
            or surface.shape[1:] != (3,)
            or positions.shape != (frame_count, count, 3)
            or rotations.shape != (frame_count, count, 3, 3)
            or confidence.shape != (frame_count, count)
            or not all(np.isfinite(array).all() for array in
                       (surface, positions, rotations, confidence))
            or np.min(np.quantile(confidence, 0.1, axis=0)) < 0.5
            or np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))) > 0.05
            or np.min(np.linalg.det(rotations)) < 0.95):
        raise ValueError("object collision requires complete, confident aligned tracks")
    world_to_robot_rotation = base_rotation(base_rpy_deg).T
    boxes = []
    for index, identifier in enumerate(identifiers):
        points = surface[offsets[index]:offsets[index + 1]]
        lower, upper = points.min(axis=0), points.max(axis=0)
        boxes.append(ObjectBox(
            int(identifier), upper - lower, (lower + upper) / 2,
            calibration_to_robot(positions[:, index], base_position, base_rpy_deg),
            world_to_robot_rotation @ rotations[:, index],
        ))
    selected = next(box for box in boxes if box.identifier == selected_id)
    return selected, tuple(box for box in boxes if box.identifier != selected_id)


class SceneCollisionKinematics:
    """Add signed robot/obstacle and carried-object/obstacle FCL barriers."""

    def __init__(
        self,
        base,
        selected: ObjectBox,
        obstacles: tuple[ObjectBox, ...],
        carry_frames: np.ndarray,
        contact_frames: np.ndarray,
        target_positions: np.ndarray,
        target_rotations: np.ndarray,
        clearance_m: float = 0.005,
    ) -> None:
        import hppfcl

        if not 0 <= clearance_m <= 0.05 or not np.isfinite(clearance_m):
            raise ValueError("object clearance must be finite and in [0, 0.05] m")
        if base.collision_model is None:
            raise ValueError("object collision requires robot collision geometry")
        frame_count = len(selected.positions)
        if (carry_frames.shape != (frame_count,)
                or contact_frames.shape != (frame_count,)
                or target_positions.shape != (frame_count, 3)
                or target_rotations.shape != (frame_count, 3, 3)):
            raise ValueError("object constraints must align with retarget frames")
        self._base = base
        self._frame = 0
        self._obstacles = (selected,) + obstacles
        self._other_obstacles = obstacles
        self._contact_frames = np.asarray(contact_frames, dtype=bool)
        self._carry_frames = np.asarray(carry_frames, dtype=bool)
        self._clearance = float(clearance_m)
        self._request = hppfcl.DistanceRequest()
        self._request.enable_signed_distance = True
        self._shapes = tuple(hppfcl.Box(*box.size) for box in self._obstacles)
        self._held_shape = hppfcl.Box(*selected.size)
        self._h = hppfcl
        if not any(int(geometry.parentJoint) != 0
                   for geometry in base.collision_model.geometryObjects):
            raise ValueError("robot has no movable collision geometry")
        if self._carry_frames.any():
            grasp_frame = int(np.flatnonzero(self._carry_frames)[0])
            rotation = target_rotations[grasp_frame]
            self._held_rotation = rotation.T @ selected.rotations[grasp_frame]
            object_center = (selected.positions[grasp_frame]
                             + selected.rotations[grasp_frame] @ selected.center_local)
            self._held_translation = rotation.T @ (object_center - target_positions[grasp_frame])

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def set_frame(self, frame: int) -> None:
        if not 0 <= int(frame) < len(self._carry_frames):
            raise IndexError("object collision frame outside aligned track")
        self._frame = int(frame)
        self._base.set_frame(frame)

    def _object_constraints(self, q: np.ndarray, *, linearize: bool):
        if not self._obstacles:
            return np.empty(0), np.empty((0, self.nq))
        pin = self._base.pin
        configuration = self._base._configuration(q)
        if linearize:
            pin.computeJointJacobians(self.model, self._base.data, configuration)
        else:
            pin.forwardKinematics(self.model, self._base.data, configuration)
        pin.updateFramePlacements(self.model, self._base.data)
        pin.updateGeometryPlacements(
            self.model, self._base.data, self._base.collision_model,
            self._base.collision_data, configuration,
        )
        margins = []
        rows = []
        jacobians = {}
        for geometry_index, geometry in enumerate(self._base.collision_model.geometryObjects):
            parent = int(geometry.parentJoint)
            if parent == 0:
                continue
            placement = self._base.collision_data.oMg[geometry_index]
            body_transform = self._h.Transform3f(placement.rotation, placement.translation)
            for obstacle_index, (obstacle, shape) in enumerate(zip(self._obstacles, self._shapes)):
                if (obstacle_index == 0 and self._contact_frames[self._frame]
                        and str(geometry.name).startswith(self._base._gripper_prefix)):
                    continue
                object_rotation = obstacle.rotations[self._frame]
                object_center = (obstacle.positions[self._frame]
                                 + object_rotation @ obstacle.center_local)
                obstacle_transform = self._h.Transform3f(object_rotation, object_center)
                result = self._h.DistanceResult()
                distance = self._h.distance(
                    geometry.geometry, body_transform, shape, obstacle_transform,
                    self._request, result,
                )
                margin = float(distance) - self._clearance
                margins.append(margin)
                if linearize:
                    if parent not in jacobians:
                        jacobians[parent] = np.asarray(pin.getJointJacobian(
                            self.model, self._base.data, parent,
                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                        ))[:, self._base._active_v_indices]
                    contact = np.asarray(result.getNearestPoint1(), dtype=float)
                    offset = contact - np.asarray(self._base.data.oMi[parent].translation)
                    point_jacobian = jacobians[parent][:3] - _skew(offset) @ jacobians[parent][3:]
                    rows.append(-np.asarray(result.normal, dtype=float) @ point_jacobian)
        if self._carry_frames[self._frame]:
            if linearize:
                tool_position, tool_rotation, position_jacobian, rotation_jacobian = (
                    self._base.pose_jacobian(q)
                )
            else:
                tool_position, tool_rotation = self._base.pose(q)
            held_rotation = tool_rotation @ self._held_rotation
            held_center = tool_position + tool_rotation @ self._held_translation
            held_transform = self._h.Transform3f(held_rotation, held_center)
            for obstacle, shape in zip(self._other_obstacles, self._shapes[1:]):
                object_rotation = obstacle.rotations[self._frame]
                object_center = (obstacle.positions[self._frame]
                                 + object_rotation @ obstacle.center_local)
                obstacle_transform = self._h.Transform3f(object_rotation, object_center)
                result = self._h.DistanceResult()
                distance = self._h.distance(
                    self._held_shape, held_transform, shape, obstacle_transform,
                    self._request, result,
                )
                margins.append(float(distance) - self._clearance)
                if linearize:
                    contact = np.asarray(result.getNearestPoint1(), dtype=float)
                    point_jacobian = (position_jacobian
                                      - _skew(contact - tool_position) @ rotation_jacobian)
                    rows.append(-np.asarray(result.normal, dtype=float) @ point_jacobian)
        return np.asarray(margins), np.asarray(rows).reshape(-1, self.nq)

    def collision_margins(self, q: np.ndarray) -> np.ndarray:
        return np.concatenate((self._base.collision_margins(q),
                               self._object_constraints(q, linearize=False)[0]))

    def collision_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_margins, base_jacobian = self._base.collision_linearization(q)
        object_margins, object_jacobian = self._object_constraints(q, linearize=True)
        return (np.concatenate((base_margins, object_margins)),
                np.vstack((base_jacobian, object_jacobian)))

    def object_margins(self, q: np.ndarray) -> np.ndarray:
        return self._object_constraints(q, linearize=False)[0]


def _skew(vector: np.ndarray) -> np.ndarray:
    first, second, third = vector
    return np.array([[0, -third, second], [third, 0, -first], [-second, first, 0]])
