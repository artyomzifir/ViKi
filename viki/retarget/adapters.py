"""User-defined transforms between a robot flange and a physical gripper.

The PoC deliberately models only the interface contract, not an exact vendor
adapter plate.  A concrete adapter supplies the fixed flange-to-gripper pose
and may add matching visual/collision geometry to the composite Pinocchio
model.  A future CAD-backed adapter can implement the same interface without
changing retarget orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "CylinderGripperAdapter",
    "GripperAdapter",
]


class GripperAdapter(ABC):
    """Physical flange-to-gripper interface used by robot composition."""

    translation_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]

    @property
    @abstractmethod
    def kind(self) -> str:
        """Stable identifier written to the plan artifact."""

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.asarray(self.translation_m, dtype=np.float64)))

    def rotation_matrix(self) -> np.ndarray:
        return Rotation.from_euler("xyz", self.rpy_deg, degrees=True).as_matrix()

    @abstractmethod
    def add_geometry(self, arm: object, mount_frame_id: int, pin: object) -> None:
        """Add this adapter's visual and collision geometry to ``arm``."""


def _rotation_z_to(direction: np.ndarray) -> np.ndarray:
    """Return a deterministic rotation that maps local +Z to ``direction``."""
    # Normalisation must not mutate ``direction``: callers also use the same
    # translation to place the cylinder centre. Mutating it turned millimetre
    # offsets into a unit vector and placed the adapter centre 0.5 m away.
    target = np.asarray(direction, dtype=np.float64).copy()
    target /= np.linalg.norm(target)
    source = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        return Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def _geometry_object(
    pin: object,
    name: str,
    parent_joint: int,
    parent_frame: int,
    placement: object,
    shape: object,
) -> object:
    """Construct a GeometryObject across supported Pinocchio Python ABIs."""
    try:
        return pin.GeometryObject(
            name, parent_joint, parent_frame, placement, shape
        )
    except TypeError:
        # Pinocchio 2.x does not expose the parent-frame overload.
        return pin.GeometryObject(name, parent_joint, placement, shape)


@dataclass(frozen=True)
class CylinderGripperAdapter(GripperAdapter):
    """PoC adapter: a user pose plus a cylinder connecting both origins.

    ``translation_m`` is authoritative for both kinematics and cylinder
    length.  The cylinder is centred on and aligned with that vector.  The RPY
    rotates the gripper at the far end; it does not twist the rotationally
    symmetric cylinder.
    """

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.035

    @property
    def kind(self) -> str:
        return "user_cylinder"

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=np.float64)
        rpy = np.asarray(self.rpy_deg, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("adapter translation must contain three finite values")
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            raise ValueError("adapter RPY must contain three finite values")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("adapter radius must be positive")

    def add_geometry(self, arm: object, mount_frame_id: int, pin: object) -> None:
        if self.length_m <= 1e-9:
            return
        try:
            import hppfcl
        except ImportError:
            try:
                import coal as hppfcl
            except ImportError as exc:
                raise RuntimeError("cylindrical adapter geometry requires hpp-fcl") from exc

        frame = arm.model.frames[mount_frame_id]
        parent_joint = int(frame.parentJoint)
        offset = np.asarray(self.translation_m, dtype=np.float64)
        local = pin.SE3(_rotation_z_to(offset), 0.5 * offset)
        placement = frame.placement * local

        for suffix, geometry_model, color in (
            ("collision", arm.collision_model, (0.35, 0.55, 0.68, 1.0)),
            ("visual", arm.visual_model, (0.25, 0.70, 0.90, 1.0)),
        ):
            # hpp-fcl's Python constructor accepts the full cylinder length;
            # the exposed ``halfLength`` property is derived internally.
            shape = hppfcl.Cylinder(float(self.radius_m), self.length_m)
            geometry = _geometry_object(
                pin,
                f"adapter__cylinder_{suffix}",
                parent_joint,
                mount_frame_id,
                placement,
                shape,
            )
            geometry.meshColor = np.asarray(color, dtype=np.float64)
            geometry_model.addGeometryObject(geometry)
