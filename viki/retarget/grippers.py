"""Physical gripper descriptions attached to robot flange frames.

This is separate from :mod:`viki.gripper`: that module estimates a semantic
normalised opening from a human hand, while this module turns the command into
a moving URDF joint, a TCP frame, collision geometry, and viewer geometry.
Only descriptions with a verified source and licence are loadable.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import viki.config as app_config
from viki.retarget.adapters import CylinderGripperAdapter, GripperAdapter
from viki.retarget.robots import RobotConfig

__all__ = [
    "DEFAULT_GRIPPER",
    "GRIPPER_CONFIGS",
    "AttachedRobot",
    "ToolGripperConfig",
    "attach_gripper",
    "gripper_catalog",
    "normalize_gripper",
]


@dataclass(frozen=True)
class ToolGripperConfig:
    """One physical two-jaw gripper and its canonical URDF contract."""

    key: str
    label: str
    description: str | None
    tcp_frame: str | None
    drive_joint: str | None
    open_position: float
    closed_position: float
    max_width_m: float
    max_speed_m_s: float
    grasp_contact_links: tuple[str, str] | None = None
    grasp_contact_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_center_name: str = "grasp_center"
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    license: str = "unverified"
    source_url: str = ""
    unavailable_reason: str = ""

    @property
    def loadable(self) -> bool:
        return bool(self.description and self.tcp_frame and self.drive_joint)

    def joint_positions(self, opening: np.ndarray) -> np.ndarray:
        """Map normalised opening (0 closed, 1 open) to the master joint."""
        state = np.clip(np.asarray(opening, dtype=np.float64), 0.0, 1.0)
        return self.closed_position + state * (
            self.open_position - self.closed_position
        )

    def opening_widths(self, opening: np.ndarray) -> np.ndarray:
        state = np.clip(np.asarray(opening, dtype=np.float64), 0.0, 1.0)
        return state * self.max_width_m

    def profile_opening(self, opening: np.ndarray, dt: float) -> np.ndarray:
        """Rate-limit a target opening using the tool's physical jaw speed."""
        target = np.clip(np.asarray(opening, dtype=np.float64), 0.0, 1.0)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError("gripper opening must be a finite 1-D trajectory")
        if not len(target):
            return target.copy()
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("gripper profile dt must be positive")
        if self.max_width_m <= 0.0 or self.max_speed_m_s <= 0.0:
            raise ValueError("gripper width and speed must be positive")
        max_step = self.max_speed_m_s * dt / self.max_width_m
        profiled = target.copy()
        for frame in range(1, len(profiled)):
            delta = np.clip(
                target[frame] - profiled[frame - 1], -max_step, max_step
            )
            profiled[frame] = profiled[frame - 1] + delta
        return np.clip(profiled, 0.0, 1.0)


DEFAULT_GRIPPER = "robotiq_2f85"


GRIPPER_CONFIGS: dict[str, ToolGripperConfig] = {
    "robotiq_2f85": ToolGripperConfig(
        key="robotiq_2f85",
        label="Robotiq 2F-85",
        description="robotiq_2f85_v4_description",
        tcp_frame="robotiq_arg2f_tcp",
        drive_joint="finger_joint",
        open_position=0.0,
        closed_position=0.8,
        max_width_m=0.085,
        max_speed_m_s=0.150,
        # The vendor URDF's robotiq_arg2f_tcp is fixed at z=174 mm, beyond
        # the fingers. The commented pad joints define the centre of each
        # rubber contact panel in the distal-phalanx frame. Retargeting
        # averages these two moving points so it follows the four-bar linkage.
        grasp_contact_links=("left_distal_phalanx", "right_distal_phalanx"),
        grasp_contact_offset_m=(0.0, -0.0220203446692936, 0.03242),
        # The canonical URDF begins at its physical base. The former hidden
        # 11 mm plate offset now lives in the explicit user adapter, together
        # with matching visual/collision geometry.
        mount_translation=(0.0, 0.0, 0.0),
        license="BSD-2-Clause",
        source_url="https://github.com/nickswalker/robotiq-2f-85",
    ),
    "robotiq_2f140": ToolGripperConfig(
        key="robotiq_2f140",
        label="Robotiq 2F-140",
        description=None,
        tcp_frame=None,
        drive_joint=None,
        open_position=0.0,
        closed_position=0.695,
        max_width_m=0.140,
        max_speed_m_s=0.150,
        license="BSD-3-Clause",
        source_url="https://github.com/robotiq/ros",
        unavailable_reason="official Xacro identified; audited packaging is pending",
    ),
    "schunk_wsg50": ToolGripperConfig(
        key="schunk_wsg50",
        label="SCHUNK WSG-50",
        description=None,
        tcp_frame=None,
        drive_joint=None,
        open_position=0.055,
        closed_position=0.0,
        max_width_m=0.110,
        max_speed_m_s=0.420,
        source_url="https://github.com/jksrecko/wsg50-ros-pkg",
        unavailable_reason="available URDFs are third-party; source/licence review is pending",
    ),
    "smc_mhz2_16d": ToolGripperConfig(
        key="smc_mhz2_16d",
        label="SMC MHZ2-16D",
        description=None,
        tcp_frame=None,
        drive_joint=None,
        open_position=0.008,
        closed_position=0.0,
        max_width_m=0.016,
        max_speed_m_s=0.050,
        source_url="https://www.smcworld.com/cad2d/en/item.do?mode=ser&word=mhz2",
        unavailable_reason="manufacturer CAD found; a redistributable URDF is not verified",
    ),
    "smc_mhz2_20d": ToolGripperConfig(
        key="smc_mhz2_20d",
        label="SMC MHZ2-20D",
        description=None,
        tcp_frame=None,
        drive_joint=None,
        open_position=0.010,
        closed_position=0.0,
        max_width_m=0.020,
        max_speed_m_s=0.050,
        source_url="https://www.smcworld.com/cad2d/en/item.do?mode=ser&word=mhz2",
        unavailable_reason="manufacturer CAD found; a redistributable URDF is not verified",
    ),
}

_ALIASES = {
    "binary": DEFAULT_GRIPPER,
    "2f85": "robotiq_2f85",
    "2f-85": "robotiq_2f85",
    "robotiq": "robotiq_2f85",
    "2f140": "robotiq_2f140",
    "2f-140": "robotiq_2f140",
    "wsg50": "schunk_wsg50",
    "wsg-50": "schunk_wsg50",
    "mhz2-16d": "smc_mhz2_16d",
    "mhz2-20d": "smc_mhz2_20d",
}


@dataclass(frozen=True)
class AttachedRobot:
    """Pinocchio robot wrapper plus the names introduced by the tool."""

    robot: object
    tcp_frame: str
    orientation_frame: str
    position_points: tuple[tuple[str, tuple[float, float, float]], ...]
    drive_joint: str
    gripper_prefix: str


def normalize_gripper(name: str | None) -> ToolGripperConfig:
    key = str(name or DEFAULT_GRIPPER).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        cfg = GRIPPER_CONFIGS[key]
    except KeyError:
        raise ValueError(
            f"unknown physical gripper {name!r}; known: "
            f"{', '.join(sorted(GRIPPER_CONFIGS))}"
        ) from None
    if not cfg.loadable:
        detail = cfg.unavailable_reason or "URDF integration is not implemented"
        raise ValueError(f"physical gripper {cfg.label} is not available: {detail}")
    return cfg


def gripper_catalog() -> list[dict[str, Any]]:
    """JSON-ready registry for the Retarget model picker."""
    return [
        {
            "key": cfg.key,
            "label": cfg.label,
            "available": cfg.loadable,
            "description": cfg.description,
            "tcp_frame": cfg.tcp_frame,
            "drive_joint": cfg.drive_joint,
            "max_width_m": cfg.max_width_m,
            "max_speed_m_s": cfg.max_speed_m_s,
            "tracking_point": (
                cfg.grasp_center_name if cfg.grasp_contact_links else cfg.tcp_frame
            ),
            "license": cfg.license,
            "source_url": cfg.source_url,
            "note": cfg.unavailable_reason,
        }
        for cfg in GRIPPER_CONFIGS.values()
    ]


def _prefix_description(robot: object, prefix: str) -> None:
    """Avoid frame/joint/geometry name collisions before ``appendModel``."""
    model = robot.model
    for index, name in enumerate(list(model.names)):
        model.names[index] = prefix + str(name)
    for frame in model.frames:
        frame.name = prefix + str(frame.name)
    for geometry_model in (robot.collision_model, robot.visual_model):
        for geometry in geometry_model.geometryObjects:
            geometry.name = prefix + str(geometry.name)


def _load_description_with_mimic(description: str) -> object:
    """Load one robot_descriptions URDF with mimic joints as one DoF."""
    os.environ.setdefault(
        "ROBOT_DESCRIPTIONS_CACHE",
        os.path.join(
            str(getattr(app_config, "MODELS_DIR", "models/")), "robot_descriptions"
        ),
    )
    try:
        import pinocchio as pin
        from robot_descriptions._package_dirs import get_package_dirs
        from robot_descriptions._xacro import get_urdf_path
    except ImportError as exc:
        raise RuntimeError(
            "physical grippers require Pinocchio and robot_descriptions"
        ) from exc

    module = importlib.import_module(f"robot_descriptions.{description}")
    urdf_path = get_urdf_path(module)
    package_dirs = get_package_dirs(module)
    model = pin.buildModelFromUrdf(urdf_path, True)
    collision_model = pin.buildGeomFromUrdf(
        model,
        urdf_path,
        pin.GeometryType.COLLISION,
        package_dirs=package_dirs,
    )
    visual_model = pin.buildGeomFromUrdf(
        model,
        urdf_path,
        pin.GeometryType.VISUAL,
        package_dirs=package_dirs,
    )
    return SimpleNamespace(
        model=model,
        collision_model=collision_model,
        visual_model=visual_model,
    )


def attach_gripper(
    arm: object,
    robot_cfg: RobotConfig,
    gripper_cfg: ToolGripperConfig,
    adapter: GripperAdapter | None = None,
) -> AttachedRobot:
    """Append ``flange -> adapter -> gripper`` and return the assembly."""
    if not gripper_cfg.loadable:
        detail = gripper_cfg.unavailable_reason or "URDF integration is not implemented"
        raise ValueError(f"physical gripper {gripper_cfg.label} is not available: {detail}")
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("physical grippers require Pinocchio") from exc

    mount_frame = arm.model.getFrameId(robot_cfg.mount_frame)
    if mount_frame >= len(arm.model.frames):
        raise ValueError(
            f"robot {robot_cfg.description} has no tool mount frame {robot_cfg.mount_frame!r}"
        )

    adapter = adapter or CylinderGripperAdapter()
    adapter.add_geometry(arm, mount_frame, pin)
    tool = _load_description_with_mimic(str(gripper_cfg.description))
    prefix = f"{gripper_cfg.key}__"
    _prefix_description(tool, prefix)
    adapter_placement = pin.SE3(
        adapter.rotation_matrix(),
        np.asarray(adapter.translation_m, dtype=np.float64),
    )
    urdf_placement = pin.SE3(
        Rotation.from_euler(
            "xyz", gripper_cfg.mount_rpy_deg, degrees=True
        ).as_matrix(),
        np.asarray(gripper_cfg.mount_translation, dtype=np.float64),
    )
    placement = adapter_placement * urdf_placement
    model, collision_model = pin.appendModel(
        arm.model,
        tool.model,
        arm.collision_model,
        tool.collision_model,
        mount_frame,
        placement,
    )
    _visual_model_copy, visual_model = pin.appendModel(
        arm.model,
        tool.model,
        arm.visual_model,
        tool.visual_model,
        mount_frame,
        placement,
    )
    orientation_frame = prefix + str(gripper_cfg.tcp_frame)
    if gripper_cfg.grasp_contact_links:
        position_points = tuple(
            (prefix + link, gripper_cfg.grasp_contact_offset_m)
            for link in gripper_cfg.grasp_contact_links
        )
        tcp_frame = prefix + gripper_cfg.grasp_center_name
    else:
        position_points = ()
        tcp_frame = orientation_frame
    drive_joint = prefix + str(gripper_cfg.drive_joint)
    if model.getFrameId(orientation_frame) >= len(model.frames):
        raise ValueError(
            f"{gripper_cfg.label} URDF has no orientation frame {orientation_frame!r}"
        )
    for frame_name, _offset in position_points:
        if model.getFrameId(frame_name) >= len(model.frames):
            raise ValueError(
                f"{gripper_cfg.label} URDF has no contact frame {frame_name!r}"
            )
    if model.getJointId(drive_joint) >= model.njoints:
        raise ValueError(f"{gripper_cfg.label} URDF has no drive joint {drive_joint!r}")
    return AttachedRobot(
        robot=SimpleNamespace(
            model=model,
            collision_model=collision_model,
            visual_model=visual_model,
        ),
        tcp_frame=tcp_frame,
        orientation_frame=orientation_frame,
        position_points=position_points,
        drive_joint=drive_joint,
        gripper_prefix=prefix,
    )
