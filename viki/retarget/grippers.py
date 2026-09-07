"""Physical gripper descriptions attached to robot flange frames.

This is separate from :mod:`viki.gripper`: that module estimates a semantic
open/closed command from a human hand, while this module turns the command into
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
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    license: str = "unverified"
    source_url: str = ""
    unavailable_reason: str = ""

    @property
    def loadable(self) -> bool:
        return bool(self.description and self.tcp_frame and self.drive_joint)

    def joint_positions(self, closed: np.ndarray) -> np.ndarray:
        """Map a binary semantic trajectory to the URDF's master joint."""
        state = np.asarray(closed, dtype=bool)
        return np.where(state, self.closed_position, self.open_position).astype(
            np.float64
        )

    def opening_widths(self, closed: np.ndarray) -> np.ndarray:
        state = np.asarray(closed, dtype=bool)
        return np.where(state, 0.0, self.max_width_m).astype(np.float64)


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
        # The standard UR-to-Robotiq plate places the gripper base 11 mm past
        # tool0. Its upstream default rotation is zero.
        mount_translation=(0.0, 0.0, 0.011),
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
) -> AttachedRobot:
    """Append a gripper URDF to the robot tool frame and return the assembly."""
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

    tool = _load_description_with_mimic(str(gripper_cfg.description))
    prefix = f"{gripper_cfg.key}__"
    _prefix_description(tool, prefix)
    placement = pin.SE3(
        Rotation.from_euler(
            "xyz", gripper_cfg.mount_rpy_deg, degrees=True
        ).as_matrix(),
        np.asarray(gripper_cfg.mount_translation, dtype=np.float64),
    )
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
    tcp_frame = prefix + str(gripper_cfg.tcp_frame)
    drive_joint = prefix + str(gripper_cfg.drive_joint)
    if model.getFrameId(tcp_frame) >= len(model.frames):
        raise ValueError(f"{gripper_cfg.label} URDF has no TCP frame {tcp_frame!r}")
    if model.getJointId(drive_joint) >= model.njoints:
        raise ValueError(f"{gripper_cfg.label} URDF has no drive joint {drive_joint!r}")
    return AttachedRobot(
        robot=SimpleNamespace(
            model=model,
            collision_model=collision_model,
            visual_model=visual_model,
        ),
        tcp_frame=tcp_frame,
        drive_joint=drive_joint,
        gripper_prefix=prefix,
    )
