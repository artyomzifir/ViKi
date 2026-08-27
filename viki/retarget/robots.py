"""
viki.retarget.robots
--------------------
Robot descriptions for retargeting: which ``robot_descriptions`` model, which
end-effector frame, and the actuated joint order.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RobotConfig", "ROBOT_CONFIGS", "ROBOT_ALIASES", "normalize_robot"]


@dataclass(frozen=True)
class RobotConfig:
    description: str  # robot_descriptions model name
    ee_frame: str  # Pinocchio frame tracked by the IK
    joint_names: tuple[str, ...]


_UR = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_IIWA = tuple(f"iiwa_joint_{i}" for i in range(1, 8))

ROBOT_CONFIGS: dict[str, RobotConfig] = {
    "ur3": RobotConfig("ur3_description", "wrist_3_link", _UR),
    "ur5": RobotConfig("ur5_description", "wrist_3_link", _UR),
    "ur10": RobotConfig("ur10_official_description", "wrist_3_link", _UR),
    "iiwa14": RobotConfig("iiwa14_description", "iiwa_link_ee", _IIWA),
}

# Legacy / verbose spellings accepted on the CLI and in saved archives.
ROBOT_ALIASES: dict[str, str] = {
    "ur10_description": "ur10",
    "ur10_official_description": "ur10",
    "ur3_description": "ur3",
    "ur5_description": "ur5",
    "iiwa14_description": "iiwa14",
}


def normalize_robot(robot: str) -> RobotConfig:
    key = robot.strip()
    key = ROBOT_ALIASES.get(key, key)
    if key not in ROBOT_CONFIGS:
        raise ValueError(
            f"unknown robot {robot!r}; known: {', '.join(sorted(ROBOT_CONFIGS))}"
        )
    return ROBOT_CONFIGS[key]
