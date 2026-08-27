"""Compatibility shim — retarget entry points moved to :mod:`viki.retarget.run`."""

from viki.retarget.run import *  # noqa: F401,F403
from viki.retarget.run import (  # noqa: F401
    RetargetInput,
    RunConfig,
    retarget,
    retarget_from_poses,
)
from viki.retarget.robots import ROBOT_CONFIGS, RobotConfig, normalize_robot  # noqa: F401
