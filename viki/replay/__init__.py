"""
viki.replay
-----------
Pipeline stage 5: plan.h5 -> replay.h5 (paper §3.8).

Execute on the manipulator, log proprioception, screen feasibility. Stub stage —
``driver="dryrun"`` runs no hardware; ``driver="ur3"`` needs ur-rtde + the robot.
"""

from viki.replay.driver import DryRunDriver, RobotDriver, load_driver  # noqa: F401
from viki.replay.run import replay_episode  # noqa: F401
from viki.replay.screen import Verdict, screen  # noqa: F401

__all__ = [
    "replay_episode",
    "RobotDriver",
    "DryRunDriver",
    "load_driver",
    "screen",
    "Verdict",
]
