"""
viki.replay
-----------
Optional offline dry-run: plan.h5 -> replay.h5.

Copies planned values and checks available joint limits. Archive names remain
compatible with existing files; no controller execution is implemented here.
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
