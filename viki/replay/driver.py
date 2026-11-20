"""
viki.replay.driver
------------------
Offline dry-run of a synthesised joint trajectory.

The only driver echoes planned values with a NaN residual. Legacy archive
field names are retained for compatibility with existing episode files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ProprioLog:
    """Dry-run values; ``*_attained`` are legacy names for copies of the plan."""

    q_attained: np.ndarray  # (T, nq)
    gripper_attained: np.ndarray  # (T,) float, normalised opening
    controller_residual: np.ndarray  # (T,) float, NaN if not measured


class RobotDriver(ABC):
    name: str

    @abstractmethod
    def execute(
        self, q_traj: np.ndarray, gripper: np.ndarray, dt: float
    ) -> ProprioLog: ...

    def close(self) -> None:
        """Release local resources. Default: nothing to do."""


class DryRunDriver(RobotDriver):
    """Echoes the plan; residual is NaN."""

    name = "dryrun"

    def execute(self, q_traj: np.ndarray, gripper: np.ndarray, dt: float) -> ProprioLog:
        q = np.asarray(q_traj, dtype=np.float64)
        g = np.asarray(gripper, dtype=np.float64)
        if g.shape != (len(q),) or not np.isfinite(g).all():
            raise ValueError("gripper opening must be a finite (T,) trajectory")
        if np.any(g < 0.0) or np.any(g > 1.0):
            raise ValueError("gripper opening must stay in [0, 1]")
        return ProprioLog(
            q_attained=q.copy(),
            gripper_attained=g.copy(),
            controller_residual=np.full(len(q), np.nan),
        )


_DRIVERS: dict[str, type[RobotDriver]] = {"dryrun": DryRunDriver}


def load_driver(name: str = "dryrun", **kwargs) -> RobotDriver:
    try:
        cls = _DRIVERS[name]
    except KeyError:
        raise ValueError(
            f"unknown driver {name!r}; known: {', '.join(sorted(_DRIVERS))}"
        ) from None
    return cls(**kwargs)
