"""
viki.replay.driver
------------------
Execution of a synthesised joint trajectory on the manipulator.

``RobotDriver`` is the abstraction. ``UR3Driver`` is a STUB — real execution
needs ``ur-rtde`` and the physical UR3 (paper §3.8). ``DryRunDriver`` runs no
hardware: it reports the planned trajectory back as if it were attained, with a
NaN controller residual, so the rest of the pipeline (screening, export) can be
exercised end to end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ProprioLog:
    """What the robot actually did during execution."""

    q_attained: np.ndarray  # (T, nq)
    gripper_attained: np.ndarray  # (T,) bool
    controller_residual: np.ndarray  # (T,) float, NaN if not measured


class RobotDriver(ABC):
    name: str

    @abstractmethod
    def execute(
        self, q_traj: np.ndarray, gripper: np.ndarray, dt: float
    ) -> ProprioLog: ...

    def close(self) -> None:
        """Release the connection. Default: nothing to do."""


class DryRunDriver(RobotDriver):
    """No hardware. Echoes the plan; residual is NaN."""

    name = "dryrun"

    def execute(self, q_traj: np.ndarray, gripper: np.ndarray, dt: float) -> ProprioLog:
        q = np.asarray(q_traj, dtype=np.float64)
        g = np.asarray(gripper, dtype=bool)
        return ProprioLog(
            q_attained=q.copy(),
            gripper_attained=g.copy(),
            controller_residual=np.full(len(q), np.nan),
        )


class UR3Driver(RobotDriver):
    """STUB. Stream ``q_traj`` to a UR3 over ur-rtde and log proprioception."""

    name = "ur3"

    def __init__(self, host: str | None = None) -> None:
        raise NotImplementedError(
            "UR3Driver needs the `ur-rtde` package and a physical UR3 "
            "(paper §3.8). Use driver='dryrun' for a hardware-free run."
        )

    def execute(self, q_traj: np.ndarray, gripper: np.ndarray, dt: float) -> ProprioLog:
        raise NotImplementedError


_DRIVERS: dict[str, type[RobotDriver]] = {"dryrun": DryRunDriver, "ur3": UR3Driver}


def load_driver(name: str = "dryrun", **kwargs) -> RobotDriver:
    try:
        cls = _DRIVERS[name]
    except KeyError:
        raise ValueError(
            f"unknown driver {name!r}; known: {', '.join(sorted(_DRIVERS))}"
        ) from None
    return cls(**kwargs)
