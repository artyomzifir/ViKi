"""
viki.export.lerobot
-------------------
Write a LeRobot dataset from screened, labelled episodes (paper §3.9).

Delegates to the ``lerobot`` package (``LeRobotDataset.create`` / ``add_frame`` /
``save_episode`` / ``finalize``). ``lerobot`` is an optional dependency
(``pip install viki[export]``) because it pulls in torch.

STUB status: the frame dict below is assembled from what ViKi actually has —
replay-attained joint+gripper states, both wrist-trajectory forms, ω_t, the
controller residual, and the phase label. Fields marked ``# placeholder`` are
shape-correct but not yet meaningful; ``info.json`` is stamped
``viki_export_status="partial"``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _require_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset
    except Exception:  # pragma: no cover - optional dep
        try:
            from lerobot.datasets import LeRobotDataset  # type: ignore

            return LeRobotDataset
        except Exception as exc:
            raise RuntimeError(
                "the LeRobot exporter needs the `lerobot` package: "
                "pip install 'viki[export]'"
            ) from exc


def _features(nq: int, cams: list[str], fps: int) -> dict:
    img = {"dtype": "video", "shape": [3, 0, 0], "names": ["channels", "height", "width"]}
    feats = {cam: dict(img) for cam in (f"observation.images.{c}" for c in cams)}
    feats.update(
        {
            "observation.state": {"dtype": "float32", "shape": [nq + 1], "names": None},
            "action": {"dtype": "float32", "shape": [nq + 1], "names": None},
            "annotation.wrist_pose": {"dtype": "float32", "shape": [16], "names": None},
            "annotation.object_relative": {"dtype": "float32", "shape": [16], "names": None},  # placeholder
            "annotation.confidence": {"dtype": "float32", "shape": [1], "names": None},
            "annotation.replay_residual": {"dtype": "float32", "shape": [1], "names": None},
            "annotation.phase": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return feats


class LeRobotWriter:
    """One dataset, many episodes."""

    def __init__(self, out_dir: str | Path, *, fps: int, robot_type: str, cams: list[str], nq: int):
        LeRobotDataset = _require_lerobot()
        self._ds = LeRobotDataset.create(
            repo_id=Path(out_dir).name,
            fps=fps,
            root=str(out_dir),
            robot_type=robot_type or "unknown",
            features=_features(nq, cams, fps),
        )
        self._cams = cams

    def add_episode(self, frames: list[dict], task: str) -> None:
        for frame in frames:
            self._ds.add_frame({**frame, "task": task})
        self._ds.save_episode()

    def finalize(self) -> None:
        # v3 needs finalize(); older builds no-op.
        if hasattr(self._ds, "finalize"):
            self._ds.finalize()
