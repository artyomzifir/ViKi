"""
viki.perception.recorder
------------------------
The single ``rec.npz`` writer: per-camera hand landmark trajectories, world
frame, plus a per-landmark fusion-weight column. Capture-time fusion is not done
here — the per-camera trajectories are kept for :mod:`viki.prepare` to fuse.

Schema (``viki.contracts.REC_KEYS``)::

    device_ids   (N,)        camera id per recorded frame
    timestamps   (N,)  int64 µs
    points       (N,21,3)    world-frame landmark XYZ (NaN where missing)
    landmark_ids (21,) int32
    confidence   (N,21) f32  per-landmark fusion weight (paper §3.5, eq. 2)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from viki.contracts import HAND_LM_COUNT, LM, REC_KEYS, SkeletonFrame

_NAN3 = np.full(3, np.nan, dtype=np.float32)
_IDS = list(range(HAND_LM_COUNT))


def _stack(records: list[tuple[str, SkeletonFrame, dict | None]]):
    dev, ts, pts, conf = [], [], [], []
    for dev_id, frame, weights in records:
        dev.append(dev_id)
        ts.append(int(frame.timestamp_us))
        pts.append(
            np.array([frame.points.get(LM(i), _NAN3) for i in _IDS], dtype=np.float32)
        )
        w = weights or {}
        conf.append(
            np.array([float(w.get(LM(i), 0.0)) for i in _IDS], dtype=np.float32)
        )
    return dev, ts, pts, conf


def write_rec(path: str | Path, records: list[tuple[str, SkeletonFrame, dict | None]]) -> str:
    """Write ``records`` (device_id, frame, per-landmark weights) to ``path``."""
    dev, ts, pts, conf = _stack(records)
    npz = {
        "device_ids": np.array(dev) if dev else np.empty((0,), dtype="<U1"),
        "timestamps": np.array(ts, dtype=np.int64),
        "points": np.stack(pts) if pts else np.empty((0, HAND_LM_COUNT, 3), np.float32),
        "landmark_ids": np.array(_IDS, dtype=np.int32),
        "confidence": np.stack(conf) if conf else np.empty((0, HAND_LM_COUNT), np.float32),
    }
    assert set(npz) == set(REC_KEYS)
    np.savez_compressed(path, **npz)
    return str(path)


class SkeletonRecorder:
    """Buffer per-camera ``SkeletonFrame``s over a session, then ``write_rec``."""

    def __init__(self, base_dir: str | Path = "data/skeleton_recs", **_ignored) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._current: Path | None = None
        self._records: list[tuple[str, SkeletonFrame, dict | None]] = []

    def start(self) -> str:
        self._records = []
        name = f"rec-{datetime.now():%H.%M-%d.%m.%Y}.npz"
        self._current = self._base_dir / name
        return name

    def record(self, frame: SkeletonFrame, weights: dict | None = None, **_ignored) -> None:
        if self._current is not None:
            self._records.append((frame.device_id, frame, weights))

    def stop(self) -> str | None:
        if self._current is None:
            return None
        path = write_rec(self._current, self._records)
        self._current, self._records = None, []
        return path

    @property
    def is_recording(self) -> bool:
        return self._current is not None
