"""A time-bounded spatial corridor around a recorded object trajectory."""

from __future__ import annotations

import numpy as np


class ObjectTrajectoryReference:
    def __init__(self, timestamps_us: np.ndarray, positions_m: np.ndarray,
                 reset_frame: int, time_scale: float = 2.0,
                 lag_s: float = 0.6, radius_m: float = 0.02):
        timestamps = np.asarray(timestamps_us, dtype=float)
        positions = np.asarray(positions_m, dtype=float)
        if (timestamps.ndim != 1 or positions.shape != (len(timestamps), 3)
                or not 0 <= reset_frame < len(timestamps) - 1
                or not np.isfinite(timestamps).all()
                or not np.isfinite(positions).all()
                or np.any(np.diff(timestamps) <= 0)
                or not np.isfinite([time_scale, lag_s, radius_m]).all()
                or time_scale <= 0 or lag_s < 0 or radius_m <= 0):
            raise ValueError("invalid object trajectory reference")
        self.times_s = (timestamps[reset_frame:] - timestamps[reset_frame]) * time_scale / 1e6
        self.positions_m = positions[reset_frame:]
        self.duration_s = float(self.times_s[-1])
        self.time_scale = time_scale
        self.lag_s = lag_s
        self.radius_m = radius_m

    def target(self, elapsed_s: float) -> np.ndarray:
        time_s = np.clip(elapsed_s, 0.0, self.duration_s)
        return np.array([
            np.interp(time_s, self.times_s, self.positions_m[:, axis])
            for axis in range(3)
        ])

    def distance_m(self, position_m: np.ndarray, elapsed_s: float) -> float:
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(elapsed_s):
            raise ValueError("invalid trajectory query")
        end_s = min(max(elapsed_s, 0.0), self.duration_s)
        start_s = max(0.0, end_s - self.lag_s)
        inside = (self.times_s > start_s) & (self.times_s < end_s)
        path = np.vstack((
            self.target(start_s), self.positions_m[inside], self.target(end_s),
        ))
        segments = np.diff(path, axis=0)
        lengths_squared = np.einsum("ij,ij->i", segments, segments)
        fractions = np.clip(
            np.divide(
                np.einsum("ij,ij->i", position - path[:-1], segments),
                lengths_squared, out=np.zeros_like(lengths_squared),
                where=lengths_squared > 0,
            ), 0.0, 1.0,
        )
        nearest = path[:-1] + fractions[:, None] * segments
        return float(np.min(np.linalg.norm(nearest - position, axis=1)))

    def outside_m(self, position_m: np.ndarray, elapsed_s: float) -> float:
        return max(0.0, self.distance_m(position_m, elapsed_s) - self.radius_m)
