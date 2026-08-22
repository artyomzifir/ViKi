"""
viki.optimization.preparation.fusion
-------------------------------------
Fuse per‑camera 3‑D hand trajectories into a single world‑frame skeleton
trajectory.

The live skeleton pipeline deliberately does NOT fuse at capture time — it emits
one ``SkeletonFrame`` per camera (tagged with ``device_id``). The trajectories
are recorded separately and fused here, at the smooth/optimisation stage, right
after interpolation. This keeps capture cheap and lets the frontend visualise
every camera's hand in its own colour before any averaging happens.

The fusion is a per‑landmark, per‑timestep weighted average across the cameras
that observed that landmark at that time. Cameras are synchronised, so their
trajectories share timestamps; we resample each camera onto the common time
grid (union of all timestamps) before averaging, which also gracefully handles
frames where a camera missed the hand.
"""

from __future__ import annotations

import numpy as np


def _resample_to_grid(points: np.ndarray, ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Linearly resample a (Tc, L, 3) trajectory onto a common time ``grid``.

    Parameters
    ----------
    points : np.ndarray
        Camera trajectory, shape (Tc, L, 3).
    ts : np.ndarray
        Monotonically non‑decreasing sample timestamps, shape (Tc,).
    grid : np.ndarray
        Target timestamps, shape (G,), sorted ascending.

    Returns
    -------
    np.ndarray
        Resampled trajectory, shape (G, L, 3). Filled with NaN if ``points``
        is empty; broadcast (G, L, 3) if it contains a single frame.
    """
    Tc, L, _ = points.shape
    G = grid.shape[0]
    if Tc == 0:
        return np.full((G, L, 3), np.nan, dtype=np.float32)
    if Tc == 1:
        return np.broadcast_to(points[0].astype(np.float32), (G, L, 3)).copy()
    out = np.empty((G, L, 3), dtype=np.float32)
    for li in range(L):
        for c in range(3):
            out[:, li, c] = np.interp(grid, ts, points[:, li, c])
    return out


def fuse_trajectories(
    trajectories: dict[str, np.ndarray],
    timestamps: dict[str, np.ndarray],
    landmark_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse per‑camera world‑frame trajectories into one trajectory.

    Parameters
    ----------
    trajectories : dict[str, np.ndarray]
        Mapping ``device_id -> (Tc, L, 3)`` world‑frame landmark positions.
    timestamps : dict[str, np.ndarray]
        Mapping ``device_id -> (Tc,)`` sample timestamps (µs) aligned with
        ``trajectories``.
    landmark_ids : np.ndarray
        Landmark id array (length L) describing the second axis of the points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(fused_points, common_grid)`` where ``fused_points`` has shape
        ``(G, L, 3)`` (NaN where no camera observed a landmark at a step) and
        ``common_grid`` is the sorted union of input timestamps, shape ``(G,)``.
    """
    device_ids = list(trajectories.keys())
    if not device_ids:
        return (
            np.empty((0, len(landmark_ids), 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    all_ts = np.unique(
        np.concatenate([np.asarray(t, dtype=np.int64) for t in timestamps.values()])
    )
    all_ts = np.sort(all_ts)
    G = all_ts.shape[0]
    L = len(landmark_ids)

    resampled: dict[str, np.ndarray] = {}
    for dev_id in device_ids:
        pts = np.asarray(trajectories[dev_id], dtype=np.float32)
        ts = np.asarray(timestamps[dev_id], dtype=np.int64)
        order = np.argsort(ts)
        resampled[dev_id] = _resample_to_grid(pts[order], ts[order], all_ts)
        # A camera only "observes" the grid steps that match its own sample
        # timestamps. Mask everything else as NaN so the average below ignores
        # frames it never actually captured (resampling would otherwise fabricate
        # a vote at timestamps where only another camera saw the hand).
        observed = np.isin(all_ts, ts[order])
        resampled[dev_id][~observed] = np.nan

    fused = np.full((G, L, 3), np.nan, dtype=np.float32)
    # Stack cameras: (C, G, L, 3) -> per step average over cameras.
    stacked = np.stack([resampled[d] for d in device_ids], axis=0)  # (C, G, L, 3)
    valid = np.isfinite(stacked)
    with np.errstate(invalid="ignore"):
        summed = np.nansum(stacked, axis=0)  # (G, L, 3)
        counted = np.sum(valid, axis=0)  # (G, L, 3)
    present = counted > 0
    fused[present] = summed[present] / counted[present]

    return fused, all_ts.astype(np.int64)
