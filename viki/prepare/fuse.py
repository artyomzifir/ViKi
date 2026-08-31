"""
viki.prepare.fuse
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
    weights: dict[str, np.ndarray] | None = None,
    grid: np.ndarray | None = None,
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
    weights : dict[str, np.ndarray] | None
        Optional per-camera ``(Tc, L)`` fusion weights (paper §3.5, eq. 2).
        When ``None`` every observation gets weight 1 (plain mean). STUB: the
        caller currently passes detector visibility only — the range and
        incidence factors are not computed yet.
    grid : np.ndarray | None
        Optional explicit output time grid (µs). When given, every camera is
        resampled onto it and a step counts as "observed" by a camera if that
        camera has a real sample within half a grid period of it. Pass the raw
        synced-frame timestamps here so ``cln.npz`` shares one index with the
        point cloud and every other per-frame artifact. When ``None`` the grid
        is the sorted union of the input timestamps (irregular dt, ~2x frames
        with two software-synced cameras).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(fused_points, common_grid)`` where ``fused_points`` has shape
        ``(G, L, 3)`` (NaN where no camera observed a landmark at a step) and
        ``common_grid`` is ``grid`` if given, else the sorted union of input
        timestamps, shape ``(G,)``.
    """
    device_ids = list(trajectories.keys())
    if not device_ids:
        return (
            np.empty((0, len(landmark_ids), 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    if grid is not None and np.asarray(grid).size:
        all_ts = np.sort(np.asarray(grid, dtype=np.int64))
        _steps = np.diff(all_ts)
        _tol = int(np.median(_steps) // 2) if _steps.size else 0
    else:
        all_ts = np.unique(
            np.concatenate([np.asarray(t, dtype=np.int64) for t in timestamps.values()])
        )
        all_ts = np.sort(all_ts)
        _tol = 0
    G = all_ts.shape[0]
    L = len(landmark_ids)

    resampled: dict[str, np.ndarray] = {}
    resampled_w: dict[str, np.ndarray] = {}
    for dev_id in device_ids:
        pts = np.asarray(trajectories[dev_id], dtype=np.float32)
        ts = np.asarray(timestamps[dev_id], dtype=np.int64)
        order = np.argsort(ts)
        ts_s = ts[order]
        resampled[dev_id] = _resample_to_grid(pts[order], ts_s, all_ts)
        # A camera only "observes" a grid step it actually captured near in
        # time. Mask everything else as NaN so the average below ignores frames
        # it never saw (resampling would otherwise fabricate a vote at
        # timestamps where only another camera saw the hand). With an exact
        # union grid that's an equality test; with an explicit grid it's
        # "nearest real sample within half a grid period" and ``src`` picks the
        # weight from that sample.
        if _tol > 0 and ts_s.size:
            pos = np.clip(np.searchsorted(ts_s, all_ts), 1, ts_s.size - 1)
            dl = np.abs(all_ts - ts_s[pos - 1])
            dr = np.abs(all_ts - ts_s[pos])
            src = np.where(dl <= dr, pos - 1, pos)
            observed = np.minimum(dl, dr) <= _tol
        else:
            observed = np.isin(all_ts, ts_s)
            src = np.clip(np.searchsorted(ts_s, all_ts), 0, max(ts_s.size - 1, 0))
        resampled[dev_id][~observed] = np.nan

        if weights is not None and dev_id in weights:
            w = np.asarray(weights[dev_id], dtype=np.float64)[order]  # (Tc, L)
            w_grid = np.full((G, L), np.nan, dtype=np.float64)
            w_grid[observed] = w[src[observed]]
            resampled_w[dev_id] = np.clip(w_grid, 0.0, None)
        else:
            resampled_w[dev_id] = np.ones((G, L), dtype=np.float64)

    fused = np.full((G, L, 3), np.nan, dtype=np.float32)
    stacked = np.stack([resampled[d] for d in device_ids], axis=0)  # (C, G, L, 3)
    w_stack = np.stack([resampled_w[d] for d in device_ids], axis=0)  # (C, G, L)
    # Drop a camera's vote where it has no point or a non-positive weight.
    valid = np.isfinite(stacked).all(axis=3) & np.isfinite(w_stack) & (w_stack > 0)
    w_stack = np.where(valid, w_stack, 0.0)
    with np.errstate(invalid="ignore"):
        num = np.nansum(stacked * w_stack[..., None], axis=0)  # (G, L, 3)
        den = np.sum(w_stack, axis=0)  # (G, L)
    present = den > 0
    fused[present] = (num[present] / den[present][..., None]).astype(np.float32)

    return fused, all_ts.astype(np.int64)
