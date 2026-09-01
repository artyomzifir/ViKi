"""
viki.render.skeleton3d
----------------------
Headless 3-D matplotlib figures of an episode's hand data — a quick sanity check
on the kinematics without the interactive web viewer.

``stage="rec"``  per-camera raw landmark clouds + per-camera wrist tracks
``stage="cln"``  fused wrist trajectory + palm-frame triads + a few hand poses
"""

from __future__ import annotations

import numpy as np

from viki import config
from viki.contracts import cln_pose_keys

_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
_CAM_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]


def _new_ax():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax


def _equal_aspect(ax, pts: np.ndarray) -> None:
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return
    c = pts.mean(axis=0)
    r = float(np.max(np.abs(pts - c))) or 0.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def _draw_bones(ax, frame_pts: np.ndarray, color: str, alpha: float = 0.6) -> None:
    for a, b in _HAND_BONES:
        pa, pb = frame_pts[a], frame_pts[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            ax.plot(*zip(pa, pb), color=color, lw=1.4, alpha=alpha)


def _draw_triad(ax, origin: np.ndarray, R: np.ndarray, scale: float) -> None:
    for k, col in enumerate(("r", "g", "b")):
        tip = origin + R[:, k] * scale
        ax.plot(*zip(origin, tip), color=col, lw=2.0)


def render_episode_figure(ep, stage: str = "cln"):
    """Return a matplotlib ``Figure`` for episode ``ep`` (``stage`` = rec|cln)."""
    fig, ax = _new_ax()

    if stage == "rec":
        if not ep.rec_npz.exists():
            raise FileNotFoundError(f"no rec.npz for episode {ep.id}")
        with np.load(ep.rec_npz) as d:
            pts = np.asarray(d["points"], dtype=np.float64)  # (N, 21, 3)
            devs = [str(x) for x in d["device_ids"]]
        all_pts = pts.reshape(-1, 3)
        for i, dev in enumerate(sorted(set(devs))):
            mask = np.array([x == dev for x in devs])
            cam_pts = pts[mask].reshape(-1, 3)
            cam_pts = cam_pts[np.isfinite(cam_pts).all(axis=1)]
            col = _CAM_COLORS[i % len(_CAM_COLORS)]
            ax.scatter(*cam_pts.T, s=4, alpha=0.25, color=col, label=dev)
            wrists = pts[mask][:, 0, :]
            wrists = wrists[np.isfinite(wrists).all(axis=1)]
            if len(wrists) > 1:
                ax.plot(*wrists.T, color=col, lw=1.5)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{ep.id} — rec.npz (per-camera, pre-fusion)")
        _equal_aspect(ax, all_pts)
        return fig

    if stage == "cln":
        if not ep.cln_npz.exists():
            raise FileNotFoundError(f"no cln.npz for episode {ep.id}")
        with np.load(ep.cln_npz) as d:
            hand = np.asarray(d["smoothed_points"], dtype=np.float64)  # (T, 21, 3)
            p_key, r_key = cln_pose_keys(
                d.files, getattr(config, "PERCEPTION_HAND_POSE_SOURCE", "landmarks")
            )
            pos = np.asarray(d[p_key], dtype=np.float64)  # (T, 3)
            rot = np.asarray(d[r_key], dtype=np.float64)  # (T, 3, 3)
            valid = np.asarray(d["valid"], dtype=bool)
        T = len(pos)
        track = pos[np.isfinite(pos).all(axis=1)]
        if len(track) > 1:
            ax.plot(*track.T, color="#333333", lw=1.8, label="wrist trajectory")
        scale = 0.03
        if len(track) > 1:
            scale = float(np.linalg.norm(track.max(0) - track.min(0))) * 0.06 or 0.03
        idx = np.linspace(0, T - 1, min(T, 10)).astype(int)
        for i in idx:
            if valid[i] and np.isfinite(rot[i]).all():
                _draw_triad(ax, pos[i], rot[i], scale)
        for i in (idx[0], idx[len(idx) // 2], idx[-1]):
            _draw_bones(ax, hand[i], "#4363d8", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{ep.id} — cln.npz (fused, {valid.sum()}/{T} valid)")
        _equal_aspect(ax, hand.reshape(-1, 3))
        return fig

    raise ValueError(f"stage must be 'rec' or 'cln', got {stage!r}")
