from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viki.viz.mjpeg import mjpeg_chunk, placeholder
from viki.viz.smooth_viz_shared import SmoothVizConfig, extract_wrist


def smooth_trajectory_stream(
    npz_path: Path,
    cfg: SmoothVizConfig | None = None,
) -> Iterator[bytes]:
    if cfg is None:
        cfg = SmoothVizConfig()

    if not npz_path.exists():
        while True:
            yield mjpeg_chunk(placeholder(640, 480, f"File not found: {npz_path.name}"))
            time.sleep(1)

    with np.load(npz_path) as data:
        positions = data["positions"]
        timestamps = data["timestamps"]
        raw_points = data.get("raw_points")
        landmark_ids = data.get("landmark_ids")

    smooth_wrist = np.asarray(positions, dtype=np.float64)
    n_frames = len(smooth_wrist)

    fps = 15.0
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = dt[dt > 0]
        if len(dt) > 0:
            med = float(np.median(dt))
            scale = 1_000_000.0 if med > 1_000.0 else 1.0
            fps = float(1.0 / np.median(dt / scale))

    raw_wrist: np.ndarray | None = None
    if raw_points is not None and landmark_ids is not None:
        raw_wrist = extract_wrist(raw_points, landmark_ids)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_pts: list[np.ndarray] = [smooth_wrist[0], smooth_wrist[-1]]
    if raw_wrist is not None:
        all_pts.extend([raw_wrist[0], raw_wrist[-1]])

    stacked = np.stack(all_pts)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = max((hi - lo).max(), cfg.axes_length) * 0.6
    centre_at = np.zeros(3)

    (smooth_trail,) = ax.plot([], [], [], c="#2ecc71", linewidth=2.5, label="Smoothed")
    (smooth_dot,) = ax.plot([], [], [], "o", c="#2ecc71", ms=8)

    raw_trail: Any = None
    raw_dot: Any = None
    if raw_wrist is not None:
        (raw_trail,) = ax.plot(
            [], [], [], c="#e67e22", linewidth=1.5, alpha=0.7, linestyle="--", label="Original"
        )
        (raw_dot,) = ax.plot([], [], [], "o", c="#e67e22", ms=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Smoothing — {npz_path.stem}")
    f_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)
    legend = ax.legend(loc="upper left", fontsize=8)

    interval_s = 1.0 / max(fps, 1)
    frame_idx = 0

    while True:
        if frame_idx >= n_frames:
            yield mjpeg_chunk(placeholder(640, 480, "Done — trajectory complete"))
            time.sleep(2)
            continue

        ax.set_xlim(centre_at[0] - span, centre_at[0] + span)
        ax.set_ylim(centre_at[1] - span, centre_at[1] + span)
        ax.set_zlim(centre_at[2] - span, centre_at[2] + span)

        smooth_trail.set_visible(cfg.show_smooth)
        smooth_dot.set_visible(cfg.show_smooth)
        if raw_trail is not None:
            raw_trail.set_visible(cfg.show_raw)
        if raw_dot is not None:
            raw_dot.set_visible(cfg.show_raw)

        if cfg.show_smooth:
            end = min(frame_idx + 1, n_frames)
            seg = smooth_wrist[:end]
            smooth_trail.set_data(seg[:, 0], seg[:, 1])
            smooth_trail.set_3d_properties(seg[:, 2])
            smooth_dot.set_data([seg[-1, 0]], [seg[-1, 1]])
            smooth_dot.set_3d_properties([seg[-1, 2]])
        else:
            smooth_trail.set_data([], [])
            smooth_trail.set_3d_properties([])
            smooth_dot.set_data([], [])
            smooth_dot.set_3d_properties([])

        if raw_wrist is not None and cfg.show_raw:
            end = min(frame_idx + 1, len(raw_wrist))
            seg = raw_wrist[:end]
            raw_trail.set_data(seg[:, 0], seg[:, 1])
            raw_trail.set_3d_properties(seg[:, 2])
            raw_dot.set_data([seg[-1, 0]], [seg[-1, 1]])
            raw_dot.set_3d_properties([seg[-1, 2]])
        elif raw_trail is not None:
            raw_trail.set_data([], [])
            raw_trail.set_3d_properties([])
            raw_dot.set_data([], [])
            raw_dot.set_3d_properties([])

        f_text.set_text(f"Frame {frame_idx + 1} / {n_frames}")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, [2, 1, 0]]
        yield mjpeg_chunk(img, 85)
        frame_idx += 1
        time.sleep(interval_s)
