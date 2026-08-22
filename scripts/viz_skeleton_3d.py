#! /usr/bin/env python3
"""
3D skeleton visualizer for .npz recordings.

Usage:
    python scripts/viz_skeleton_3d.py                                    # pick latest
    python scripts/viz_skeleton_3d.py data/skeleton_recs/rec-XX.XX.npz
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

HAND_CONNS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

LM_NAMES = [
    "Wrist", "ThumbCMC", "ThumbMCP", "ThumbIP", "ThumbTip",
    "IndexMCP", "IndexPIP", "IndexDIP", "IndexTip",
    "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip",
    "RingMCP", "RingPIP", "RingDIP", "RingTip",
    "PinkyMCP", "PinkyPIP", "PinkyDIP", "PinkyTip",
    "Elbow", "Shoulder",
]

FINGER_COLORS = {
    0: "#aaaaaa",
    1: "#ff5555", 2: "#ff5555", 3: "#ff5555", 4: "#ff5555",
    5: "#55ff55", 6: "#55ff55", 7: "#55ff55", 8: "#55ff55",
    9: "#5555ff", 10: "#5555ff", 11: "#5555ff", 12: "#5555ff",
    13: "#ffff55", 14: "#ffff55", 15: "#ffff55", 16: "#ffff55",
    17: "#ff55ff", 18: "#ff55ff", 19: "#ff55ff", 20: "#ff55ff",
    21: "#888888", 22: "#888888",
}


def _nearest_valid(points: np.ndarray, frame: int, lm: int) -> np.ndarray | None:
    """Search forward then backward for a non-NaN landmark."""  
    N = len(points)
    for dr in range(N):
        for sign in (1, -1):
            t = frame + dr * sign
            if 0 <= t < N:
                p = points[t, lm]
                if np.all(np.isfinite(p)):
                    return p
    return None


REC_DIRS = [Path("data/skeleton_recs"), Path("data/skeleton_smoothed")]


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    data = np.load(path)
    timestamps = data["timestamps"]
    landmark_ids = data["landmark_ids"]
    if "smoothed_points" in data:
        return timestamps, data["smoothed_points"], landmark_ids, "smoothed"
    if "points" in data:
        return timestamps, data["points"], landmark_ids, "raw"
    if "raw_points" in data:
        return timestamps, data["raw_points"], landmark_ids, "cleaned"
    raise ValueError(f"Unknown .npz format (keys: {list(data.keys())})")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D skeleton recording")
    parser.add_argument("path", nargs="?", type=str, help="Path to .npz file")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval in ms (default 50)")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS")
    args = parser.parse_args()

    if args.path:
        npz_path = Path(args.path)
    else:
        npz_files = sorted(p for d in REC_DIRS for p in d.glob("*.npz"))
        if not npz_files:
            print("No .npz files found in data/skeleton_recs/ or data/skeleton_smoothed/", file=sys.stderr)
            sys.exit(1)
        npz_path = npz_files[-1]

    if not npz_path.exists():
        print(f"File not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    timestamps, points, landmark_ids, fmt = _load_npz(str(npz_path))
    timestamps_s = (timestamps - timestamps[0]) / 1e6
    N, n_lm, _ = points.shape
    print(f"Frames: {N}, Landmarks: {n_lm}, File: {npz_path.name}")
    nan_per_frame = np.sum(np.any(~np.isfinite(points), axis=2), axis=1)
    print(f"NaN frames (any lm): {np.sum(nan_per_frame > 0)}/{N}")
    print(f"Per-landmark NaN count: {np.sum(np.any(~np.isfinite(points), axis=2), axis=0)}")

    interval = args.interval
    if args.fps:
        interval = int(1000 / args.fps)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 0.35])

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")

    paused = [False]
    current_frame = [0]
    wrist_traj: list[np.ndarray] = []

    dummy = np.full((n_lm, 3), np.nan)
    scat = ax.scatter(dummy[:, 0], dummy[:, 1], dummy[:, 2], c="gray", s=30, alpha=0.5)
    lines = []
    for _ in HAND_CONNS:
        (ln,) = ax.plot([], [], [], "b-", alpha=0.4, lw=1.5)
        lines.append(ln)
    (wrist_line,) = ax.plot([], [], [], "r-", alpha=0.3, lw=1)
    (wrist_dot,) = ax.plot([], [], [], "ro", markersize=5)

    fig.subplots_adjust(bottom=0.12)
    ax_play = fig.add_axes([0.45, 0.02, 0.08, 0.04])
    btn_play = Button(ax_play, "Pause")
    play_label = ["Pause"]

    def toggle_pause(_):
        paused[0] = not paused[0]
        play_label[0] = "Play" if paused[0] else "Pause"
        btn_play.label.set_text(play_label[0])
        if not paused[0]:
            ani.event_source.start()
        else:
            ani.event_source.stop()

    btn_play.on_clicked(toggle_pause)

    def init():
        return []

    def get_bounds():
        valid = points[np.all(np.isfinite(points), axis=2)]
        if len(valid) == 0:
            return (-0.5, 0.5, -0.5, 0.5, 0, 3)
        xmin, xmax = float(np.min(valid[:, 0])), float(np.max(valid[:, 0]))
        ymin, ymax = float(np.min(valid[:, 1])), float(np.max(valid[:, 1]))
        zmin, zmax = float(np.min(valid[:, 2])), float(np.max(valid[:, 2]))
        r = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
        cx, cy, cz = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
        return (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)

    xl, xr, yl, yr, zl, zr = get_bounds()

    def update(frame):
        current_frame[0] = frame
        pts = points[frame]
        valid_mask = np.all(np.isfinite(pts), axis=1)

        # Scatter plot
        cols = [FINGER_COLORS.get(i, "#888888") for i in range(n_lm)]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        scat.set_facecolor(cols)
        scat.set_edgecolor("none")

        # Connections
        for i, (a, b) in enumerate(HAND_CONNS):
            if valid_mask[a] and valid_mask[b]:
                lines[i].set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
                lines[i].set_3d_properties([pts[a, 2], pts[b, 2]])
                lines[i].set_alpha(0.6)
            else:
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
                lines[i].set_alpha(0)

        # Wrist trajectory
        if valid_mask[0]:
            wrist_traj.append(pts[0].copy())
            if len(wrist_traj) > 500:
                wrist_traj.pop(0)
        traj_arr = np.array(wrist_traj) if wrist_traj else np.empty((0, 3))
        if len(traj_arr) > 1:
            wrist_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
            wrist_line.set_3d_properties(traj_arr[:, 2])
        else:
            wrist_line.set_data([], [])
            wrist_line.set_3d_properties([])

        if valid_mask[0]:
            wrist_dot.set_data([pts[0, 0]], [pts[0, 1]])
            wrist_dot.set_3d_properties([pts[0, 2]])
        else:
            wrist_dot.set_data([], [])
            wrist_dot.set_3d_properties([])

        ax.set_xlim(xl, xr)
        ax.set_ylim(yl, yr)
        ax.set_zlim(zl, zr)

        t_sec = timestamps_s[frame]
        nans = int(nan_per_frame[frame])
        ax.set_title(f"Frame {frame}/{N}  t={t_sec:.1f}s  NaN={nans}/{n_lm}")

        # Info panel
        ax_info.clear()
        ax_info.axis("off")
        nan_lms = [LM_NAMES[i] for i in range(n_lm) if not np.all(np.isfinite(pts[i]))]
        info_lines = [f"File: {npz_path.name}", f"Frames: {N}", f"Frame: {frame}/{N}", f"Time: {t_sec:.2f}s"]
        if nan_lms:
            info_lines.append(f"NaN LMs: {', '.join(nan_lms)}")
        else:
            info_lines.append("All landmarks valid")
        if valid_mask[0]:
            wp = pts[0]
            info_lines.append(f"Wrist: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
        ax_info.text(0, 0.95, "\n".join(info_lines), fontsize=10, verticalalignment="top",
                     fontfamily="monospace", transform=ax_info.transAxes)

        return [scat, wrist_line, wrist_dot] + lines

    ani = FuncAnimation(fig, update, frames=N, init_func=init, interval=interval, blit=False, repeat=True)

    def on_key(event):
        if event.key == "left":
            f = max(0, current_frame[0] - 1)
            ani.event_source.stop()
            update(f)
            fig.canvas.draw_idle()
        elif event.key == "right":
            f = min(N - 1, current_frame[0] + 1)
            ani.event_source.stop()
            update(f)
            fig.canvas.draw_idle()
        elif event.key == " ":
            toggle_pause(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
