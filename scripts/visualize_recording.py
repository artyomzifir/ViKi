#!/usr/bin/env python3
"""
Visualize a skeleton recording with a live animation window: hand trajectories
+ depth debug markers, played back frame by frame.

Reads a recording produced by ``SkeletonRecorder`` (e.g.
``data/skeleton_recs/rec-21.23-14.07.2026.npz``) and opens an interactive
matplotlib window that animates the hand (wrist) position and the per‑camera
depth diagnostics over time. No files are written.

Controls
--------
    space           play / pause
    left / right    step one frame back / forward
    q / Esc         close the window

Usage
-----
    python scripts/visualize_recording.py data/skeleton_recs/rec-21.23-14.07.2026.npz
    python scripts/visualize_recording.py <file.npz> --fps 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

# Prefer an interactive backend so a window actually opens; fall back silently
# if none is available (e.g. headless), in which case --save is the only option.
for _be in ("TkAgg", "QtAgg", "GTKAgg", "MacOSX"):
    try:
        matplotlib.use(_be, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


CAM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
HOLE_COLOR = "#cc0000"


def load_recording(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def camera_palette(rec: dict) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (cams, colors_per_frame, color_map) for the recorded frames."""
    device_ids = list(rec["device_ids"])
    cams = sorted(set(device_ids))
    color_map = {cam: CAM_COLORS[i % len(CAM_COLORS)] for i, cam in enumerate(cams)}
    colors = [color_map[d] for d in device_ids]
    return cams, colors, color_map


def summarize(rec: dict, cams: list[str]) -> None:
    dev = rec["device_ids"]
    W = rec["points"][:, 0, :]
    vv = rec.get("depth_valid_fraction")
    wd = rec.get("hand_wrist_depth_m")
    print(f"frames: {len(dev)}  cams: {cams}")
    print(f"per-camera frame counts: {{{', '.join(f'{c}:{int((dev==c).sum())}' for c in cams)}}}")
    if len(W) > 1:
        diff = np.linalg.norm(np.diff(W, axis=0), axis=1)
        switch = np.array(dev[1:] != dev[:-1])
        big = diff > 0.05
        print(f"wrist jumps >5cm: {int(big.sum())}  (at camera switch: {int((big & switch).sum())}, "
              f"within same camera: {int((big & ~switch).sum())})")
        print(f"max wrist jump: {float(np.nanmax(diff)):.3f} m")
    if vv is not None:
        cols = {c: list(cams).index(c) for c in cams}
        for c in cams:
            idx = np.where(dev == c)[0]
            if len(idx) == 0:
                continue
            vf = vv[idx, cols[c]]
            w = wd[idx, cols[c]] if wd is not None else np.full(len(idx), np.nan)
            print(f"  {c}: valid_frac med={np.nanmedian(vf):.3f} min={np.nanmin(vf):.3f} | "
                  f"wrist_depth med={np.nanmedian(w):.3f} m")


def build_figure(rec: dict, cams: list[str], colors: list[str]):
    dev = rec["device_ids"]
    W = rec["points"][:, 0, :]
    ts = rec["timestamps"]
    t = (ts - ts[0]) / 1e6 if len(ts) else np.arange(len(W))
    cols = {c: list(cams).index(c) for c in cams}

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(7, 1, height_ratios=[2, 1, 1, 1, 1, 1, 1], hspace=0.35)

    # --- 3D trajectory ---
    ax3d = fig.add_subplot(gs[0], projection="3d")
    mask = np.isfinite(W).all(axis=1)
    Wv = W[mask]
    ax3d.scatter(Wv[:, 0], Wv[:, 1], Wv[:, 2], c=np.array(colors)[mask].tolist(),
                 s=16, depthshade=False)
    for i in range(1, len(W)):
        if np.isfinite(W[i - 1]).all() and np.isfinite(W[i]).all():
            ax3d.plot([W[i - 1, 0], W[i, 0]], [W[i - 1, 1], W[i, 1]],
                      [W[i - 1, 2], W[i, 2]], color="0.6", lw=0.5, alpha=0.6)
    if len(Wv):
        lo, hi = np.nanmin(Wv, axis=0), np.nanmax(Wv, axis=0)
        c = (lo + hi) / 2
        h = max(np.nanmax(hi - lo) / 2, 1e-3)
        ax3d.set_xlim(c[0] - h, c[0] + h)
        ax3d.set_ylim(c[1] - h, c[1] + h)
        ax3d.set_zlim(c[2] - h, c[2] + h)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Hand (wrist) 3D trajectory — coloured by source camera")

    # moving marker (3D)
    (marker3d,) = ax3d.plot([0], [0], [0], "ko", ms=12, mec="white", zorder=5)

    # --- wrist X/Y/Z time series ---
    ax_xyz = [fig.add_subplot(gs[r + 1]) for r in range(3)]
    xyz_handles = []
    for j, axis in enumerate("XYZ"):
        ax = ax_xyz[j]
        ax.scatter(t, W[:, j], c=colors, s=12)
        ax.set_ylabel(f"wrist {axis} (m)")
        ax.grid(True, alpha=0.3)
        (mk,) = ax.plot([t[0]], [W[0, j]], "ko", ms=8, mec="white", zorder=5)
        (vl,) = ax.plot([t[0], t[0]], ax.get_ylim(), "k--", lw=0.8, alpha=0.5)
        xyz_handles.append((mk, vl))
    ax_xyz[-1].set_xlabel("time (s)")
    leg = [Line2D([0], [0], marker="o", linestyle="", color=colors[list(dev).index(c)]
                   if c in list(dev) else "#000", label=c) for c in cams]
    ax_xyz[0].legend(handles=leg, fontsize=8, loc="upper right")

    # --- depth debug ---
    vv = rec.get("depth_valid_fraction")
    md = rec.get("depth_median_m")
    wd_arr = rec.get("hand_wrist_depth_m")
    ax_depth = [fig.add_subplot(gs[r + 4]) for r in range(3)]
    depth_handles = []  # (marker, vline, per-camera artist list)

    def panel(ax, arr, title, ylabel, is_frac=False):
        per_cam = {}
        for c in cams:
            col = cols[c]
            idx = np.where(dev == c)[0]
            if len(idx) == 0:
                continue
            ax.plot(t[idx], arr[idx, col], color=CAM_COLORS[list(cams).index(c) % len(CAM_COLORS)],
                    marker=".", markersize=4, linestyle="-", lw=1, label=c)
            per_cam[c] = col
        if is_frac:
            for c in cams:
                col = cols[c]
                idx = np.where(dev == c)[0]
                low = idx[arr[idx, col] < 0.1]
                if len(low):
                    ax.scatter(t[low], arr[low, col], color=HOLE_COLOR, s=22, zorder=5,
                               label="hole (<0.1)" if c == cams[0] else None)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        (mk,) = ax.plot([t[0]], [arr[0, cols[dev[0]]] if np.isfinite(arr[0, cols[dev[0]]]) else np.nan],
                        "ko", ms=8, mec="white", zorder=5)
        (vl,) = ax.plot([t[0], t[0]], ax.get_ylim(), "k--", lw=0.8, alpha=0.5)
        return mk, vl

    mk_v, vl_v = panel(ax_depth[0], vv, "depth_valid_fraction (red = <0.1, depth hole)",
                       "fraction", is_frac=True)
    mk_m, vl_m = panel(ax_depth[1], md, "depth_median_m", "depth (m)")
    mk_w, vl_w = panel(ax_depth[2], wd_arr, "hand_wrist_depth_m (depth at wrist → drives position)",
                       "depth (m)")
    ax_depth[-1].set_xlabel("time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    state = {
        "t": t, "W": W, "dev": dev, "cols": cols, "cams": cams, "vv": vv,
        "md": md, "wd": wd_arr,
        "marker3d": marker3d, "xyz": xyz_handles, "depth": [(mk_v, vl_v), (mk_m, vl_m), (mk_w, vl_w)],
        "ax_xyz": ax_xyz, "ax_depth": ax_depth,
    }
    return fig, state


def update(frame: int, state: dict):
    t = state["t"]; W = state["W"]; dev = state["dev"]; cols = state["cols"]
    cam = dev[frame]
    col = cols[cam]
    # 3D marker
    if np.isfinite(W[frame]).all():
        state["marker3d"].set_data([W[frame, 0]], [W[frame, 1]])
        state["marker3d"].set_3d_properties([W[frame, 2]])
        state["marker3d"].set_visible(True)
    else:
        state["marker3d"].set_visible(False)
    # wrist xyz
    for j, (mk, vl) in enumerate(state["xyz"]):
        mk.set_data([t[frame]], [W[frame, j]])
        vl.set_xdata([t[frame], t[frame]])
        if j == 0:
            ax = state["ax_xyz"][0]
            vl.set_ydata(ax.get_ylim())
    # depth markers
    darrs = [state["vv"], state["md"], state["wd"]]
    for k, (mk, vl) in enumerate(state["depth"]):
        arr = darrs[k]
        if arr is not None and np.isfinite(arr[frame, col]):
            mk.set_data([t[frame]], [arr[frame, col]])
        else:
            mk.set_data([], [])
        vl.set_xdata([t[frame], t[frame]])
        ax = state["ax_depth"][k]
        vl.set_ydata(ax.get_ylim())
    return [state["marker3d"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="recording .npz file")
    ap.add_argument("--fps", type=int, default=12, help="animation playback speed")
    args = ap.parse_args()

    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    rec = load_recording(path)
    cams, colors, _ = camera_palette(rec)
    summarize(rec, cams)

    fig, state = build_figure(rec, cams, colors)
    fig.suptitle(path.name, fontsize=10)

    n = len(rec["device_ids"])
    state["_frame"] = 0
    state["_playing"] = True

    def draw(i):
        state["_frame"] = i % n
        update(state["_frame"], state)
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, draw, frames=n, interval=1000 / args.fps,
                         blit=False, repeat=True)

    # --- controls ---
    def on_key(event):
        if event.key == " ":
            if anim.running:
                anim.event_source.stop()
                state["_playing"] = False
            else:
                anim.event_source.start()
                state["_playing"] = True
        elif event.key == "right":
            anim.event_source.stop()
            draw(state["_frame"] + 1)
        elif event.key == "left":
            anim.event_source.stop()
            draw(state["_frame"] - 1)
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Playing. Controls: space=play/pause, ←/→=step, q/Esc=close")
    plt.show()


if __name__ == "__main__":
    main()
