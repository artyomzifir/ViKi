#!/usr/bin/env python3
"""
Standalone 3D visualisation of extrinsics + retargeting trajectory.

Generates a static Matplotlib plot saved to scripts/extrinsics_viz.png.
Most logic is delegated to viki.viz.robot_viz_shared so that the server
stream generator and this script stay in sync.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from viki.config import RETARGET_DEFAULT_ROBOT
from viki.viz.robot_viz_shared import (
    EXTRINSICS_FILE,
    camera_gaze_dir,
    camera_world_pos,
    get_neutral_ee,
    get_reach_radius,
    get_robot_world_pos,
    load_debug_data,
    load_extrinsics,
    load_skeleton_wrist,
    project_to_sphere,
    resolve_robot_alias,
    to_robot_frame,
)

SQUARE_SIZE_MULTIPLIER = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, help="Path to skeleton .npz recording")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame for trajectory plot")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame for trajectory plot")
    parser.add_argument("--robot", type=str, default=RETARGET_DEFAULT_ROBOT, help="Robot alias")
    parser.add_argument("--debug-file", type=str, default="data/retarget_debug.json",
                        help="Path to retargeting debug JSON")
    args = parser.parse_args()

    print(f"Loading extrinsics from {EXTRINSICS_FILE}...")
    extrinsics = load_extrinsics(EXTRINSICS_FILE)
    if extrinsics is None:
        print(f"Error: File {EXTRINSICS_FILE} not found.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ── 1. World Origin (Board) ────────────────────────────────────────
    ax.scatter(0, 0, 0, c="red", marker="s", s=100, label="World Origin (Board)")
    plane_range = 0.2 * SQUARE_SIZE_MULTIPLIER
    px = np.linspace(-plane_range, plane_range, 10)
    py = np.linspace(-plane_range, plane_range, 10)
    px, py = np.meshgrid(px, py)
    pz = np.zeros_like(px)
    ax.plot_surface(px, py, pz, alpha=0.2, color="gray")

    # ── 2. Cameras ─────────────────────────────────────────────────────
    for entry in extrinsics:
        dev_id = entry.get("device_id", "unknown")
        rvec = entry.get("rvec", [0, 0, 0])
        tvec = entry.get("tvec", [0, 0, 0])
        cp = camera_world_pos(rvec, tvec)
        ax.scatter(cp[0], cp[1], cp[2], s=100, label=f"Camera {dev_id}")
        ax.text(cp[0], cp[1], cp[2], f" {dev_id}")
        gd = camera_gaze_dir(rvec)
        ax.quiver(cp[0], cp[1], cp[2],
                  gd[0], gd[1], gd[2],
                  length=0.3, color="blue", arrow_length_ratio=0.1)

    # ── 3. Robot Landmarks ─────────────────────────────────────────────
    robot_alias = resolve_robot_alias(args.robot)
    reach_radius = get_reach_radius(robot_alias)
    base_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    base_world = get_robot_world_pos(base_offset)

    ax.scatter(base_world[0], base_world[1], base_world[2],
               c="black", marker="s", s=200, label="Robot Base (Board)")

    p_neutral_robot = get_neutral_ee(robot_alias)
    p_neutral_board = get_robot_world_pos(p_neutral_robot)
    ax.scatter(p_neutral_board[0], p_neutral_board[1], p_neutral_board[2],
               c="orange", marker="*", s=150, label="Robot Neutral EE")

    # Workspace sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    xs = reach_radius * np.cos(u) * np.sin(v) + base_world[0]
    ys = reach_radius * np.sin(u) * np.sin(v) + base_world[1]
    zs = reach_radius * np.cos(v) + base_world[2]
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.5)

    if args.sample:
        sample_path = Path(args.sample)
        if sample_path.exists():
            wrist_positions, frame_count = load_skeleton_wrist(sample_path)

            start_f = args.start_frame if args.start_frame is not None else args.frame
            end_f = args.end_frame if args.end_frame is not None else args.frame + 1
            start_f = max(0, min(start_f, frame_count - 1))
            end_f = max(start_f + 1, min(end_f, frame_count))

            WRIST_0 = wrist_positions[0]

            # Transform trajectory
            traj_world = wrist_positions[start_f:end_f]
            traj_robot = []
            for pw in traj_world:
                pr = to_robot_frame(pw, WRIST_0, p_neutral_robot)
                traj_robot.append(get_robot_world_pos(pr))
            traj_robot = np.array(traj_robot)

            # Project EE positions onto reach sphere
            traj_robot_proj = np.array([
                project_to_sphere(p, base_world, reach_radius)[0] for p in traj_robot
            ])

            ax.plot(traj_world[:, 0], traj_world[:, 1], traj_world[:, 2],
                    c="magenta", label="Human Wrist Path")
            ax.plot(traj_robot_proj[:, 0], traj_robot_proj[:, 1], traj_robot_proj[:, 2],
                    c="cyan", label="Robot EE Path")

            # Current frame
            frame_idx = min(args.frame, len(wrist_positions) - 1)
            p_w = wrist_positions[frame_idx]
            p_r = get_robot_world_pos(
                project_to_sphere(
                    get_robot_world_pos(to_robot_frame(p_w, WRIST_0, p_neutral_robot)),
                    base_world, reach_radius,
                )[0]
            )

            ax.scatter(p_w[0], p_w[1], p_w[2], c="magenta", marker="o", s=100, label="Human Wrist (Current)")
            ax.scatter(p_r[0], p_r[1], p_r[2], c="cyan", marker="X", s=100, label="Robot EE (Current)")
            ax.plot([base_world[0], p_r[0]], [base_world[1], p_r[1]], [base_world[2], p_r[2]],
                    color="black", linestyle="--", alpha=0.6)
        else:
            print(f"Error: Sample file {args.sample} not found.")

    # ── 4. Debug overlay ───────────────────────────────────────────────
    if os.path.exists(args.debug_file):
        debug = load_debug_data(Path(args.debug_file))
        if debug:
            dw = np.array(debug.get("world_positions", []), dtype=np.float64)
            dr = np.array(debug.get("robot_positions", []), dtype=np.float64)
            if len(dw) > 0 and len(dr) > 0:
                ax.plot(dw[:, 0], dw[:, 1], dw[:, 2], c="darkorange", linewidth=2,
                        linestyle="--", label="Debug: Human Wrist")
                dr_proj = np.array([
                    project_to_sphere(p, base_world, reach_radius)[0] for p in dr
                ])
                ax.plot(dr_proj[:, 0], dr_proj[:, 1], dr_proj[:, 2], c="darkblue", linewidth=2,
                        linestyle="--", label="Debug: Robot EE Target")
                rbo = np.array(debug.get("robot_base_offset", [0, 0, 0]), dtype=np.float64)
                ax.scatter(rbo[0], rbo[1], rbo[2], c="black", marker="s", s=180,
                           label="Debug: Robot Base")
                if len(dr_proj) > 0:
                    ax.plot([rbo[0], dr_proj[0, 0]], [rbo[1], dr_proj[0, 1]], [rbo[2], dr_proj[0, 2]],
                            color="black", linestyle=":", alpha=0.4)
                print(f"[Debug] Loaded {len(dw)} frames from {args.debug_file}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Extrinsics & Retargeting Check")
    ax.legend(loc="upper left", fontsize=8)

    # Auto-scale
    all_pts = [np.array([0, 0, 0]), base_world, p_neutral_board]
    if args.sample and "wrist_positions" in dir():
        all_pts.append(wrist_positions[0])
        all_pts.append(wrist_positions[-1])
    stacked = np.stack(all_pts)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = max((hi - lo).max(), 2.0) * 0.6
    center = (lo + hi) / 2.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    plt.savefig("scripts/extrinsics_viz.png")
    print("Plot saved to scripts/extrinsics_viz.png")


if __name__ == "__main__":
    main()
