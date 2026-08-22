import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description


def get_joint_positions(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    positions = []
    for name in model.names:
        if name == "universe":
            continue
        frame_id = model.getFrameId(name)
        pos = data.oMf[frame_id].translation.copy()
        positions.append(pos)
    return np.array(positions)


def update_plot(frame_idx, model, data, q_all, joint_positions_all, lines, traj_line, ax, f_text, ee_positions):
    joint_positions = joint_positions_all[frame_idx]
    ee_pos = ee_positions[frame_idx]

    for i, line in enumerate(lines):
        if i < len(joint_positions) - 1:
            line.set_data(
                [joint_positions[i][0], joint_positions[i + 1][0]],
                [joint_positions[i][1], joint_positions[i + 1][1]],
            )
            line.set_3d_properties(
                [joint_positions[i][2], joint_positions[i + 1][2]]
            )
        elif i == len(joint_positions) - 1:
            line.set_data([ee_pos[0]], [ee_pos[1]])
            line.set_3d_properties([ee_pos[2]])

    traj_line.set_data(ee_positions[:frame_idx + 1, 0], ee_positions[:frame_idx + 1, 1])
    traj_line.set_3d_properties(ee_positions[:frame_idx + 1, 2])

    f_text.set_text(f"Frame {frame_idx + 1} / {q_all.shape[0]}")
    return lines + [traj_line, f_text]


def main():
    parser = argparse.ArgumentParser(
        description="Animate iiwa14 robot trajectory in 3D using Pinocchio FK."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/robot_out/smoke_iiwa14_hand_se3.h5",
        help="Path to HDF5 archive (default: data/robot_out/smoke_iiwa14_hand_se3.h5)",
    )
    parser.add_argument(
        "--q-key",
        default="q_scene_raw",
        help="Dataset key for joint angles (default: q_scene_raw)",
    )
    parser.add_argument(
        "--robot",
        default="iiwa14_description",
        help="Robot description name (default: iiwa14_description)",
    )
    parser.add_argument(
        "--ee-frame",
        default="iiwa_link_ee",
        help="End-effector frame name (default: iiwa_link_ee)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Animation interval in ms (default: 100)",
    )
    args = parser.parse_args()

    path = args.input
    print(f"Loading {path} ...")
    with h5py.File(path, "r") as f:
        q_all = f[args.q_key][:]
    n_frames, n_joints = q_all.shape
    print(f"  {n_frames} frames, {n_joints} joints")

    robot = load_robot_description(args.robot)
    model = robot.model
    data = robot.data

    all_positions = []
    for i in range(n_frames):
        pin.forwardKinematics(model, data, q_all[i])
        pin.updateFramePlacements(model, data)
        frame_pos = []
        for name in model.names:
            if name == "universe":
                continue
            fid = model.getFrameId(name)
            frame_pos.append(data.oMf[fid].translation.copy())

        ee_fid = model.getFrameId(args.ee_frame)
        ee_pos = data.oMf[ee_fid].translation.copy()
        frame_pos.append(ee_pos)
        all_positions.append(np.array(frame_pos))

    joint_positions_all = np.array(all_positions)
    ee_positions = joint_positions_all[:, -1, :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    lines = []
    n_pts = joint_positions_all.shape[1]
    for _ in range(n_pts - 1):
        (line,) = ax.plot([], [], [], "o-", color="tab:blue", lw=3, ms=4)
        lines.append(line)
    (line,) = ax.plot([], [], [], "o", color="red", ms=6)
    lines.append(line)

    (traj_line,) = ax.plot([], [], [], "--", color="tab:orange", lw=1.5, alpha=0.7)

    f_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Robot trajectory — {Path(path).name}")

    all_pts_flat = joint_positions_all.reshape(-1, 3)
    margin = 0.1
    x_min, x_max = all_pts_flat[:, 0].min(), all_pts_flat[:, 0].max()
    y_min, y_max = all_pts_flat[:, 1].min(), all_pts_flat[:, 1].max()
    z_min, z_max = all_pts_flat[:, 2].min(), all_pts_flat[:, 2].max()
    half_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 0.3) / 2
    mid = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    ax.set_xlim(mid[0] - half_range - margin, mid[0] + half_range + margin)
    ax.set_ylim(mid[1] - half_range - margin, mid[1] + half_range + margin)
    ax.set_zlim(mid[2] - half_range - margin, mid[2] + half_range + margin)

    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(
        fig,
        update_plot,
        frames=n_frames,
        fargs=(
            model, data, q_all, joint_positions_all,
            lines, traj_line, ax, f_text, ee_positions,
        ),
        interval=args.interval,
        blit=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
