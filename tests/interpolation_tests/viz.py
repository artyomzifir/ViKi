import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_skeleton(ax, landmarks, connections, color):
    ax.clear()
    # Determine plot limits from data to keep it stable
    # In a real scenario, we'd pre-calculate bounds across all frames

    points = {}
    for j_id, coords in landmarks.items():
        if not any(np.isnan(coords)):
            points[j_id] = coords

    # Plot joints
    for j_id, coords in points.items():
        ax.scatter(coords[0], coords[1], coords[2], color=color, s=20)

    # Plot connections
    for i in range(len(connections) - 1):
        start_id = str(connections[i])
        end_id = str(connections[i + 1])
        if start_id in points and end_id in points:
            p1 = points[start_id]
            p2 = points[end_id]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2
            )


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python visualize_interpolation.py <before.json> <after.json> <connections_csv>"
        )
        print(
            "Example: python visualize_interpolation.py input.json output.json 0,21,22"
        )
        sys.exit(1)

    before_path = sys.argv[1]
    after_path = sys.argv[2]
    connections = [int(x) for x in sys.argv[3].split(",")]

    before_data = load_json(before_path)
    after_data = load_json(after_path)

    # Find global bounds for consistent axes
    all_points = []
    for dataset in [before_data, after_data]:
        for frame in dataset:
            for coords in frame["landmarks"].values():
                if not any(np.isnan(coords)):
                    all_points.append(coords)

    all_points = np.array(all_points)
    min_bounds = all_points.min(axis=0)
    max_bounds = all_points.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    max_range = (max_bounds - min_bounds).max() / 2

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    ax1.set_title("Before Interpolation")
    ax2.set_title("After Interpolation")

    def update(frame_idx):
        # Wrap around if datasets have different lengths, though they should be same
        idx_before = frame_idx % len(before_data)
        idx_after = frame_idx % len(after_data)

        plot_skeleton(ax1, before_data[idx_before]["landmarks"], connections, "blue")
        plot_skeleton(ax2, after_data[idx_after]["landmarks"], connections, "green")

        for ax in [ax1, ax2]:
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    num_frames = max(len(before_data), len(after_data))
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
