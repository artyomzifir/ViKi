
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from viki.skeleton.processor import SkeletonProcessor
from viki.skeleton.models import LM, EndEffectorPose
from viki.skeleton.hand_angles import compute_end_effector_pose # For raw RPY calculation
import viki.config as config

# Temporary override for config paths to allow script to run standalone
# In a real scenario, this should be handled by proper config loading or environment variables
if not hasattr(config, "SKELETON_RECS_DIR"):
    config.SKELETON_RECS_DIR = "data/skeleton_recs"
    config.SKELETON_SMOOTHED_DIR = "data/skeleton_smoothed"

def compute_bone_lengths(points_seq: np.ndarray, bone_pairs: list[tuple[LM, LM]]) -> np.ndarray:
    T = len(points_seq)
    num_bones = len(bone_pairs)
    lengths = np.full((T, num_bones), np.nan, dtype=np.float32)
    for t in range(T):
        for i, (lm1, lm2) in enumerate(bone_pairs):
            v1 = points_seq[t, lm1.value]
            v2 = points_seq[t, lm2.value]
            if np.isfinite(v1).all() and np.isfinite(v2).all():
                lengths[t, i] = float(np.linalg.norm(v1 - v2))
    return lengths

def detect_outliers_ema(
    bone_lengths_sequence: np.ndarray,
    ema_alpha: float = 0.1,
    tolerance_factor: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects outliers in bone lengths using EMA tracking.
    Returns:
        - ema_values: (T, num_bones)
        - is_outlier: (T, num_bones) boolean mask
        - filtered_lengths: (T, num_bones) bone lengths with outliers set to NaN
    """
    T, num_bones = bone_lengths_sequence.shape
    ema_values = np.full_like(bone_lengths_sequence, np.nan)
    is_outlier = np.zeros_like(bone_lengths_sequence, dtype=bool)
    filtered_lengths = bone_lengths_sequence.copy()

    current_ema = np.full(num_bones, np.nan)

    for t in range(T):
        for i in range(num_bones):
            current_val = bone_lengths_sequence[t, i]

            if np.isnan(current_val):
                ema_values[t, i] = current_ema[i]
                continue

            if np.isnan(current_ema[i]):
                current_ema[i] = current_val
                ema_values[t, i] = current_ema[i]
            else:
                lower_bound = (1.0 - tolerance_factor) * current_ema[i]
                upper_bound = (1.0 + tolerance_factor) * current_ema[i]

                if lower_bound < current_val < upper_bound:
                    current_ema[i] = ema_alpha * current_val + (1.0 - ema_alpha) * current_ema[i]
                    ema_values[t, i] = current_ema[i]
                else:
                    is_outlier[t, i] = True
                    filtered_lengths[t, i] = np.nan # Mark as outlier
                    ema_values[t, i] = current_ema[i] # EMA does not update
    return ema_values, is_outlier, filtered_lengths

def plot_outlier_detection(
    timestamps: np.ndarray,
    raw_bone_lengths: np.ndarray,
    raw_ema_values: np.ndarray,
    raw_is_outlier: np.ndarray,
    smoothed_bone_lengths: np.ndarray,
    bone_labels: list[str],
    output_dir: Path
):
    num_bones = raw_bone_lengths.shape[1]
    fig, axes = plt.subplots(num_bones, 1, figsize=(15, 5 * num_bones), sharex=True)
    if num_bones == 1:
        axes = [axes] # Ensure axes is iterable for single bone

    for i in range(num_bones):
        ax = axes[i]
        ax.plot(timestamps, raw_bone_lengths[:, i], 'b.', label='Raw Length', alpha=0.5)
        ax.plot(timestamps, raw_ema_values[:, i], 'r-', label='EMA (Raw Data)', linewidth=1)
        
        # Plot smoothed bone lengths for comparison
        ax.plot(timestamps, smoothed_bone_lengths[:, i], 'g--', label='Smoothed Length', alpha=0.7)

        outlier_times = timestamps[raw_is_outlier[:, i]]
        outlier_lengths = raw_bone_lengths[raw_is_outlier[:, i], i]
        ax.plot(outlier_times, outlier_lengths, 'rx', label='Outlier', markersize=8)

        ax.set_title(f'Bone Length Outlier Detection: {bone_labels[i]}')
        ax.set_ylabel('Length (m)')
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel('Timestamp (us)')
    plt.tight_layout()
    plt.savefig(output_dir / "bone_length_outlier_detection.png")
    plt.close()

def plot_arm_movement_3d(
    timestamps: np.ndarray,
    raw_points_seq: np.ndarray,
    smoothed_positions: np.ndarray, # (T, 3) for wrist
    smoothed_rotations: np.ndarray, # (T, 3, 3) for palm rotation matrix
    output_dir: Path,
    key_lms: list[LM]
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Find common min/max for axes for consistent scaling
    all_x = np.concatenate([raw_points_seq[:, key_lm.value, 0].flatten() for key_lm in key_lms] + [smoothed_positions[:, 0].flatten()])
    all_y = np.concatenate([raw_points_seq[:, key_lm.value, 1].flatten() for key_lm in key_lms] + [smoothed_positions[:, 1].flatten()])
    all_z = np.concatenate([raw_points_seq[:, key_lm.value, 2].flatten() for key_lm in key_lms] + [smoothed_positions[:, 2].flatten()])
    
    min_val = min(np.nanmin(all_x), np.nanmin(all_y), np.nanmin(all_z))
    max_val = max(np.nanmax(all_x), np.nanmax(all_y), np.nanmax(all_z))

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)

    # Plot trajectories for raw data
    for lm in key_lms:
        ax.plot(raw_points_seq[:, lm.value, 0], raw_points_seq[:, lm.value, 1], raw_points_seq[:, lm.value, 2],
                '--', alpha=0.5, label=f'Raw {lm.name} Trajectory')
    
    # Plot smoothed wrist trajectory
    ax.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], smoothed_positions[:, 2],
            'g-', label='Smoothed Wrist Trajectory', linewidth=2)

    # Plot start and end points
    ax.plot([raw_points_seq[0, LM.WRIST.value, 0]], [raw_points_seq[0, LM.WRIST.value, 1]], [raw_points_seq[0, LM.WRIST.value, 2]],
            'bo', markersize=8, label='Raw Start (Wrist)')
    ax.plot([raw_points_seq[-1, LM.WRIST.value, 0]], [raw_points_seq[-1, LM.WRIST.value, 1]], [raw_points_seq[-1, LM.WRIST.value, 2]],
            'ko', markersize=8, label='Raw End (Wrist)')
    
    ax.plot([smoothed_positions[0, 0]], [smoothed_positions[0, 1]], [smoothed_positions[0, 2]],
            'go', markersize=8, label='Smoothed Start (Wrist)')
    ax.plot([smoothed_positions[-1, 0]], [smoothed_positions[-1, 1]], [smoothed_positions[-1, 2]],
            'ro', markersize=8, label='Smoothed End (Wrist)')

    # Add orientation arrows for smoothed wrist (e.g., every 10th frame)
    arrow_skip = max(1, len(timestamps) // 20) # Show about 20 arrows
    for t in range(0, len(timestamps), arrow_skip):
        pos = smoothed_positions[t]
        rot = smoothed_rotations[t] # R_world_palm

        # X-axis (forward) of palm frame in world
        ax.quiver(pos[0], pos[1], pos[2], rot[0,0], rot[1,0], rot[2,0], color='r', length=0.05, arrow_length_ratio=0.5)
        # Y-axis (up) of palm frame in world
        ax.quiver(pos[0], pos[1], pos[2], rot[0,1], rot[1,1], rot[2,1], color='g', length=0.05, arrow_length_ratio=0.5)
        # Z-axis (normal) of palm frame in world
        ax.quiver(pos[0], pos[1], pos[2], rot[0,2], rot[1,2], rot[2,2], color='b', length=0.05, arrow_length_ratio=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Arm Movement and Wrist Orientation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "arm_movement_3d.png")
    plt.close()

def compute_raw_end_effector_poses(
    raw_points_seq: np.ndarray,
    raw_landmark_ids: np.ndarray,
    timestamps: np.ndarray,
    key_lms_for_pose: list[LM]
) -> list[EndEffectorPose]:
    T, L, _ = raw_points_seq.shape
    raw_poses: list[EndEffectorPose] = []

    for t in range(T):
        current_mapping = {LM(int(raw_landmark_ids[i])): raw_points_seq[t, i] for i in range(L)}
        filtered = {}
        ok = True
        for lm in key_lms_for_pose:
            p = current_mapping.get(lm)
            if p is None or not np.all(np.isfinite(p)):
                ok = False
                break
            filtered[lm] = p
        if ok:
            pose = compute_end_effector_pose(filtered, int(timestamps[t]))
        else:
            pose = EndEffectorPose(
                position=np.full(3, np.nan, dtype=np.float32),
                R_world_palm=np.full((3, 3), np.nan, dtype=np.float32),
                rpy_deg=np.full(3, np.nan, dtype=np.float32),
                valid=False,
                timestamp_us=int(timestamps[t]),
            )
        raw_poses.append(pose)
    return raw_poses


def plot_error_graphs(
    timestamps: np.ndarray,
    raw_positions: np.ndarray, # (T, 3) for wrist
    raw_rpy: np.ndarray,      # (T, 3) for wrist
    smoothed_positions: np.ndarray, # (T, 3) for wrist
    smoothed_rpy: np.ndarray,       # (T, 3) for wrist
    output_dir: Path
):
    # Position error (Euclidean distance between raw and smoothed wrist positions)
    position_diff = np.linalg.norm(raw_positions - smoothed_positions, axis=1)

    # RPY error (Absolute difference)
    rpy_diff = np.abs(raw_rpy - smoothed_rpy)

    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    # Plot position difference
    axes[0].plot(timestamps, position_diff, 'm-', label='Position Difference (Raw vs Smoothed)')
    axes[0].set_title('Wrist Position Difference (Raw vs Smoothed)')
    axes[0].set_ylabel('Distance (m)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot RPY differences
    axes[1].plot(timestamps, rpy_diff[:, 0], 'c-', label='Roll Difference (deg)')
    axes[1].set_title('Wrist Roll Difference (Raw vs Smoothed)')
    axes[1].set_ylabel('Degrees')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(timestamps, rpy_diff[:, 1], 'y-', label='Pitch Difference (deg)')
    axes[2].set_title('Wrist Pitch Difference (Raw vs Smoothed)')
    axes[2].set_ylabel('Degrees')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(timestamps, rpy_diff[:, 2], 'k-', label='Yaw Difference (deg)')
    axes[3].set_title('Wrist Yaw Difference (Raw vs Smoothed)')
    axes[3].set_ylabel('Degrees')
    axes[3].legend()
    axes[3].grid(True)
    
    axes[-1].set_xlabel('Timestamp (us)')
    plt.tight_layout()
    plt.savefig(output_dir / "error_calculation_graphs.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton recording data before and after smoothing.")
    parser.add_argument("input_file", type=str, help="Path to the input raw recording file (e.g., rec-YYYYMMDD_HHMMSS.npz)")
    parser.add_argument("--output_dir", type=str, default="data/visualizations", help="Directory to save visualization plots.")
    parser.add_argument("--window_length", type=int, default=7, help="Window length for Savitzky-Golay filter.")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for Savitzky-Golay filter.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return

    processor = SkeletonProcessor()
    
    # 1. Load raw data
    with np.load(input_path) as data:
        raw_timestamps = data["timestamps"]
        raw_points = data["points"] # (T, L, 3)
        raw_landmark_ids = data["landmark_ids"]

    # Ensure raw_points is float and contains NaN for missing values
    if not np.issubdtype(raw_points.dtype, np.floating):
        raw_points = raw_points.astype(np.float32)

    # 2. Smooth data (this also computes end-effector poses for smoothed data)
    print(f"Smoothing recording '{input_path.name}'...")
    try:
        smoothed_file_path_str, smoothed_points = processor.smooth_recording(
            input_path.name,
            window_length=args.window_length,
            polyorder=args.polyorder
        )
        smoothed_file_path = Path(smoothed_file_path_str)
        print(f"Smoothed data saved to '{smoothed_file_path}'")
    except Exception as e:
        print(f"Error during smoothing: {e}")
        return

    with np.load(smoothed_file_path) as data:
        smoothed_timestamps = data["timestamps"]
        smoothed_positions = data["positions"] # (T, 3) - wrist positions
        smoothed_rotations = data["rotations"] # (T, 3, 3) - R_world_palm
        smoothed_rpy = data["rpy"]           # (T, 3) - RPY in deg
        smoothed_valid = data["valid"]        # (T,) - validity mask

    # --- Prepare data for plotting ---
    
    # Bone definitions for outlier detection
    tracked_bones = [
        (LM.SHOULDER, LM.ELBOW),
        (LM.ELBOW, LM.WRIST),
    ]
    bone_labels = [f"{lm1.name}-{lm2.name}" for lm1, lm2 in tracked_bones]

    # Compute bone lengths for raw and smoothed data
    raw_bone_lengths = compute_bone_lengths(raw_points, tracked_bones)
    smoothed_bone_lengths = compute_bone_lengths(smoothed_points, tracked_bones)

    raw_ema_values, raw_is_outlier, _ = detect_outliers_ema(raw_bone_lengths)

    # Compute raw end-effector poses for error calculation
    print("Computing raw end-effector poses...")
    key_lms_for_pose = [LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP]
    raw_poses = compute_raw_end_effector_poses(raw_points, raw_landmark_ids, raw_timestamps, key_lms_for_pose)
    raw_ee_positions = np.array([p.position for p in raw_poses])
    raw_ee_rpy = np.array([p.rpy_deg for p in raw_poses])


    print("Generating plots...")
    plot_outlier_detection(
        raw_timestamps,
        raw_bone_lengths,
        raw_ema_values,
        raw_is_outlier,
        smoothed_bone_lengths,
        bone_labels,
        output_dir
    )

    plot_arm_movement_3d(
        raw_timestamps, # Timestamps are the same for raw and smoothed
        raw_points,
        smoothed_positions,
        smoothed_rotations,
        output_dir,
        key_lms=[LM.WRIST, LM.ELBOW, LM.SHOULDER]
    )

    plot_error_graphs(
        raw_timestamps,
        raw_ee_positions,
        raw_ee_rpy,
        smoothed_positions,
        smoothed_rpy,
        output_dir
    )

    print(f"Visualization plots saved to '{output_dir}'")

if __name__ == "__main__":
    main()
