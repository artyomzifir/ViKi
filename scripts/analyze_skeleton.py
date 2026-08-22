import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BoneConfig:
    name: str
    p1: int
    p2: int

# Define the bones to analyze based on viki.skeleton.models.LM
BONES = [
    BoneConfig("Wrist-Elbow", 0, 21),
    BoneConfig("Elbow-Shoulder", 21, 22),
    BoneConfig("Wrist-IndexTip", 0, 8),
    BoneConfig("Wrist-MiddleTip", 0, 12),
    BoneConfig("Wrist-PinkyTip", 0, 20),
]

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

def analyze_session(file_path: str, ground_truths: Dict[str, float] = None):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # data is a list of frames: {"ts": ..., "landmarks": [[x,y,z], ...], "confidence": [...]}
    num_frames = len(data)
    bone_data = {bone.name: [] for bone in BONES}
    timestamps = []

    for frame in data:
        lms_raw = frame['landmarks']
        ts = frame['ts']

        # Handle landmarks as either a list or a dictionary (string keys)
        if isinstance(lms_raw, dict):
            lms = {}
            for k, v in lms_raw.items():
                lms[int(k)] = np.array(v)
        else:
            lms = {i: np.array(p) for i, p in enumerate(lms_raw)}

        for bone in BONES:
            # Ensure both landmarks exist and are not NaN
            if bone.p1 in lms and bone.p2 in lms:
                p1 = lms[bone.p1]
                p2 = lms[bone.p2]
                
                if not np.any(np.isnan(p1)) and not np.any(np.isnan(p2)):
                    dist = calculate_distance(p1, p2)
                    bone_data[bone.name].append(dist)

        timestamps.append(ts)

    print(f"\nAnalysis for {file_path}")
    print(f"Total frames: {num_frames}")
    print("-" * 80)
    print(f"{'Bone':<20} | {'Mean (m)':<10} | {'StdDev (m)':<12} | {'CV (%)':<10} | {'Error (m)':<10}")
    print("-" * 80)

    plt.figure(figsize=(12, 8))

    for i, bone in enumerate(BONES):
        dists = np.array(bone_data[bone.name])
        if len(dists) == 0:
            print(f"{bone.name:<20} | No valid data")
            continue

        mean_len = np.mean(dists)
        std_len = np.std(dists)
        cv = (std_len / mean_len) * 100 if mean_len != 0 else 0
        
        error_str = "N/A"
        if ground_truths and bone.name in ground_truths:
            mae = np.mean(np.abs(dists - ground_truths[bone.name]))
            error_str = f"{mae:.4f}"

        print(f"{bone.name:<20} | {mean_len:<10.4f} | {std_len:<12.4f} | {cv:<10.2f} | {error_str:<10}")

        # Plotting
        # Note: since some frames might miss some bones, we just plot the sequence of valid measurements
        plt.subplot(len(BONES), 1, i+1)
        plt.plot(dists, label=f"{bone.name} (CV: {cv:.2f}%)")
        plt.ylabel("Length (m)")
        plt.legend(loc="upper right")
        
        # Fix for zero-variance plots: force a reasonable Y-axis range and disable scalar offset
        if std_len < 1e-6:
            plt.ylim(mean_len * 0.9, mean_len * 1.1)
            plt.ticklabel_format(useOffset=False, style='plain')
            
        if i == 0:
            plt.title(f"Bone Length Stability over Time - {file_path}")

    plt.xlabel("Valid Frame Index")
    plt.tight_layout()
    
    output_plot = file_path.replace(".json", "_analysis.png")
    plt.savefig(output_plot)
    print("-" * 80)
    print(f"Plot saved to: {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze skeleton measurement accuracy from recorded JSON sessions.")
    parser.add_argument("file", help="Path to the recorded JSON file")
    parser.add_argument("--gt", nargs=2, action='append', help="Ground truth length for a bone. Usage: --gt 'Wrist-Elbow' 0.32")
    
    args = parser.parse_args()
    
    gt_dict = {}
    if args.gt:
        for name, val in args.gt:
            gt_dict[name] = float(val)
            
    analyze_session(args.file, gt_dict)
