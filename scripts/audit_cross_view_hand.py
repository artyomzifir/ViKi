"""Read-only two-camera hand self-consistency audit over recorded episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from viki.contracts import HAND_LM_COUNT, LM


def _summary(distances_mm: np.ndarray) -> dict:
    return {
        "joint_pairs": int(distances_mm.size),
        "median_mm": float(np.median(distances_mm)) if distances_mm.size else None,
        "p95_mm": float(np.percentile(distances_mm, 95)) if distances_mm.size else None,
    }


def audit_episode(episode_dir: Path) -> tuple[dict, np.ndarray]:
    with np.load(episode_dir / "rec.npz", allow_pickle=False) as data:
        device_ids = np.asarray(data["device_ids"]).astype(str)
        timestamps = np.asarray(data["timestamps"], dtype=np.int64)
        points = np.asarray(data["points"], dtype=np.float64)
        landmark_ids = np.asarray(data["landmark_ids"], dtype=np.int64)
    if len(points) != len(timestamps) or len(points) != len(device_ids):
        raise ValueError(f"{episode_dir.name}: rec row counts differ")
    if points.shape[1:] != (HAND_LM_COUNT, 3) or not np.array_equal(
        landmark_ids, np.arange(HAND_LM_COUNT)
    ):
        raise ValueError(f"{episode_dir.name}: unexpected hand joint order")
    camera_ids = sorted(set(device_ids))
    if len(camera_ids) != 2:
        raise ValueError(f"{episode_dir.name}: expected two camera tracks, got {camera_ids}")
    ticks = json.loads((episode_dir / "raw/timestamps.json").read_text())
    lookups = {}
    ambiguous = {}
    for camera in camera_ids:
        lookup = {}
        duplicates = set()
        for row_index in np.flatnonzero(device_ids == camera):
            tick = int(timestamps[row_index])
            if tick in lookup:
                duplicates.add(tick)
            lookup[tick] = points[row_index]
        lookups[camera] = lookup
        expected = [
            int(frame["sync_us"]) + int(frame.get("offsets_us", {}).get(camera, 0))
            for frame in ticks
        ]
        expected_counts = Counter(expected)
        ambiguous[camera] = duplicates | {
            tick for tick, count in expected_counts.items() if count > 1
        }
    distances = []
    tip_distances = []
    frame_pairs = 0
    ambiguous_frames_skipped = 0
    tips = [int(lm) for lm in (
        LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP
    )]
    for frame in ticks:
        selected = []
        frame_ambiguous = False
        for camera in camera_ids:
            tick = int(frame["sync_us"]) + int(
                frame.get("offsets_us", {}).get(camera, 0)
            )
            frame_ambiguous |= tick in ambiguous[camera]
            selected.append(lookups[camera].get(tick))
        if frame_ambiguous:
            ambiguous_frames_skipped += 1
            continue
        if any(track is None for track in selected):
            continue
        frame_pairs += 1
        valid = np.isfinite(selected[0]).all(axis=1) & np.isfinite(selected[1]).all(axis=1)
        if not valid.any():
            continue
        difference = np.linalg.norm(selected[0] - selected[1], axis=1) * 1000.0
        distances.extend(difference[valid])
        tip_distances.extend(difference[tips][valid[tips]])
    values = np.asarray(distances, dtype=np.float64)
    return {
        "episode": episode_dir.name,
        "cameras": camera_ids,
        "capture_frames": len(ticks),
        "paired_frames": frame_pairs,
        "ambiguous_timestamp_frames_skipped": ambiguous_frames_skipped,
        "paired_frame_fraction": frame_pairs / len(ticks) if ticks else None,
        "joint_pair_fraction": len(values) / (len(ticks) * HAND_LM_COUNT) if ticks else None,
        "all_joints": _summary(values),
        "fingertips": _summary(np.asarray(tip_distances)),
    }, values


def audit_dataset(dataset: Path, manifest: Path) -> dict:
    source = json.loads(manifest.read_text())
    episode_names = [row["episode"] for row in source["episodes"]]
    if len(episode_names) != len(set(episode_names)) or len(episode_names) != source["episode_count"]:
        raise ValueError("manifest episode count is inconsistent")
    episodes = []
    distances = []
    for name in episode_names:
        result, values = audit_episode(dataset / name)
        episodes.append(result)
        distances.append(values)
    all_distances = np.concatenate(distances) if distances else np.empty(0)
    return {
        "definition": "same-frame Euclidean separation of finite rec.npz world-frame joints",
        "evidence": "cross-view self-consistency, not independent 3-D ground truth",
        "episode_count": len(episodes),
        "capture_frames": sum(row["capture_frames"] for row in episodes),
        "paired_frames": sum(row["paired_frames"] for row in episodes),
        "all_joints": _summary(all_distances),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/40mm-pnp"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/experiments/2026-10-09-full-audit.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/experiments/cross-view-hand-audit.json")
    )
    args = parser.parse_args()
    result = audit_dataset(args.dataset, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
