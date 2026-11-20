"""Copy one HO-Cap interval into independent hand/object retarget episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRACK_KEYS = (
    "schema", "object_ids", "model_offsets", "model_xyz_object",
    "translation_world", "rotation_world_object", "track_confidence",
)
FRAME_TRACK_KEYS = {"translation_world", "rotation_world_object", "track_confidence"}


def prepare(source: Path, hand_output: Path, object_output: Path,
            first: int, last: int) -> None:
    if first < 0 or last <= first:
        raise ValueError("frame interval must be non-empty and non-negative")
    metadata = json.loads((source / "meta.json").read_text())
    if metadata.get("source") != "hocap" or metadata.get("hocap_world_frame") != "tag_1":
        raise ValueError("pilot source must be HO-Cap in tag_1 world coordinates")
    track_path = source / "intermediates/segmentation/hocap-oracle-v1/object_models.npz"
    with np.load(source / "cln.npz", allow_pickle=False) as prepared:
        frame_count = len(prepared["timestamps"])
        if last > frame_count or str(prepared["coordinate_frame"]) != "hocap_tag_1":
            raise ValueError("pilot interval or prepared coordinate frame is invalid")
        hand_arrays = {
            name: (value[first:last] if value.ndim and value.shape[0] == frame_count else value)
            for name, value in prepared.items()
        }
    with np.load(track_path, allow_pickle=False) as tracked:
        if len(tracked["translation_world"]) != frame_count:
            raise ValueError("object and hand tracks have different frame counts")
        track_arrays = {
            name: (tracked[name][first:last] if name in FRAME_TRACK_KEYS else tracked[name])
            for name in TRACK_KEYS
        }
    if hand_output.exists() or object_output.exists():
        raise FileExistsError("pilot output already exists; refusing to overwrite")
    hand_output.mkdir(parents=True)
    object_output.mkdir(parents=True)
    try:
        for output in (hand_output, object_output):
            (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
            np.savez_compressed(output / "cln.npz", **hand_arrays)
        np.savez_compressed(object_output / "object_track.npz", **track_arrays)
    except Exception:
        for output in (hand_output, object_output):
            for filename in ("meta.json", "cln.npz", "object_track.npz"):
                (output / filename).unlink(missing_ok=True)
            output.rmdir()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("hand_output", type=Path)
    parser.add_argument("object_output", type=Path)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.hand_output, arguments.object_output,
            arguments.first, arguments.last)
