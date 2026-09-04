"""Non-destructive storage for validated episode baselines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from viki.prepare.checkpoints import atomic_write_json


_CORE_ARRAYS = (
    "positions",
    "rotations",
    "valid",
    "omega",
    "gripper",
    "timestamps",
    "raw_points",
    "smoothed_points",
    "landmark_confidence",
    "landmark_ids",
    "coordinate_frame",
    "perception_fuse_mode",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(ep) -> dict[str, str]:
    candidates = {
        "rec.npz": ep.rec_npz,
        "raw/observations.npz": ep.raw_dir / "observations.npz",
        "raw/observations_meta.json": ep.raw_dir / "observations_meta.json",
        "raw/joints3d.npz": ep.raw_dir / "joints3d.npz",
        "raw/joints3d_summary.json": ep.raw_dir / "joints3d_summary.json",
        "raw/timestamps.json": ep.raw_dir / "timestamps.json",
        "raw/intrinsics.json": ep.raw_dir / "intrinsics.json",
        "raw/extrinsics.json": ep.raw_dir / "extrinsics.json",
    }


def _same_core_arrays(left: Path, right: Path) -> bool:
    """Compare trajectory content while ignoring added provenance arrays."""
    with np.load(left, allow_pickle=False) as left_npz, np.load(
        right, allow_pickle=False
    ) as right_npz:
        if any(key not in left_npz.files or key not in right_npz.files
               for key in _CORE_ARRAYS):
            return False
        for key in _CORE_ARRAYS:
            a = left_npz[key]
            b = right_npz[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if np.issubdtype(a.dtype, np.inexact):
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            elif not np.array_equal(a, b):
                return False
    return True
    return {
        name: file_sha256(path)
        for name, path in candidates.items()
        if path.is_file()
    }


def protect_baseline(ep, profile, source: Path | None = None) -> dict[str, object]:
    """Save the first validated artifact for this profile and never replace it.

    A later rerun can still replace the episode's active ``cln.npz``.  The
    protected copy remains byte-for-byte stable; its manifest hash makes an
    accidental modification detectable.
    """
    source = Path(source or ep.cln_npz)
    if not source.is_file():
        raise FileNotFoundError(source)
    root = ep.intermediates_dir / "baselines" / profile.name
    target = root / "cln.npz"
    manifest_path = root / "manifest.json"
    source_hash = file_sha256(source)

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    if target.exists() != manifest_path.exists():
        raise RuntimeError(
            f"incomplete protected baseline at {root}: cln.npz and "
            "manifest.json must either both exist or both be absent"
        )
    if not target.exists():
        tmp = root / f".cln.tmp-{os.getpid()}.npz"
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        atomic_write_json(manifest_path, {
            "schema": 1,
            "profile": profile.manifest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact": "cln.npz",
            "artifact_sha256": source_hash,
            "inputs_sha256": _input_hashes(ep),
        })

    protected_hash = file_sha256(target)
    manifest = json.loads(manifest_path.read_text())
    recorded_hash = manifest.get("artifact_sha256")
    if recorded_hash != protected_hash:
        raise RuntimeError(
            f"protected baseline hash mismatch at {target}: "
            f"manifest={recorded_hash}, actual={protected_hash}"
        )
    if manifest.get("profile") != profile.manifest():
        raise RuntimeError(
            f"profile definition changed for protected baseline {target}; "
            "create a new versioned profile instead of mutating it"
        )
    matches_bytes = protected_hash == source_hash
    matches_core = matches_bytes or _same_core_arrays(target, source)
    return {
        "path": str(target),
        "sha256": protected_hash,
        # ``matches_active`` retains its user-facing meaning: the actual
        # trajectory is unchanged even if a newer file adds metadata arrays.
        "matches_active": matches_core,
        "matches_active_core": matches_core,
        "matches_active_bytes": matches_bytes,
    }
