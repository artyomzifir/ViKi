"""Non-destructive storage for validated episode baselines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from functools import lru_cache
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
    return {
        name: file_sha256(path)
        for name, path in candidates.items()
        if path.is_file()
    }


@lru_cache(maxsize=1)
def code_sha256() -> str:
    """Fingerprint of the ``viki`` source tree that produced an artifact.

    Notebook rule 3 asks every comparison to record its code revision. Git is
    not available here - only ``viki/`` is mounted into the container, not the
    repository - and a commit id would in any case not describe a working tree
    with uncommitted edits, which is the state most experiments actually run
    in. Hashing the sources is both available and more precise.

    This exists because a protected baseline pins its profile and its inputs
    but, until now, nothing about the code: two artifacts could differ with an
    identical manifest, and on 2026-09-09 two of them did.
    """
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def _added_keys(target: Path, source: Path) -> set[str]:
    """Keys present in ``source`` but not in the already protected artifact."""
    with np.load(target, allow_pickle=False) as t, np.load(
        source, allow_pickle=False
    ) as s:
        return set(s.files) - set(t.files)


def protect_baseline(
    ep, profile, source: Path | None = None, *, additive_refresh: bool = False,
) -> dict[str, object]:
    """Save the first validated artifact for this profile and never replace it.

    A later rerun can still replace the episode's active ``cln.npz``.  The
    protected copy remains byte-for-byte stable; its manifest hash makes an
    accidental modification detectable.

    ``additive_refresh`` is the one sanctioned exception, and it does not weaken
    what immutability is for. A profile may declare an output that is produced
    *after* the landmark trajectory has been protected - the articulated hand
    fit is one - and a write-once artifact could never receive it. The result
    was that a protected baseline silently lacked one of its own profile's
    declared outputs, while the episode's active ``cln.npz`` had it: any
    experiment run off the baseline then used a different pose source than
    production, which is what happened throughout the 2026-09-09 comparisons.

    A refresh is accepted only when it *adds*: every core array must be
    bit-identical and the new artifact must be a strict superset of the old.
    The measurement cannot change; only a declared output may land. Both hashes
    are kept in the manifest so the substitution stays auditable.
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
    def _write(previous: dict | None, added: set[str] | None = None) -> None:
        tmp = root / f".cln.tmp-{os.getpid()}.npz"
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        payload = {
            "schema": 2,
            "profile": profile.manifest(),
            "created_at": (previous or {}).get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "artifact": "cln.npz",
            "artifact_sha256": source_hash,
            "inputs_sha256": _input_hashes(ep),
            "code_sha256": code_sha256(),
        }
        if previous is not None:
            # Keep the superseded identity: the artifact grew, and the record
            # of what it grew from is the point of protecting it at all.
            payload["additive_refresh"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "previous_artifact_sha256": previous.get("artifact_sha256"),
                "previous_code_sha256": previous.get("code_sha256"),
                # Computed before the copy below replaced the target: after
                # it, source and target are identical and the difference is
                # gone.
                "added_keys": sorted(added or ()),
            }
        atomic_write_json(manifest_path, payload)

    if not target.exists():
        _write(None)
    elif additive_refresh:
        added = _added_keys(target, source)
        if added and _same_core_arrays(target, source):
            _write(json.loads(manifest_path.read_text()), added)
        elif added:
            raise RuntimeError(
                f"refusing to refresh protected baseline {target}: it would add "
                f"{sorted(added)} but the core trajectory arrays changed"
            )

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
