"""
viki.calibration.artifacts
--------------------------
The setup artifacts, each in its own file under ``data/calibrations/<preset>/``
with an independent lifecycle:

``extrinsics.json``
    Rig-rigid pose: ``T_ref_cam`` per camera (camera → reference-camera frame;
    the reference is identity) plus the raw ChArUco observations the solve used.
    Invalidated by moving any camera.

``world_anchor.json``
    ``T_world_display`` — applied **only** to visualisation, the working AABB and
    the top-level export, never to the extrinsics solve, the cloud or hand-fit.
    Carries ``extrinsics_hash`` (the extrinsics file it was computed against) so
    a stale anchor is detectable; recomputed automatically from its stored
    observations when the extrinsics change.

``validation_report.json``
    Cloud-agreement verdict (green / amber / red). Carries ``extrinsics_hash``.

``<device_id>_bg.npz``
    Per-camera empty-scene depth plate: median depth (mm) + a validity mask.
    Backgrounds are a 2-D property of the depth image and do **not** depend on
    the extrinsics — re-solving never invalidates a background.

Legacy ``data/calibrations/<preset>.json`` (a v1 list or a v2 dict) is read and
migrated into this layout on first access (:func:`ensure_migrated`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from viki.calibration.presets import PRESETS_DIR, _safe_name, preset_path

EXTRINSICS_SCHEMA = 2
WORLD_ANCHOR_SCHEMA = 1
VALIDATION_SCHEMA = 1
BACKGROUND_SCHEMA = 2  # v1 had no validity mask

_EXTRINSICS_FILE = "extrinsics.json"
_WORLD_ANCHOR_FILE = "world_anchor.json"
_VALIDATION_FILE = "validation_report.json"


# ── paths ────────────────────────────────────────────────────────────────


def preset_dir(name: str) -> Path:
    d = PRESETS_DIR / _safe_name(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extrinsics_path(name: str) -> Path:
    return preset_dir(name) / _EXTRINSICS_FILE


def _world_anchor_path(name: str) -> Path:
    return preset_dir(name) / _WORLD_ANCHOR_FILE


def _validation_path(name: str) -> Path:
    return preset_dir(name) / _VALIDATION_FILE


def background_path(name: str, device_id: str) -> Path:
    return preset_dir(name) / f"{device_id}_bg.npz"


# ── helpers ──────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump(path: Path, obj: dict) -> None:
    """Write pretty, key-sorted JSON so the file bytes (and therefore the hash)
    are a deterministic function of the content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _mat(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(4, 4)


def as_camera_extrinsics(T_ref_cam: Any) -> dict:
    """A 4x4 ``T_ref_cam`` (camera → reference frame) as the ``{rvec, tvec}`` pair
    whose :pyattr:`~viki.contracts.CalibrationExtrinsics.transform_matrix` equals
    it — i.e. the form the episode ``raw/extrinsics.json`` and the downstream
    cloud/lift code already consume, but now in the rig (reference) frame."""
    import cv2

    T = _mat(T_ref_cam)
    M, c = T[:3, :3], T[:3, 3]
    # transform_matrix builds [R.T | -R.T t]; invert that: R = M.T, t = -M.T c
    R = M.T
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ c
    return {"rvec": rvec.reshape(3).tolist(), "tvec": tvec.reshape(3).tolist()}


# ── extrinsics.json ──────────────────────────────────────────────────────


def write_extrinsics(
    name: str,
    *,
    reference_device: str,
    devices: dict[str, Any],
    sets: list[dict],
    solve: dict,
    intrinsics: dict | None = None,
    board: dict | None = None,
) -> dict:
    """``devices[dev]`` is a 4x4 ``T_ref_cam`` (list-of-lists or ndarray); the
    reference device must be present and (near-)identity. ``sets`` is the list of
    ``{set_id, captured_at, observations: {dev: {charuco_ids, charuco_corners}}}``
    the solve consumed. ``intrinsics`` (``{dev: {fx,fy,cx,cy,dist_coeffs?}}``) and
    ``board`` are stored alongside so the solve and the world anchor can be
    redone offline from this one file."""
    if reference_device not in devices:
        raise ValueError(f"reference_device {reference_device!r} not in devices")
    prev = read_extrinsics(name) or {}
    payload = {
        "schema": EXTRINSICS_SCHEMA,
        "created_at": _now_iso(),
        "reference_device": reference_device,
        "devices": {
            dev: {"T_ref_cam": _mat(T).tolist()} for dev, T in devices.items()
        },
        "sets": sets,
        "solve": solve,
        "intrinsics": intrinsics if intrinsics is not None else prev.get("intrinsics", {}),
        "board": board if board is not None else prev.get("board"),
    }
    _dump(_extrinsics_path(name), payload)
    return payload


def resolve_from_observations(name: str, *, reference_device: str | None = None) -> dict:
    """Re-run the bundle solve from the ``sets`` / ``intrinsics`` / ``board``
    stored in ``extrinsics.json`` and rewrite it in place. Used after a set is
    dropped, or to re-solve with a better solver. The world anchor and the
    validation report go stale (their ``extrinsics_hash`` no longer matches)
    until recomputed."""
    from viki.calibration.bundle import solve_bundle

    data = read_extrinsics(name)
    if not data:
        raise FileNotFoundError(f"no extrinsics.json for preset {name!r}")
    out = solve_bundle(
        data.get("sets", []),
        data.get("intrinsics", {}),
        data.get("board") or {},
        reference_device=reference_device or data.get("reference_device"),
    )
    return write_extrinsics(
        name,
        reference_device=out["reference_device"],
        devices=out["devices"],  # {dev: 4x4 list}
        sets=data.get("sets", []),
        solve=out["solve"],
        intrinsics=data.get("intrinsics", {}),
        board=data.get("board"),
    )


def read_extrinsics(name: str) -> dict | None:
    return _load(_extrinsics_path(name))


def extrinsics_hash(name: str) -> str | None:
    """sha256 of the on-disk ``extrinsics.json`` bytes, or ``None`` if absent."""
    return _sha256(_extrinsics_path(name))


def device_transforms(name: str) -> dict[str, np.ndarray]:
    """``{device_id: T_ref_cam (4x4 ndarray)}`` from ``extrinsics.json``."""
    data = read_extrinsics(name) or {}
    return {
        dev: _mat(entry["T_ref_cam"])
        for dev, entry in (data.get("devices") or {}).items()
    }


def rig_extrinsics(name: str) -> dict[str, dict]:
    """``{device_id: {rvec, tvec}}`` in the rig (reference-camera) frame — the
    shape the episode ``raw/extrinsics.json`` carries and the cloud/lift code
    consumes. No world anchor is applied."""
    return {dev: as_camera_extrinsics(T) for dev, T in device_transforms(name).items()}


# ── world_anchor.json ────────────────────────────────────────────────────


def write_world_anchor(
    name: str,
    *,
    T_world_display: Any,
    observations: dict[str, Any],
    extrinsics_hash_: str | None = None,
) -> dict:
    payload = {
        "schema": WORLD_ANCHOR_SCHEMA,
        "created_at": _now_iso(),
        "extrinsics_hash": extrinsics_hash_ or extrinsics_hash(name),
        "T_world_display": _mat(T_world_display).tolist(),
        "observations": observations,
    }
    _dump(_world_anchor_path(name), payload)
    return payload


def read_world_anchor(name: str) -> dict | None:
    return _load(_world_anchor_path(name))


def world_display_matrix(name: str) -> np.ndarray:
    """``T_world_display`` (4x4), identity if there is no anchor yet."""
    data = read_world_anchor(name)
    if not data or "T_world_display" not in data:
        return np.eye(4)
    return _mat(data["T_world_display"])


# ── validation_report.json ───────────────────────────────────────────────


def write_validation(
    name: str, *, verdict: str, pairs: list[dict], extrinsics_hash_: str | None = None
) -> dict:
    if verdict not in ("green", "amber", "red"):
        raise ValueError(f"verdict must be green/amber/red, got {verdict!r}")
    payload = {
        "schema": VALIDATION_SCHEMA,
        "created_at": _now_iso(),
        "extrinsics_hash": extrinsics_hash_ or extrinsics_hash(name),
        "verdict": verdict,
        "pairs": pairs,
    }
    _dump(_validation_path(name), payload)
    return payload


def read_validation(name: str) -> dict | None:
    return _load(_validation_path(name))


# ── staleness / status ───────────────────────────────────────────────────


def _artifact_state(artifact: dict | None, current_extr_hash: str | None) -> str:
    if artifact is None:
        return "absent"
    if current_extr_hash is None:
        return "orphan"  # artifact exists but no extrinsics.json
    return "ok" if artifact.get("extrinsics_hash") == current_extr_hash else "stale"


def world_anchor_stale(name: str) -> bool:
    return _artifact_state(read_world_anchor(name), extrinsics_hash(name)) == "stale"


def validation_stale(name: str) -> bool:
    return _artifact_state(read_validation(name), extrinsics_hash(name)) == "stale"


def background_devices(name: str) -> list[str]:
    d = PRESETS_DIR / _safe_name(name)
    if not d.is_dir():
        return []
    return sorted(p.name[: -len("_bg.npz")] for p in d.glob("*_bg.npz"))


def artifact_status(name: str) -> dict:
    """Snapshot for the setup wizard: presence + freshness of each artifact."""
    extr = read_extrinsics(name)
    h = extrinsics_hash(name)
    return {
        "preset": _safe_name(name),
        "extrinsics": {
            "state": "present" if extr else "absent",
            "hash": h,
            "reference_device": (extr or {}).get("reference_device"),
            "devices": sorted((extr or {}).get("devices", {}).keys()),
            "n_sets": len((extr or {}).get("sets", [])),
        },
        "world_anchor": {"state": _artifact_state(read_world_anchor(name), h)},
        "validation": {
            "state": _artifact_state(read_validation(name), h),
            "verdict": (read_validation(name) or {}).get("verdict"),
        },
        "background": {"devices": background_devices(name)},
    }


def record_ready(name: str, *, allow_amber: bool = False) -> tuple[bool, str]:
    """Whether a recording may start against this preset (spec §7). Returns
    ``(ok, reason)``; ``reason`` is empty when ``ok``."""
    st = artifact_status(name)
    if st["extrinsics"]["state"] != "present":
        return False, "no extrinsics — run the Calibrate step"
    if st["world_anchor"]["state"] in ("absent", "orphan"):
        return False, "no world anchor — run the Anchor step"
    if st["world_anchor"]["state"] == "stale":
        return False, "world anchor is stale (extrinsics changed) — recapture the Anchor step"
    if not st["background"]["devices"]:
        return False, "no background plate — run the Background step"
    vstate = st["validation"]["state"]
    if vstate in ("absent", "orphan"):
        return False, "no validation report — run the Validate step"
    if vstate == "stale":
        return False, "validation is stale (extrinsics changed) — re-run the Validate step"
    verdict = st["validation"]["verdict"]
    if verdict == "red":
        return False, "validation verdict is red — the camera clouds do not agree; recalibrate"
    if verdict == "amber" and not allow_amber:
        return False, "validation verdict is amber — confirm to record anyway"
    return True, ""


# ── background plate (median depth + validity mask) ──────────────────────


def write_background(
    name: str, plates: dict[str, tuple[Any, Any]]
) -> list[str]:
    """``plates[dev] = (median_depth_mm, valid_mask)``. ``valid_mask`` is a bool
    array, True where enough samples backed the median. Written to
    ``<dev>_bg.npz`` with ``schema``, ``depth_mm`` (float32), ``valid`` (bool)."""
    written: list[str] = []
    for dev, (depth_mm, valid) in plates.items():
        d = np.asarray(depth_mm, dtype=np.float32)
        v = np.asarray(valid, dtype=bool)
        if v.shape != d.shape:
            raise ValueError(f"{dev}: valid mask {v.shape} != depth {d.shape}")
        np.savez_compressed(
            background_path(name, dev),
            schema=np.int32(BACKGROUND_SCHEMA),
            depth_mm=d,
            valid=v,
        )
        written.append(dev)
    return sorted(written)


def read_background(name: str, device_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(depth_mm float32, valid bool)`` for ``device_id``, or ``None``.
    A v1 plate (no mask) is read with ``valid = depth_mm > 0``."""
    p = background_path(name, device_id)
    if not p.is_file():
        return None
    with np.load(p) as z:
        depth = z["depth_mm"].astype(np.float32)
        valid = z["valid"].astype(bool) if "valid" in z.files else (depth > 0)
    return depth, valid


def background_depth_masked(name: str, device_id: str) -> np.ndarray | None:
    """Median depth (mm) with invalid pixels forced to 0 (the no-reading marker
    the rest of the pipeline expects). Compatibility shim for callers that want a
    single array."""
    got = read_background(name, device_id)
    if got is None:
        return None
    depth, valid = got
    out = depth.copy()
    out[~valid] = 0.0
    return out


# ── migration from the legacy single-file preset ─────────────────────────


def _canonical_pose_to_matrix(rvec: Any, tvec: Any) -> np.ndarray:
    """Legacy per-camera ``{rvec, tvec}`` (board→camera, solvePnP) as the 4x4
    camera→world(board) transform — the same maths as
    :pyattr:`CalibrationExtrinsics.transform_matrix`."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def migrate_legacy(name: str) -> dict | None:
    """Build ``extrinsics.json`` + ``world_anchor.json`` from a legacy
    ``data/calibrations/<name>.json`` (v1 list or v2 dict). Returns the new
    extrinsics payload, or ``None`` if there is nothing to migrate."""
    legacy_file = preset_path(name)
    legacy = _load(legacy_file)
    if legacy is None:
        return None
    extr_list = legacy if isinstance(legacy, list) else legacy.get("extrinsics", [])
    if not extr_list:
        return None

    reference_device = extr_list[0]["device_id"]
    world_by_dev = {
        e["device_id"]: _canonical_pose_to_matrix(e["rvec"], e["tvec"])
        for e in extr_list
    }
    T_world_ref = world_by_dev[reference_device]
    T_ref_world = np.linalg.inv(T_world_ref)
    devices = {dev: T_ref_world @ T_wc for dev, T_wc in world_by_dev.items()}

    # legacy v2 sets: {dev: [{corners, c_ids, resolution}, ...]} — one physical
    # capture-all is column k across every device.
    legacy_sets = legacy.get("sets", {}) if isinstance(legacy, dict) else {}
    n_sets = max((len(v) for v in legacy_sets.values()), default=0)
    sets: list[dict] = []
    for k in range(n_sets):
        obs: dict[str, dict] = {}
        for dev, rows in legacy_sets.items():
            if k < len(rows):
                r = rows[k]
                obs[dev] = {
                    "charuco_ids": np.asarray(r.get("c_ids", []), int).reshape(-1).tolist(),
                    "charuco_corners": np.asarray(
                        r.get("corners", []), float
                    ).reshape(-1, 2).tolist(),
                }
        if obs:
            sets.append({"set_id": f"legacy-{k:03d}", "captured_at": None, "observations": obs})

    payload = write_extrinsics(
        name,
        reference_device=reference_device,
        devices=devices,
        sets=sets,
        solve={
            "method": "legacy-migrated",
            "rms_reproj_px": {},
            "n_sets": len(sets),
            "n_points": 0,
        },
        intrinsics=legacy.get("intrinsics", {}) if isinstance(legacy, dict) else {},
        board=legacy.get("board") if isinstance(legacy, dict) else None,
    )
    # the legacy world frame == the reference camera's board pose; preserve it as
    # the display anchor so migrated episodes render unchanged.
    ref_obs = sets[0]["observations"].get(reference_device, {}) if sets else {}
    write_world_anchor(
        name,
        T_world_display=T_world_ref,
        observations={reference_device: ref_obs} if ref_obs else {},
    )
    return payload


def ensure_migrated(name: str) -> None:
    """Idempotently migrate a legacy preset if the new ``extrinsics.json`` is
    not there yet."""
    if not _extrinsics_path(name).is_file():
        migrate_legacy(name)
