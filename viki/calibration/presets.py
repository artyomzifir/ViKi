"""
viki.calibration.presets
------------------------
Named calibration presets under ``data/calibrations/``. One is *active*: its name
lives in ``ACTIVE_CALIBRATION`` in the user config, and activating a preset
writes its extrinsics onto ``EXTRINSICS_FILENAME`` so the rest of the code path
is unchanged.

File formats (both read):
  * **v1** — a plain list ``[{device_id, rvec, tvec}, ...]`` (legacy).
  * **v2** — ``{"version": 2, "extrinsics": [...v1 list...], "sets": {dev: [...]},
    "intrinsics": {dev: {...}}, "board": {...}}``. The extra fields let a preset
    be reopened: drop a capture set and re-solve extrinsics offline.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from viki.config import EXTRINSICS_FILENAME, USER_CONFIG_PATH

PRESETS_DIR = Path("data/calibrations")


def _safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_. ").strip()
    if not cleaned:
        raise ValueError(f"invalid preset name: {name!r}")
    return cleaned


def preset_path(name: str) -> Path:
    return PRESETS_DIR / f"{_safe_name(name)}.json"


def _read(name: str) -> dict | list:
    p = preset_path(name)
    if not p.exists():
        raise FileNotFoundError(f"no calibration preset {name!r}")
    return json.loads(p.read_text())


def _extrinsics_of(data: dict | list) -> list[dict]:
    """The flat ``[{device_id, rvec, tvec}]`` list from a v1 or v2 preset."""
    if isinstance(data, list):
        return data
    return data.get("extrinsics", [])


# ── active-preset pointer ─────────────────────────────────────────────────


def current_active() -> str:
    p = Path(USER_CONFIG_PATH)
    if not p.exists():
        return ""
    try:
        return json.loads(p.read_text()).get("ACTIVE_CALIBRATION", "") or ""
    except (json.JSONDecodeError, OSError):
        return ""


def _set_active(name: str) -> None:
    p = Path(USER_CONFIG_PATH)
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg["ACTIVE_CALIBRATION"] = name
    p.write_text(json.dumps(cfg, indent=2))


def _write_active_extrinsics(extr: list[dict], dst: str | None = None) -> None:
    Path(dst or EXTRINSICS_FILENAME).write_text(json.dumps(extr, indent=2))


# ── CRUD ─────────────────────────────────────────────────────────────────


def list_presets() -> list[dict]:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    active = current_active()
    out: list[dict] = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        k4a: list[str] = []
        try:
            data = json.loads(f.read_text())
            extr = _extrinsics_of(data)
            cams = [e.get("device_id") for e in extr]
            n_sets = (
                max((len(v) for v in data.get("sets", {}).values()), default=0)
                if isinstance(data, dict) else 0
            )
            if isinstance(data, dict):
                k4a = sorted((data.get("k4a_raw") or {}).keys())
        except (json.JSONDecodeError, OSError):
            cams, n_sets = [], 0
        out.append({
            "name": f.stem,
            "solved_at": f.stat().st_mtime,
            "cameras": cams,
            "sets": n_sets,
            "k4a": k4a,
            "active": f.stem == active,
        })
    return out


def _set_images(name: str) -> dict:
    """{set_index: {device: image_url}} for the preview thumbnails."""
    from viki.calibration import captures

    safe = _safe_name(name)
    return {
        r["index"]: {
            d: f"/api/calibration/presets/{safe}/sets/{r['index']}/{d}.jpg"
            for d in r["devices"]
        }
        for r in captures.list_sets(safe)
    }


def read_detail(name: str) -> dict:
    """Full preset content for the reopen view."""
    data = _read(name)
    if isinstance(data, list):
        return {"name": _safe_name(name), "version": 1, "extrinsics": data,
                "sets": {}, "intrinsics": {}, "board": None, "set_images": {}}
    return {
        "name": _safe_name(name),
        "version": data.get("version", 2),
        "extrinsics": data.get("extrinsics", []),
        "sets": data.get("sets", {}),
        "intrinsics": data.get("intrinsics", {}),
        "board": data.get("board"),
        "set_images": _set_images(name),
        "k4a_devices": sorted((data.get("k4a_raw") or {}).keys()),
    }


# ── k4a raw calibration (device colour↔depth model, for offline lifting) ───


def attach_k4a(
    name: str, blobs: dict[str, bytes], depth_mode_int: int | None, color_res_int: int | None
) -> dict:
    """Store each Kinect's raw calibration blob (base64) + the depth-mode /
    colour-resolution enum ints on an existing v2 preset, without re-solving.
    The blob is a device property, so it stays valid for any recording made
    against this preset at the same depth mode / colour resolution."""
    data = _read(name)
    if isinstance(data, list):
        raise ValueError("preset is v1 (legacy list) — re-solve to upgrade before attaching k4a")
    if not blobs:
        raise ValueError("no raw calibration blobs to attach")
    data.setdefault("k4a_raw", {})
    for dev, blob in blobs.items():
        data["k4a_raw"][dev] = base64.b64encode(blob).decode("ascii")
    if depth_mode_int is not None:
        data["k4a_depth_mode_int"] = int(depth_mode_int)
    if color_res_int is not None:
        data["k4a_color_res_int"] = int(color_res_int)
    preset_path(name).write_text(json.dumps(data, indent=2))
    return read_detail(name)


def k4a_calibration(name: str, dev_id: str):
    """Rebuilt :class:`~viki.perception.k4a_offline.K4ACalibration` for ``dev_id``
    from this preset's stored blob, or ``None``."""
    try:
        data = _read(name)
    except FileNotFoundError:
        return None
    if isinstance(data, list):
        return None
    b64 = (data.get("k4a_raw") or {}).get(dev_id)
    if not b64:
        return None
    from viki.perception.k4a_offline import K4ACalibration

    return K4ACalibration.from_blob(
        base64.b64decode(b64),
        data.get("k4a_depth_mode_int"),
        data.get("k4a_color_res_int"),
        tag=f"{name}/{dev_id}",
    )


def save_as(
    name: str,
    *,
    extrinsics: list[dict],
    sets: dict | None = None,
    intrinsics: dict | None = None,
    board: dict | None = None,
) -> Path:
    from viki.calibration import captures

    if not extrinsics:
        raise ValueError("no solved extrinsics to save")
    if _safe_name(name) == captures.LIVE:
        raise ValueError(f"{captures.LIVE!r} is reserved")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    dst = preset_path(name)
    dst.write_text(json.dumps({
        "version": 2,
        "extrinsics": extrinsics,
        "sets": sets or {},
        "intrinsics": intrinsics or {},
        "board": board,
    }, indent=2))
    captures.copy(captures.LIVE, _safe_name(name))  # freeze the session photos
    return dst


def activate(name: str, dst: str | None = None) -> Path:
    data = _read(name)
    _write_active_extrinsics(_extrinsics_of(data), dst)
    _set_active(_safe_name(name))
    return preset_path(name)


def delete(name: str) -> None:
    from viki.calibration import captures

    p = preset_path(name)
    if p.exists():
        p.unlink()
    captures.wipe(_safe_name(name))
    if current_active() == _safe_name(name):
        _set_active("")


def delete_set(name: str, index: int) -> dict:
    """Drop capture set ``index`` from a preset and re-solve its extrinsics.

    Returns the updated :func:`read_detail`. If the preset is active, the new
    extrinsics are also written to ``EXTRINSICS_FILENAME``.
    """
    from viki.calibration.samples import solve_extrinsics

    data = _read(name)
    if isinstance(data, list) or not data.get("sets"):
        raise ValueError("preset has no stored capture sets to edit")

    sets = data["sets"]
    for dev in list(sets):
        if 0 <= index < len(sets[dev]):
            sets[dev].pop(index)

    board = data.get("board") or {}
    intr = data.get("intrinsics") or {}
    data["extrinsics"] = solve_extrinsics(sets, intr, board)
    preset_path(name).write_text(json.dumps(data, indent=2))

    from viki.calibration import captures
    captures.delete_set(_safe_name(name), index)

    if current_active() == _safe_name(name):
        _write_active_extrinsics(data["extrinsics"])
    return read_detail(name)


def apply_active_on_startup(dst: str | None = None) -> str | None:
    """Called from the app lifespan before ``load_all_extrinsics()``."""
    name = current_active()
    if not name or not preset_path(name).exists():
        return None
    _write_active_extrinsics(_extrinsics_of(_read(name)), dst)
    return name
