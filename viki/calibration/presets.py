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
        try:
            data = json.loads(f.read_text())
            extr = _extrinsics_of(data)
            cams = [e.get("device_id") for e in extr]
            n_sets = (
                max((len(v) for v in data.get("sets", {}).values()), default=0)
                if isinstance(data, dict) else 0
            )
        except (json.JSONDecodeError, OSError):
            cams, n_sets = [], 0
        out.append({
            "name": f.stem,
            "solved_at": f.stat().st_mtime,
            "cameras": cams,
            "sets": n_sets,
            "active": f.stem == active,
        })
    return out


def read_detail(name: str) -> dict:
    """Full preset content for the reopen view."""
    data = _read(name)
    if isinstance(data, list):
        return {"name": _safe_name(name), "version": 1, "extrinsics": data,
                "sets": {}, "intrinsics": {}, "board": None}
    return {
        "name": _safe_name(name),
        "version": data.get("version", 2),
        "extrinsics": data.get("extrinsics", []),
        "sets": data.get("sets", {}),
        "intrinsics": data.get("intrinsics", {}),
        "board": data.get("board"),
    }


def save_as(
    name: str,
    *,
    extrinsics: list[dict],
    sets: dict | None = None,
    intrinsics: dict | None = None,
    board: dict | None = None,
) -> Path:
    if not extrinsics:
        raise ValueError("no solved extrinsics to save")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    dst = preset_path(name)
    dst.write_text(json.dumps({
        "version": 2,
        "extrinsics": extrinsics,
        "sets": sets or {},
        "intrinsics": intrinsics or {},
        "board": board,
    }, indent=2))
    return dst


def activate(name: str, dst: str | None = None) -> Path:
    data = _read(name)
    _write_active_extrinsics(_extrinsics_of(data), dst)
    _set_active(_safe_name(name))
    return preset_path(name)


def delete(name: str) -> None:
    p = preset_path(name)
    if p.exists():
        p.unlink()
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
