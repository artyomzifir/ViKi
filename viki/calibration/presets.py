"""
viki.calibration.presets
------------------------
Named extrinsics sets. Each preset is one file under ``data/calibrations/`` in
the same list-of-``{device_id, rvec, tvec}`` format as ``EXTRINSICS_FILENAME``
(see :mod:`viki.calibration.file`). One preset is *active*: its name is stored
under ``ACTIVE_CALIBRATION`` in the user config, and activating a preset copies
it onto ``EXTRINSICS_FILENAME`` so the rest of the code path is unchanged.
"""

from __future__ import annotations

import json
import shutil
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


def current_active() -> str:
    """The active preset name, read fresh from the user config (may be empty)."""
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


def list_presets() -> list[dict]:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    active = current_active()
    out: list[dict] = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            cams = [e.get("device_id") for e in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            cams = []
        out.append(
            {
                "name": f.stem,
                "solved_at": f.stat().st_mtime,
                "cameras": cams,
                "active": f.stem == active,
            }
        )
    return out


def save_as(name: str, src: str = EXTRINSICS_FILENAME) -> Path:
    """Copy the current solved extrinsics into a named preset."""
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError("no current extrinsics to save; run the solve first")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    dst = preset_path(name)
    shutil.copyfile(src_p, dst)
    return dst


def activate(name: str, dst: str = EXTRINSICS_FILENAME) -> Path:
    """Make ``name`` the active preset: copy it onto ``dst`` and record it."""
    src = preset_path(name)
    if not src.exists():
        raise FileNotFoundError(f"no calibration preset {name!r}")
    shutil.copyfile(src, Path(dst))
    _set_active(_safe_name(name))
    return src


def delete(name: str) -> None:
    p = preset_path(name)
    if p.exists():
        p.unlink()
    if current_active() == _safe_name(name):
        _set_active("")


def apply_active_on_startup(dst: str = EXTRINSICS_FILENAME) -> str | None:
    """If an active preset is set and its file exists, copy it onto ``dst``.

    Called from the app lifespan before ``load_all_extrinsics()``. Returns the
    preset name it applied, or ``None``.
    """
    name = current_active()
    if not name:
        return None
    src = PRESETS_DIR / f"{name}.json"
    if not src.exists():
        return None
    shutil.copyfile(src, Path(dst))
    return name
