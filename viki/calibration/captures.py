"""
viki.calibration.captures
-------------------------
On-disk calibration capture photos, next to the calibration files:

    data/calibrations/<owner>/set-000/<device>.jpg

``<owner>`` is ``_live`` for the current (unsaved) session; saving a preset
copies ``_live/`` to ``data/calibrations/<preset-name>/``. Each image is the
captured colour frame with the detected board drawn on it, so a bad capture is
obvious at a glance.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

ROOT = Path("data/calibrations")
LIVE = "_live"


def _owner_dir(owner: str) -> Path:
    return ROOT / owner


def set_dir(owner: str, index: int) -> Path:
    return _owner_dir(owner) / f"set-{index:03d}"


def image_path(owner: str, index: int, device: str) -> Path:
    return set_dir(owner, index) / f"{device}.jpg"


def save_set(owner: str, index: int, images: dict) -> dict[str, str]:
    d = set_dir(owner, index)
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for dev, img in images.items():
        p = d / f"{dev}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out[dev] = str(p)
    return out


def list_sets(owner: str) -> list[dict]:
    base = _owner_dir(owner)
    if not base.is_dir():
        return []
    rows = []
    for sd in sorted(base.glob("set-*")):
        try:
            i = int(sd.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        rows.append({"index": i, "devices": sorted(p.stem for p in sd.glob("*.jpg"))})
    return rows


def wipe(owner: str) -> None:
    d = _owner_dir(owner)
    if d.is_dir():
        shutil.rmtree(d)


def delete_set(owner: str, index: int) -> None:
    """Remove set ``index`` and shift every higher set down by one."""
    base = _owner_dir(owner)
    if not base.is_dir():
        return
    tgt = set_dir(owner, index)
    if tgt.is_dir():
        shutil.rmtree(tgt)
    for sd in sorted(base.glob("set-*")):
        try:
            i = int(sd.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if i > index:
            sd.rename(base / f"set-{i - 1:03d}")


def copy(src_owner: str, dst_owner: str) -> None:
    src, dst = _owner_dir(src_owner), _owner_dir(dst_owner)
    if dst.is_dir():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
