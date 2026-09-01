"""
viki.server.routes.skeleton
---------------------------
Only the static-background depth capture survives here — it feeds offline
``perception.lift`` (scene subtraction). Live estimation / recording / the 3-D
WebSocket stream are gone: skeletons are extracted offline from recordings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from viki import config
from viki.cameras.manager import CameraManager
from viki.perception.camera_prep import prepare_frame
from viki.server.deps import get_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/skeleton", tags=["skeleton"])


@router.post("/capture_base/{device_id}")
async def capture_base_depth(
    device_id: str, mgr: CameraManager = Depends(get_manager)
):
    """Snapshot the current depth frame as the static background for a camera."""
    base_dir = Path(getattr(config, "SKELETON_DEPTH_BASE_DIR", "data/depth_bases/"))
    base_dir.mkdir(parents=True, exist_ok=True)

    frame = mgr.latest_frame(device_id)
    if frame is None or not frame.has_depth():
        raise HTTPException(400, f"no depth frame available for {device_id}")

    prepared = prepare_frame(frame)
    path = base_dir / f"{device_id}.npy"
    np.save(path, prepared.depth_m)
    logger.info("skeleton depth base captured for %s -> %s", device_id, path)
    return {"status": "success", "device_id": device_id, "path": str(path)}
