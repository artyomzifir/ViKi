"""
viki.server.routes.skeleton
--------------------------
Endpoints for controlling skeleton estimation and recording,
and a WebSocket for streaming the latest skeleton frame.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
from viki.skeleton.models import LM

import viki.config as config
from viki.capture.manager import CameraManager
from viki.skeleton.camera_prep import prepare_frame
from viki.server.deps import get_worker, get_manager
from viki.server.skeleton_worker import SkeletonWorker


def sanitize_nan(val):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(val, dict):
        return {k: sanitize_nan(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_nan(x) for x in val]
    if isinstance(val, np.ndarray):
        return sanitize_nan(val.tolist())
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


router = APIRouter(prefix="/skeleton", tags=["skeleton"])
logger = logging.getLogger(__name__)


class ToggleRequest(BaseModel):
    enabled: bool


@router.post("/toggle")
async def toggle_estimation(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable skeleton estimation.

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "enabled": bool}
    """
    worker.set_enabled(req.enabled)
    return {"status": "updated", "enabled": worker.is_enabled}


@router.post("/record")
async def toggle_recording(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable recording of skeleton data to disk.

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "recording": bool}
    """
    worker.set_recording(req.enabled)
    return {"status": "updated", "recording": worker.is_recording}


@router.post("/depth-debug")
async def toggle_depth_debug(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable depth-projection debug marks (red dots on the 3D panel).

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "depth_debug": bool}
    """
    worker.set_depth_debug(req.enabled)
    return {"status": "updated", "depth_debug": req.enabled}


@router.post("/capture_base/{device_id}")
async def capture_base_depth(
    device_id: str,
    mgr: CameraManager = Depends(get_manager),
):
    """
    Capture the current background depth for a camera and persist it as the
    static "base" depth map.

    During skeleton estimation, ``lift_to_3d`` subtracts this base from the live
    depth so the tracked hand stands out from the (static) background, which
    stabilises depth at the hand even when the IR pattern is sparse.

    Parameters
    ----------
    device_id : str
        Camera to capture the base depth for.

    Returns
    -------
    dict
        {"status": "success", "device_id": str, "path": str}
    """
    base_dir = Path(config.SKELETON_DEPTH_BASE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)

    frame = mgr.latest_frame(device_id)
    if frame is None or not frame.has_depth():
        raise HTTPException(
            status_code=400, detail=f"No depth frame available for {device_id}"
        )

    # Save in the same depth space the live pipeline uses: mm -> m, 0 -> NaN.
    prepared = prepare_frame(frame)
    path = base_dir / f"{device_id}.npy"
    np.save(path, prepared.depth_m)
    logger.info("Saved base depth for %s -> %s", device_id, path)
    return {"status": "success", "device_id": device_id, "path": str(path)}


@router.get("/status")
async def get_status(worker: SkeletonWorker = Depends(get_worker)):
    """
    Get current skeleton estimation and recording status.

    Returns
    -------
    dict
        {"enabled": bool, "recording": bool}
    """
    return {
        "enabled": worker.is_enabled,
        "recording": worker.is_recording,
    }


@router.websocket("/stream")
async def skeleton_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams the latest skeleton result.

    Sends JSON with one entry per camera in ``frames`` (each tagged with its
    ``device_id`` so the frontend can draw it in its own colour), the per‑camera
    2D ``detections``, and the (un‑fused) ``debug_depth_marks``. Updates at
    approximately 20 Hz.
    """
    await websocket.accept()
    worker: SkeletonWorker = websocket.app.state.skeleton_worker
    try:
        while True:
            result = worker.get_latest_result()
            detections = worker.get_latest_detections()

            if result or detections:
                debug_marks = (
                    result.debug_depth_marks if result is not None else None
                )
                ts = (
                    result.frames[0].timestamp_us
                    if result and result.frames
                    else time.time_ns() // 1000
                )
                data = {
                    "ts": ts,
                    "frames": [
                        {
                            "device_id": f.device_id,
                            "landmarks": sanitize_nan(
                                {lm.value: vec for lm, vec in f.points.items()}
                            ),
                            "end_effector": (
                                sanitize_nan(f.end_effector.as_dict())
                                if f.end_effector else None
                            ),
                        }
                        for f in (result.frames if result else [])
                    ],
                    "detections": {
                        dev_id: (sanitize_nan(det.points) if det else {})
                        for dev_id, det in detections.items()
                    },
                    "debug_depth_marks": (
                        {
                            dev_id: {
                                lm.value: sanitize_nan(vec.tolist())
                                for lm, vec in marks.items()
                            }
                            for dev_id, marks in debug_marks.items()
                        }
                        if debug_marks else {}
                    ),
                }
                await websocket.send_json(data)

            # Stream at ~20 fps (comment out for unbound stream)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
