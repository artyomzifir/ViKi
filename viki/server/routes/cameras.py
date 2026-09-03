"""
viki.server.routes.cameras
--------------------------
Camera device endpoints: discovery, start/stop, info, and colour/depth
MJPEG streams. Handlers stay thin — they delegate to the CameraManager
and to ``viki.server.streams``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from viki import config
from viki.calibration.manager import CalibrationManager
from viki.cameras.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.streams import camera_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])

# 0 = standalone, 1 = master, 2 = subordinate (viki.cameras.kinect enum values)
_WIRED_MASTER, _WIRED_SUBORDINATE = 1, 2


def _wired_sync_for(device_id: str) -> tuple[int, int]:
    """Resolve (wired_sync_mode, subordinate_delay_us) for ``device_id`` from
    ``config.KINECT_SYNC``. Returns (0, 0) when the device has no assigned role,
    so an un-cabled rig just starts standalone."""
    spec = getattr(config, "KINECT_SYNC", {}) or {}
    if device_id == spec.get("master"):
        return _WIRED_MASTER, 0
    if device_id in (spec.get("subordinates") or []):
        return _WIRED_SUBORDINATE, int(spec.get("subordinate_delay_us", 0))
    return 0, 0

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


class StartRequest(BaseModel):
    fps: int = config.DEFAULT_FPS
    color_width: int = config.DEFAULT_COLOR_WIDTH
    color_height: int = config.DEFAULT_COLOR_HEIGHT
    depth_mode: str = config.DEFAULT_DEPTH_MODE
    # Kinect-only: hardware sync wiring (ignored for RealSense)
    # 0 = standalone, 1 = master, 2 = subordinate
    wired_sync_mode: int = 0
    subordinate_delay_us: int = 0
    # Require color and depth to arrive in the same capture (recommended for sync recording).
    synchronized_images_only: bool = config.DEFAULT_SYNCHRONIZED_IMAGES_ONLY


@router.get("/devices")
async def list_devices(mgr: CameraManager = Depends(get_manager)):
    """
    List all detected camera devices (RealSense and Kinect).

    Returns
    -------
    dict
        Keys: "realsense", "kinect", "active" (currently running), and error keys if any.
    """
    return mgr.list_devices()


@router.post("/{device_id}/start")
async def start_camera(
    device_id: str,
    req: StartRequest,
    mgr: CameraManager = Depends(get_manager),
):
    """
    Start streaming from a camera.

    Parameters
    ----------
    device_id : str
        Camera identifier (serial for RealSense, "kinect_N" for Kinect).
    req : StartRequest
        FPS, resolution, depth mode, and Kinect sync options.

    Returns
    -------
    dict
        {"status": "started", "device_id": device_id}

    Raises
    ------
    HTTPException 500
        If the backend fails to start.
    """
    # Honour an explicit request; otherwise fall back to the rig's KINECT_SYNC
    # wiring so "Start cameras" produces a hardware-synced Kinect pair whenever
    # the sync cable + config are present (paper §3.3), and a plain standalone
    # start otherwise. Start subordinates before the master (they must be
    # listening when the master starts sending trigger pulses).
    wired_sync_mode = req.wired_sync_mode
    subordinate_delay_us = req.subordinate_delay_us
    if wired_sync_mode == 0:
        wired_sync_mode, subordinate_delay_us = _wired_sync_for(device_id)
        if wired_sync_mode:
            role = "master" if wired_sync_mode == _WIRED_MASTER else "subordinate"
            logger.info("cameras: %s starting as hardware-sync %s", device_id, role)

    # Ordering: a subordinate must be running (listening for trigger pulses)
    # before the master starts firing them. If this is the master, bring any
    # configured-but-inactive subordinate up first with matching capture params.
    if wired_sync_mode == _WIRED_MASTER and req.wired_sync_mode == 0:
        spec = getattr(config, "KINECT_SYNC", {}) or {}
        for sub in spec.get("subordinates") or []:
            if sub in mgr.active_device_ids():
                continue
            _sm, _sd = _wired_sync_for(sub)
            try:
                mgr.start(
                    sub, fps=req.fps, color_width=req.color_width,
                    color_height=req.color_height, depth_mode=req.depth_mode,
                    wired_sync_mode=_sm, subordinate_delay_us=_sd,
                    synchronized_images_only=req.synchronized_images_only,
                )
                logger.info("cameras: auto-started subordinate %s before master %s", sub, device_id)
            except Exception:  # noqa: BLE001 — master start will still be attempted
                logger.exception("cameras: subordinate %s auto-start failed", sub)

    try:
        outcome = mgr.start(
            device_id,
            fps=req.fps,
            color_width=req.color_width,
            color_height=req.color_height,
            depth_mode=req.depth_mode,
            wired_sync_mode=wired_sync_mode,
            subordinate_delay_us=subordinate_delay_us,
            synchronized_images_only=req.synchronized_images_only,
        )
    except Exception as e:
        logger.exception("camera %s start failed", device_id)
        raise HTTPException(status_code=500, detail=str(e))
    # outcome: "started" | "restarted" (config changed) | "unchanged"
    logger.info("camera %s %s @ %dx%d/%dfps %s%s", device_id, outcome or "started",
                req.color_width, req.color_height, req.fps, req.depth_mode,
                f" sync={wired_sync_mode}" if wired_sync_mode else "")
    return {"status": outcome or "started", "device_id": device_id}


@router.post("/{device_id}/stop")
async def stop_camera(device_id: str, mgr: CameraManager = Depends(get_manager)):
    """
    Stop streaming from a camera.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        {"status": "stopped", "device_id": device_id}
    """
    mgr.stop(device_id)
    logger.info("camera %s stopped", device_id)
    return {"status": "stopped", "device_id": device_id}


@router.get("/{device_id}/info")
async def camera_info(device_id: str, mgr: CameraManager = Depends(get_manager)):
    """
    Get camera info (resolution, intrinsics, running status, latest frame timestamp).

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        Camera info or 404 if not found/started.
    """
    info = mgr.get_info(device_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Camera not found or not started")
    return info


@router.get("/{device_id}/stream")
def colour_stream(
    device_id: str,
    undistort: bool = True,
    mgr: CameraManager = Depends(get_manager),
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    MJPEG stream of the colour image.

    Parameters
    ----------
    device_id : str
        Camera ID.
    undistort : bool, default=True
        Apply undistortion using loaded intrinsics.

    Returns
    -------
    StreamingResponse
        Multipart MJPEG stream.
    """
    return StreamingResponse(
        camera_stream(mgr, cal, device_id, "color", undistort=undistort),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )


@router.get("/{device_id}/depth")
def depth_stream(
    device_id: str,
    mgr: CameraManager = Depends(get_manager),
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    MJPEG stream of the colour-mapped depth image.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    StreamingResponse
        Multipart MJPEG stream.
    """
    return StreamingResponse(
        camera_stream(mgr, cal, device_id, "depth"),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )
