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
from viki.cameras.hw_sync import HardwareSyncError, WIRED_STANDALONE
from viki.cameras.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.streams import camera_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])

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
    try:
        if device_id.startswith("kinect_"):
            plan = mgr.kinect_sync_plan()
        else:
            plan = {}

        if len(plan) >= 2:
            role = plan.get(device_id)
            if role is None:
                raise HardwareSyncError(
                    f"{device_id} is not assigned in the Kinect HW_SYNC rig"
                )
            if req.wired_sync_mode not in (WIRED_STANDALONE, role.mode):
                raise HardwareSyncError(
                    f"{device_id} must run as {role.name}, not mode {req.wired_sync_mode}"
                )
            if req.subordinate_delay_us not in (0, role.delay_us):
                raise HardwareSyncError(
                    f"{device_id} must use subordinate_delay_us={role.delay_us}"
                )
            if not req.synchronized_images_only:
                raise HardwareSyncError(
                    "multi-Kinect HW_SYNC requires synchronized_images_only=true"
                )
            # Starting either Kinect means starting the entire rig as one
            # transaction: subordinate(s) first, master last, rollback on error.
            outcomes = mgr.start_configured_kinect_rig(
                fps=req.fps,
                color_width=req.color_width,
                color_height=req.color_height,
                depth_mode=req.depth_mode,
            )
            outcome = outcomes[device_id]
        else:
            outcome = mgr.start(
                device_id,
                fps=req.fps,
                color_width=req.color_width,
                color_height=req.color_height,
                depth_mode=req.depth_mode,
                wired_sync_mode=req.wired_sync_mode,
                subordinate_delay_us=req.subordinate_delay_us,
                synchronized_images_only=req.synchronized_images_only,
            )
    except HardwareSyncError as exc:
        logger.error("camera %s HW_SYNC start refused: %s", device_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        logger.exception("camera %s start failed", device_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
    # outcome: "started" | "restarted" (config changed) | "unchanged"
    logger.info("camera %s %s @ %dx%d/%dfps %s%s", device_id, outcome or "started",
                req.color_width, req.color_height, req.fps, req.depth_mode,
                " HW_SYNC" if len(plan) >= 2 else "")
    return {
        "status": outcome or "started",
        "device_id": device_id,
        "active": mgr.active_device_ids(),
        "hardware_sync": mgr.hardware_sync_status(),
    }


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
    stopped = [device_id]
    if device_id.startswith("kinect_"):
        try:
            plan = mgr.kinect_sync_plan()
        except HardwareSyncError:
            # The config may have been edited after a valid rig was started.
            # Still stop every active Kinect rather than leave a half-rig.
            active_kinects = [
                active for active in mgr.active_device_ids()
                if active.startswith("kinect_")
            ]
            plan = {active: None for active in active_kinects}
        if len(plan) >= 2:
            # Do not leave a master/subordinate half-rig running after a card's
            # Stop button is pressed.
            stopped = list(plan)
            for kinect_id in stopped:
                mgr.stop(kinect_id)
        else:
            mgr.stop(device_id)
    else:
        mgr.stop(device_id)
    logger.info("camera rig stopped via %s: %s", device_id, stopped)
    return {"status": "stopped", "device_id": device_id, "stopped": stopped}


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
