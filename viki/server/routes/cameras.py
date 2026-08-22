"""
viki.server.routes.cameras
--------------------------
Camera device endpoints: discovery, start/stop, info, and colour/depth
MJPEG streams. Handlers stay thin — they delegate to the CameraManager
and to ``viki.server.streams``.
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from viki import config
from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.streams import camera_stream

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
        mgr.start(
            device_id,
            fps=req.fps,
            color_width=req.color_width,
            color_height=req.color_height,
            depth_mode=req.depth_mode,
            wired_sync_mode=req.wired_sync_mode,
            subordinate_delay_us=req.subordinate_delay_us,
            synchronized_images_only=req.synchronized_images_only,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "started", "device_id": device_id}


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
