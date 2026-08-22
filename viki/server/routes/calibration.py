"""
viki.server.routes.calibration
------------------------------
Calibration endpoints: live mosaic preview, sample capture, running the
calibration solve, status, and clearing collected samples.
"""

from __future__ import annotations

import cv2
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from viki.calibration.models import ArucoBoardParameters

logger = logging.getLogger(__name__)

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.streams import marked_camera_stream
from viki.server.routes.models import (
    ArucoBoardParametersData,
    BoardParametersData,
    IntrinsicsResponse,
    ExtrinsicsResponse,
)
from viki.config import (
    INTRINSICS_FILENAME,
    EXTRINSICS_FILENAME,
)

router = APIRouter(prefix="/calibration", tags=["calibration"])

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


@router.post("/reset")
async def reset(cal: CalibrationManager = Depends(get_calibrator)):
    """
    Stop all calibration workers.

    Returns
    -------
    dict
        {"status": "success"}
    """
    cal.stop_all()
    return {"status": "success"}


@router.post("/sync")
async def sync(
    params: ArucoBoardParametersData | BoardParametersData,
    board_type: str,
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    Synchronise board parameters across all calibration workers.

    This updates the board size, square size, and (for ChArUco) marker size
    and dictionary for every active worker. Call this before starting calibration
    if the physical board changes.

    Parameters
    ----------
    params : ArucoBoardParametersData or BoardParametersData
        Board parameters (fields depend on board_type).
    board_type : str
        "chess" or "aruco".

    Returns
    -------
    dict
        {"status": "success"}

    Raises
    ------
    HTTPException 422
        If `aruco_dict` is invalid.
    """
    # Extract common params
    board_size = params.board_size
    square_size = params.square_size

    # Extract aruco specific
    marker_size = 0.025
    aruco_dict = cv2.aruco.DICT_6X6_250

    if board_type == "aruco" and isinstance(params, ArucoBoardParameters):
        marker_size = params.marker_size
        try:
            aruco_dict = getattr(cv2.aruco, str(params.aruco_dict))
        except Exception:
            raise HTTPException(422, f"wrong aruco_dict: {params.aruco_dict}")

    cal.sync_params(board_type, board_size, square_size, marker_size, aruco_dict)
    return {"status": "success"}


@router.post("/capture/{device_id}")
async def capture(
    device_id: str,
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    Manually capture a calibration sample for a specific camera.

    The sample is added to the worker's internal collection if the board is detected.

    Parameters
    ----------
    device_id : str
        Camera device ID.

    Returns
    -------
    dict
        Always returns {"status": "success"} – check worker status for actual sample count.
    """
    cal.capture(device_id)


@router.post("/capture")
async def capture_all(
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    Manually capture a calibration sample from all active workers.

    Returns
    -------
    dict
        {"status": "success"}

    Raises
    ------
    HTTPException 400
        If no calibration workers are active (i.e., sync not called).
    """
    if not cal._workers:
        raise HTTPException(
            400,
            "Calibration session not started. Please click 'Sync Parameters' first.",
        )
    cal.capture_all()


@router.post("/start/{device_id}")
async def start_worker(
    device_id: str,
    mode: str = "auto",
    params: BoardParametersData | None = None,
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    Start a chessboard calibration worker for the given device.

    Parameters
    ----------
    device_id : str
        Camera ID.
    mode : str, default="auto"
        "auto" for background capture thread, "manual" for explicit captures.
    params : BoardParametersData, optional
        Board size and square size; defaults to (8,6) and 0.025 m.

    Returns
    -------
    dict
        {"status": "success"}
    """
    if not params:
        board_size = (8, 6)
        square_size = 0.025
    else:
        board_size = params.board_size
        square_size = params.square_size
    cal.start(device_id, mode, "chess", board_size, square_size)


@router.post("/start/aruco/{device_id}")
async def start_aruco_worker(
    device_id: str,
    mode: str = "auto",
    params: ArucoBoardParametersData | None = None,
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    Start a ChArUco board calibration worker for the given device.

    Parameters
    ----------
    device_id : str
        Camera ID.
    mode : str, default="auto"
        "auto" or "manual".
    params : ArucoBoardParametersData, optional
        Board parameters; defaults to (10,8) board, square 0.05 m, marker 0.035 m,
        and DICT_5X5_100.

    Returns
    -------
    dict
        {"status": "success"}

    Raises
    ------
    HTTPException 422
        If `aruco_dict` string is invalid.
    """
    if not params:
        board_size = (10, 8)
        square_size = 0.05
        marker_size = 0.035
        aruco_dict = cv2.aruco.DICT_5X5_100
    else:
        board_size = params.board_size
        square_size = params.square_size
        marker_size = params.marker_size
        try:
            aruco_dict = getattr(cv2.aruco, params.aruco_dict)
        except Exception:
            raise HTTPException(422, f"wrong aruco_dict: {params.aruco_dict}")
    cal.start(
        device_id, mode, "aruco", board_size, square_size, marker_size, aruco_dict
    )


@router.get("/status/{device_id}")
async def status(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    """
    Get calibration status for a device.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        {"samples_count": int, "started": bool}
    """
    # logger.debug(f"calibration status for {device_id}: {cal.status(device_id)}")
    return cal.status(device_id)


@router.get("/samples_count/{device_id}")
async def samples_count(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    """
    Get the number of collected samples for a device.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        {"samples_count": int}
    """
    return {"samples_count": cal.samples_count(device_id)}


@router.get("/is_device_active/{device_id}")
async def is_device_active(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    """
    Check if a calibration worker is active for the device.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        {"is_device_active": bool}
    """
    return {"is_device_active": cal.is_device_active(device_id)}


@router.post("/clear/{device_id}")
async def clear(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    """
    Clear all collected calibration samples for a device.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    dict
        {"status": "cleared"}
    """
    cal.clear(device_id)
    return {"status": "cleared"}


@router.post("/intrinsics/{device_id}", response_model=IntrinsicsResponse)
async def intrinsics_post(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    """
    Compute and save intrinsic parameters for a device (POST).

    Uses the collected samples to run the calibration solve and persists the
    result to the default intrinsics file.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    IntrinsicsResponse
        Focal lengths, principal point, distortion coefficients.

    Raises
    ------
    RuntimeError
        If calibration fails (propagated as HTTP 500).
    """
    intrinsics = cal.intrinsics_calibration(device_id, INTRINSICS_FILENAME)
    return IntrinsicsResponse(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        dist_coeffs=intrinsics.dist_coeffs.tolist(),
    )


@router.get("/intrinsics/{device_id}", response_model=IntrinsicsResponse)
async def intrinsics(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    """
    Retrieve previously computed intrinsic parameters (GET).

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    IntrinsicsResponse
        Focal lengths, principal point, distortion coefficients.

    Raises
    ------
    HTTPException 404
        If intrinsics not found.
    """
    intrinsics = cal.get_intrinsics(device_id)
    if not intrinsics:
        raise HTTPException(
            status_code=404, detail="Intrinsics not found for this device"
        )
    return IntrinsicsResponse(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        dist_coeffs=intrinsics.dist_coeffs.tolist(),
    )


@router.post("/extrinsics", response_model=list[ExtrinsicsResponse])
async def extrinsics_post_all(
    cal: CalibrationManager = Depends(get_calibrator),
    mgr: CameraManager = Depends(get_manager),
):
    """
    Compute and save extrinsic parameters for all active devices.

    Runs extrinsics calibration (pose estimation) using the most recent sample
    for each active device.

    Returns
    -------
    list[ExtrinsicsResponse]
        List of extrinsics (rvec, tvec) for each successfully calibrated device.

    Raises
    ------
    HTTPException 400
        If no active cameras.
    HTTPException 422
        If calibration fails for all devices (e.g., insufficient samples).
    """
    active_devices = mgr.active_device_ids()
    if not active_devices:
        raise HTTPException(400, "No active cameras to calibrate")

    results = []
    for device_id in active_devices:
        try:
            extr = cal.extrinsics_calibration(device_id, EXTRINSICS_FILENAME)
            results.append(
                ExtrinsicsResponse(
                    device_id=device_id,
                    rvec=extr.rvec.flatten().tolist(),
                    tvec=extr.tvec.flatten().tolist(),
                )
            )
        except Exception as e:
            logger.error(f"Extrinsics calibration failed for {device_id}: {e}")
            continue

    if not results:
        # If we got here, it means all active devices failed to calibrate
        # (e.g. due to lack of samples)
        raise HTTPException(
            422,
            "Extrinsics calibration failed for all devices. Make sure you have captured enough samples.",
        )

    return results


@router.get("/extrinsics/{device_id}", response_model=ExtrinsicsResponse)
async def extrinsics(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    """
    Retrieve previously computed extrinsic parameters (GET).

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    ExtrinsicsResponse
        Rotation vector and translation vector.

    Raises
    ------
    HTTPException 404
        If extrinsics not found.
    """
    extrinsics = cal.get_extrinsics(device_id)
    if not extrinsics:
        raise HTTPException(
            status_code=404, detail="Extrinsics not found for this device"
        )
    return ExtrinsicsResponse(
        device_id=device_id,
        rvec=extrinsics.rvec.tolist(),
        tvec=extrinsics.tvec.tolist(),
    )


@router.get("/viz")
async def extrinsics_viz(
    cal: CalibrationManager = Depends(get_calibrator),
    mgr: CameraManager = Depends(get_manager),
):
    """
    Return extrinsics, intrinsics and board info for the 3D skeleton panel.
    """
    active = mgr.active_device_ids()
    cameras = []
    for dev_id in active:
        extr = cal.get_extrinsics(dev_id)
        if not extr:
            continue
        intr = cal.get_intrinsics(dev_id)
        info = mgr.get_info(dev_id)
        cam = {
            "device_id": dev_id,
            "rvec": extr.rvec.flatten().tolist(),
            "tvec": extr.tvec.flatten().tolist(),
        }
        if intr is not None:
            cam.update(
                fx=float(intr.fx),
                fy=float(intr.fy),
                cx=float(intr.cx),
                cy=float(intr.cy),
            )
        if info:
            shape = info.get("color_shape")
            if shape and len(shape) >= 2:
                cam["color_width"] = int(shape[1])
                cam["color_height"] = int(shape[0])
            elif info.get("color_intrinsics"):
                di = info["color_intrinsics"]
                cam["color_width"] = int(di.get("width", 0))
                cam["color_height"] = int(di.get("height", 0))
        cameras.append(cam)

    bp = cal.get_board_params()
    board = (
        {"board_size": list(bp.board_size), "square_size": bp.square_size}
        if bp
        else None
    )

    return {"board": board, "cameras": cameras}


@router.get("/{device_id}/stream")
def marked_stream(
    device_id: str,
    mgr: CameraManager = Depends(get_manager),
    cal: CalibrationManager = Depends(get_calibrator),
):
    """
    MJPEG stream of the camera with calibration board overlay.

    If a calibration worker exists, the stream shows detected markers/corners.
    Useful for checking board visibility during calibration.

    Parameters
    ----------
    device_id : str
        Camera ID.

    Returns
    -------
    StreamingResponse
        Multipart MJPEG stream.
    """
    logging.info("marked_stream started" + "!" * 10)
    return StreamingResponse(
        marked_camera_stream(mgr, cal, device_id, "color"),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )
