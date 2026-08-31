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

from viki.contracts import ArucoBoardParameters

logger = logging.getLogger(__name__)

from viki.calibration.manager import CalibrationManager
from viki.cameras.manager import CameraManager
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
    return cal.capture_all()


@router.post("/clear")
async def clear_all(cal: CalibrationManager = Depends(get_calibrator)):
    """Drop every sample on every camera and wipe the live capture photos."""
    cal.clear_all()
    return {"status": "cleared"}


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


# ── extrinsics presets ──────────────────────────────────────────────────────
# Named calibration sets under data/calibrations/. One is "active" (its name in
# ACTIVE_CALIBRATION); activating copies it onto EXTRINSICS_FILENAME.

import json as _json  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from viki.calibration import presets as _presets  # noqa: E402
from viki.config import EXTRINSICS_FILENAME as _EXTR_FILE  # noqa: E402


class _PresetName(BaseModel):
    name: str


# ── live capture sets ───────────────────────────────────────────────────────


@router.get("/samples")
async def list_sample_sets(cal: CalibrationManager = Depends(get_calibrator)):
    """One row per capture set: which cameras saw the board in it."""
    return cal.list_sample_sets()


@router.delete("/samples/{index}")
async def delete_sample_set(
    index: int, cal: CalibrationManager = Depends(get_calibrator)
):
    cal.delete_sample_set(index)
    return {"status": "deleted", "index": index}


def _capture_image(owner: str, index: int, device: str):
    from fastapi.responses import FileResponse

    from viki.calibration import captures

    p = captures.image_path(owner, index, device)
    if not p.is_file():
        raise HTTPException(404, "no capture image")
    return FileResponse(str(p), media_type="image/jpeg")


@router.get("/samples/{index}/{device}.jpg")
async def sample_image(index: int, device: str):
    """Preview photo of a live capture set (annotated with the detected board)."""
    from viki.calibration import captures

    return _capture_image(captures.LIVE, index, device)


@router.get("/presets/{name}/sets/{index}/{device}.jpg")
async def preset_sample_image(name: str, index: int, device: str):
    return _capture_image(name, index, device)


# ── presets ─────────────────────────────────────────────────────────────────


@router.get("/presets")
async def list_presets():
    """name, mtime, cameras, #sets, active flag."""
    return _presets.list_presets()


@router.get("/presets/{name}")
async def get_preset(name: str):
    try:
        return _presets.read_detail(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/save-as")
async def save_preset(
    body: _PresetName,
    cal: CalibrationManager = Depends(get_calibrator),
    mgr: CameraManager = Depends(get_manager),
):
    """Bundle the current solve + its capture sets + SDK intrinsics + board
    params into a named preset so it can be reopened and re-solved later."""
    try:
        extr = _json.loads(open(_EXTR_FILE).read())
    except (OSError, ValueError):
        raise HTTPException(400, "no current extrinsics — run the solve first")
    try:
        path = _presets.save_as(
            body.name,
            extrinsics=extr,
            sets=cal.sets_payload(),
            intrinsics=cal.color_intrinsics_payload(),
            board=cal.board_cfg(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _grab_k4a_best_effort(path.stem, mgr)  # cameras are live during calibration
    return {"status": "success", "name": path.stem}


def _collect_k4a_blobs(mgr: CameraManager) -> tuple[dict, int | None, int | None]:
    """Raw k4a calibration blob + depth-mode / colour-res enum ints from every
    running Kinect backend."""
    from viki.cameras.kinect import _COLOR_RES_MAP, _DEPTH_MODE_MAP

    blobs: dict[str, bytes] = {}
    depth_int = color_int = None
    for dev in mgr.active_device_ids():
        be = mgr.get_backend(dev)
        blob = getattr(be, "get_raw_calibration", lambda: None)() if be else None
        if not blob:
            continue
        blobs[dev] = blob
        cfg = be.config or {}
        depth_int = _DEPTH_MODE_MAP.get(cfg.get("depth_mode"))
        color_int = _COLOR_RES_MAP.get(
            (int(cfg.get("color_width", 0)), int(cfg.get("color_height", 0)))
        )
    return blobs, depth_int, color_int


def _grab_k4a_best_effort(name: str, mgr: CameraManager) -> None:
    try:
        blobs, di, ci = _collect_k4a_blobs(mgr)
        if blobs:
            _presets.attach_k4a(name, blobs, di, ci)
    except Exception:  # noqa: BLE001 — never break save-as on this
        logger.warning("grab k4a for preset %r failed", name, exc_info=True)


@router.post("/presets/{name}/grab-k4a")
async def grab_preset_k4a(
    name: str, mgr: CameraManager = Depends(get_manager)
):
    """Attach the running Kinects' raw calibration blob to an existing preset,
    so recordings made against it can do offline colour↔depth lifting without a
    re-record. Requires the Kinect cameras to be running."""
    try:
        _presets.read_detail(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    blobs, di, ci = _collect_k4a_blobs(mgr)
    if not blobs:
        raise HTTPException(400, "no raw calibration from running cameras — start the Kinects first")
    try:
        detail = _presets.attach_k4a(name, blobs, di, ci)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "success", "devices": sorted(blobs), "detail": detail}


def _collect_background(mgr: CameraManager, n: int = 30) -> dict:
    """Median depth (mm) over ~``n`` frames per running camera — the static
    empty scene captured during calibration. 0 stays 0 (no IR reading)."""
    import time

    import numpy as np

    stacks: dict[str, list] = {d: [] for d in mgr.active_device_ids()}
    for _ in range(n):
        for dev in list(stacks):
            fr = mgr.latest_frame(dev)
            if fr is not None and fr.has_depth():
                stacks[dev].append(np.asarray(fr.depth, dtype=np.float32))
        time.sleep(0.05)
    out: dict = {}
    for dev, frames in stacks.items():
        if not frames:
            continue
        arr = np.stack(frames)                 # (F, H, W) mm, 0 = missing
        arr[arr <= 0] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(arr, axis=0)
        out[dev] = np.nan_to_num(med, nan=0.0).astype(np.float32)
    return out


@router.post("/presets/{name}/grab-background")
async def grab_preset_background(
    name: str, mgr: CameraManager = Depends(get_manager)
):
    """Snapshot the empty scene's depth (per running camera) onto the preset so
    recordings made against it can subtract the static background from the point
    cloud. Run this during calibration, before the operator / objects enter."""
    try:
        _presets.read_detail(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    depths = _collect_background(mgr)
    if not depths:
        raise HTTPException(400, "no depth from running cameras — start the Kinects first")
    try:
        detail = _presets.attach_background(name, depths)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "success", "devices": sorted(depths), "detail": detail}


@router.post("/activate")
async def activate_preset(
    body: _PresetName, cal: CalibrationManager = Depends(get_calibrator)
):
    try:
        _presets.activate(body.name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    cal.load_all_extrinsics()
    return {"status": "success", "name": body.name}


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    try:
        _presets.delete(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "deleted", "name": name}


@router.delete("/presets/{name}/sets/{index}")
async def delete_preset_set(
    name: str, index: int, cal: CalibrationManager = Depends(get_calibrator)
):
    """Drop a capture set from a saved preset, re-solve, re-save. If the preset
    is active, the new extrinsics are pushed live too."""
    try:
        detail = _presets.delete_set(name, index)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if _presets.current_active() == name:
        cal.load_all_extrinsics()
    return detail
