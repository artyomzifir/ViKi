"""
viki.server.routes.calibration
------------------------------
Calibration endpoints: live mosaic preview, sample capture, running the
calibration solve, status, and clearing collected samples.
"""

from __future__ import annotations

import asyncio
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
    ExtrinsicsResponse,
)
from viki.config import EXTRINSICS_FILENAME

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
    res = cal.capture_all()
    logger.info("calibration capture-all: %s", res)
    return res


@router.get("/readiness")
async def readiness(cal: CalibrationManager = Depends(get_calibrator)):
    """Solve-ready criteria over the collected capture sets (spec §4.2): set
    count, all-camera-covisible count, tilted-set count and per-camera frame
    coverage, each with its threshold, plus a single ``ready`` flag that gates
    the *Solve* button."""
    if not cal._workers:
        raise HTTPException(400, "no calibration session — start one first")
    return cal.readiness()


@router.post("/solve")
async def solve_bundle(
    force: bool = False, cal: CalibrationManager = Depends(get_calibrator)
):
    """Joint multi-pose bundle solve (spec §4.3) over every collected set.
    Refused unless the readiness criteria are met (pass ``force=true`` to
    override — the result carries ``solve.degenerate`` when the geometry is
    under-constrained)."""
    if not cal._workers:
        raise HTTPException(400, "no calibration session — start one first")
    rd = cal.readiness()
    if not rd["ready"] and not force:
        unmet = [c["name"] for c in rd["criteria"] if not c["ok"]]
        raise HTTPException(422, f"not ready to solve — unmet: {', '.join(unmet)}")
    try:
        out = cal.solve_bundle_live(EXTRINSICS_FILENAME)
    except (ValueError, RuntimeError) as exc:
        logger.warning("bundle solve failed: %s", exc)
        raise HTTPException(422, f"bundle solve failed: {exc}") from exc
    return {"reference_device": out["reference_device"], "solve": out["solve"]}


@router.post("/anchor")
async def capture_anchor(cal: CalibrationManager = Depends(get_calibrator)):
    """Anchor step (spec §5): board at the marked home spot, ONE frame →
    ``T_world_display`` (viz / AABB / export only — never the solve)."""
    if not cal._workers:
        raise HTTPException(400, "no calibration session — start one first")
    try:
        return cal.capture_anchor()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/anchor")
async def get_anchor(cal: CalibrationManager = Depends(get_calibrator)):
    a = cal.world_anchor()
    if a is None:
        raise HTTPException(404, "no world anchor — run the Anchor step")
    return a


@router.post("/validate")
async def validate(cal: CalibrationManager = Depends(get_calibrator)):
    """Validate step (spec §6): build a per-camera empty-scene cloud in the rig
    frame and score how well the cameras agree. Returns the report; a ``red``
    verdict (or a stale one) blocks recording."""
    if not cal._workers:
        raise HTTPException(400, "no calibration session — start one first")
    try:
        return await asyncio.to_thread(cal.validate_live)
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/validate")
async def get_validation(cal: CalibrationManager = Depends(get_calibrator)):
    import json as _json

    from viki.config import VALIDATION_FILENAME

    try:
        return _json.loads(open(VALIDATION_FILENAME).read())
    except (OSError, ValueError):
        raise HTTPException(404, "no validation report — run the Validate step")


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


@router.get("/intrinsics/{device_id}")
async def intrinsics(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    """The running camera's SDK colour intrinsics (read-only, for inspection).

    There is no intrinsics calibration / storage any more — the SDK is the only
    source of truth. 404 when the camera isn't live.
    """
    intr = cal.get_intrinsics(device_id)
    if not intr:
        raise HTTPException(404, f"{device_id} is not running — no SDK intrinsics")
    return {
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        "dist_coeffs": intr.dist_coeffs.tolist(), "source": "sdk",
    }


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

    logger.info("extrinsics solve: devices=%s", sorted(active_devices))
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
        logger.warning("extrinsics solve failed for every device")
        raise HTTPException(
            422,
            "Extrinsics calibration failed for all devices. Make sure you have captured enough samples.",
        )

    logger.info("extrinsics solved for %s", [r.device_id for r in results])
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
        logger.warning("save-as %r -> 400 %s", body.name, exc)
        raise HTTPException(400, str(exc)) from exc
    logger.info("calibration preset %r saved (%d cams)", path.stem, len(extr))
    # Fold the setup artifacts into calib/<preset>/ : migrate the just-written
    # legacy JSON to extrinsics.json, then snapshot the live world anchor.
    try:
        from viki.calibration import artifacts as _artifacts

        _artifacts.ensure_migrated(path.stem)
        live_anchor = cal.world_anchor()
        if live_anchor and live_anchor.get("observations"):
            _artifacts.write_world_anchor(
                path.stem,
                observations=live_anchor["observations"],
                T_world_display=live_anchor.get("T_world_display"),
            )
    except Exception:  # noqa: BLE001 — never break save-as on this
        logger.warning("preset %r: setup-artifact fold-in failed", path.stem, exc_info=True)
    # Cameras are live during calibration, so grab both device/scene snapshots
    # now — no separate button. k4a raw calibration is a device property; the
    # background depth is the static scene as it sits during the solve (the
    # ChArUco board included — it's part of the fixed workspace).
    await asyncio.to_thread(_grab_k4a_best_effort, path.stem, mgr)
    await asyncio.to_thread(_grab_background_best_effort, path.stem, mgr)  # ~1.5 s of depth median
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
            logger.info("preset %r: k4a calibration attached for %s", name, sorted(blobs))
        else:
            logger.warning(
                "preset %r: no k4a raw calibration from the running cameras "
                "(none active, or not Kinect backends)", name,
            )
    except Exception:  # noqa: BLE001 — never break save-as on this
        logger.warning("grab k4a for preset %r failed", name, exc_info=True)


def _grab_background_best_effort(name: str, mgr: CameraManager) -> None:
    try:
        depths = _collect_background(mgr)
        if depths:
            _presets.attach_background(name, depths)
            logger.info("preset %r: empty-scene background attached for %s", name, sorted(depths))
        else:
            logger.warning(
                "preset %r: no background depth captured — the running cameras "
                "produced no depth frames in the ~1.5 s window (active: %s). "
                "Make sure depth is streaming, then Save again.",
                name, sorted(mgr.active_device_ids()),
            )
    except Exception:  # noqa: BLE001 — never break save-as on this
        logger.warning("grab background for preset %r failed", name, exc_info=True)


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


@router.post("/activate")
async def activate_preset(
    body: _PresetName, cal: CalibrationManager = Depends(get_calibrator)
):
    try:
        _presets.activate(body.name)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("activate preset %r -> 404 %s", body.name, exc)
        raise HTTPException(404, str(exc)) from exc
    cal.load_all_extrinsics()
    logger.info("calibration preset %r activated", body.name)
    return {"status": "success", "name": body.name}


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    try:
        _presets.delete(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    logger.info("calibration preset %r deleted", name)
    return {"status": "deleted", "name": name}


class _PresetRename(BaseModel):
    new: str


@router.patch("/presets/{name}")
async def rename_preset(name: str, body: _PresetRename):
    try:
        path = _presets.rename(name, body.new)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    logger.info("calibration preset %r renamed to %r", name, path.stem)
    return {"status": "renamed", "name": path.stem}


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
