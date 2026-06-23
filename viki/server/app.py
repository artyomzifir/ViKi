"""
viki.server.app
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from viki.capture.manager import CameraManager

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = CameraManager()
    yield
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Capture Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class StartRequest(BaseModel):
    fps: int = 30
    color_width: int = 640
    color_height: int = 480
    # Kinect-only: MJPG avoids Python decode/re-encode for preview.
    color_format: str = "MJPG"
    depth_mode: str = "NFOV_UNBINNED"
    # Kinect-only: hardware sync wiring (ignored for RealSense)
    # 0 = standalone, 1 = master, 2 = subordinate
    wired_sync_mode: int = 0
    # Subordinate capture delay relative to master trigger, microseconds.
    # A small positive value (e.g. 160) staggers the depth IR projectors.
    subordinate_delay_us: int = 0
    # Require color and depth to arrive in the same capture (recommended for sync recording).
    synchronized_images_only: bool = False
    # Kinect-only: continuous depth->color transform. Expensive; keep off for realtime preview.
    align_depth_to_color: bool = False
    enable_depth_preview: bool = True
    preview_fps: int = 10
    preview_width: int = 640


class SnapshotRequest(BaseModel):
    aligned_depth: bool = True
    save: bool = True
    root_dir: str = "data/snapshots"


class PairSnapshotRequest(BaseModel):
    device_ids: list[str]
    aligned_depth: bool = True
    save: bool = True
    root_dir: str = "data/snapshots"


class PairSnapshotSeriesRequest(BaseModel):
    device_ids: list[str]
    count: int = 50
    interval_sec: float = 0.2
    settle_sec: float = 0.0
    retry_attempts: int = 3
    retry_delay_sec: float = 0.5
    aligned_depth: bool = True
    save: bool = True
    root_dir: str = "data/datasets/static_board_world/snapshots"
    stop_on_error: bool = True
    include_snapshots: bool = False


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/devices")
def list_devices():
    return app.state.manager.list_devices()


@app.post("/api/cameras/{device_id}/start")
def start_camera(device_id: str, req: StartRequest):
    try:
        app.state.manager.start(
            device_id,
            fps=req.fps,
            color_width=req.color_width,
            color_height=req.color_height,
            color_format=req.color_format,
            depth_mode=req.depth_mode,
            wired_sync_mode=req.wired_sync_mode,
            subordinate_delay_us=req.subordinate_delay_us,
            synchronized_images_only=req.synchronized_images_only,
            align_depth_to_color=req.align_depth_to_color,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "started", "device_id": device_id}


@app.post("/api/cameras/{device_id}/stop")
async def stop_camera(device_id: str):
    app.state.manager.stop(device_id)
    return {"status": "stopped", "device_id": device_id}


@app.get("/api/cameras/{device_id}/info")
async def camera_info(device_id: str):
    info = app.state.manager.get_info(device_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Camera not found or not started")
    return info


@app.post("/api/cameras/{device_id}/snapshot")
def camera_snapshot(
    device_id: str,
    req: SnapshotRequest | None = Body(default=None),
    aligned_depth: bool = True,
    save: bool = True,
):
    if req is not None:
        aligned_depth = req.aligned_depth
        save = req.save
        root_dir = req.root_dir
    else:
        root_dir = "data/snapshots"
    try:
        return app.state.manager.snapshot(
            device_id,
            aligned_depth=aligned_depth,
            save=save,
            root_dir=root_dir,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/capture/pair_snapshot")
def pair_snapshot(req: PairSnapshotRequest):
    try:
        return app.state.manager.pair_snapshot(
            req.device_ids,
            aligned_depth=req.aligned_depth,
            save=req.save,
            root_dir=req.root_dir,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/capture/pair_snapshot_series")
def pair_snapshot_series(req: PairSnapshotSeriesRequest):
    if req.count < 1:
        raise HTTPException(status_code=400, detail="count must be >= 1")
    if req.count > 1000:
        raise HTTPException(status_code=400, detail="count must be <= 1000")
    if req.interval_sec < 0:
        raise HTTPException(status_code=400, detail="interval_sec must be >= 0")
    if req.settle_sec < 0:
        raise HTTPException(status_code=400, detail="settle_sec must be >= 0")
    if req.retry_attempts < 1:
        raise HTTPException(status_code=400, detail="retry_attempts must be >= 1")
    if req.retry_attempts > 20:
        raise HTTPException(status_code=400, detail="retry_attempts must be <= 20")
    if req.retry_delay_sec < 0:
        raise HTTPException(status_code=400, detail="retry_delay_sec must be >= 0")

    snapshots = []
    snapshot_roots = []
    errors = []
    started_at = time.time()
    if req.settle_sec > 0:
        time.sleep(req.settle_sec)

    for index in range(req.count):
        snapshot = None
        last_error = None
        for attempt in range(1, req.retry_attempts + 1):
            try:
                snapshot = app.state.manager.pair_snapshot(
                    req.device_ids,
                    aligned_depth=req.aligned_depth,
                    save=req.save,
                    root_dir=req.root_dir,
                )
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                last_error = str(e)
                print(
                    f"[capture-series] snapshot {index + 1}/{req.count} "
                    f"failed attempt {attempt}/{req.retry_attempts}: {last_error}"
                )
                if attempt < req.retry_attempts and req.retry_delay_sec > 0:
                    time.sleep(req.retry_delay_sec)

        if snapshot is None:
            errors.append({"index": index, "error": last_error})
            if req.stop_on_error:
                break
        else:
            snapshot["series_index"] = index
            root = snapshot.get("root")
            snapshot_roots.append(
                {
                    "series_index": index,
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "root": root,
                }
            )
            if req.include_snapshots:
                snapshots.append(snapshot)
            print(f"[capture-series] saved {index + 1}/{req.count}: {root}")

        if index < req.count - 1 and req.interval_sec > 0:
            time.sleep(req.interval_sec)

    saved_count = len(snapshot_roots)
    completed = saved_count == req.count and not errors
    return {
        "status": "completed" if completed else "partial",
        "requested_count": req.count,
        "saved_count": saved_count,
        "error_count": len(errors),
        "root_dir": req.root_dir,
        "device_ids": req.device_ids,
        "elapsed_sec": time.time() - started_at,
        "snapshot_roots": snapshot_roots,
        "snapshots": snapshots,
        "errors": errors,
    }


@app.get("/api/cameras/{device_id}/stream")
def colour_stream(device_id: str, preview_fps: int = 10):
    return StreamingResponse(
        _mjpeg_gen(app.state.manager, device_id, "color", preview_fps=preview_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

@app.get("/api/cameras/{device_id}/depth")
def depth_stream(device_id: str, preview_fps: int = 10, preview_width: int = 640):
    return StreamingResponse(
        _mjpeg_gen(
            app.state.manager,
            device_id,
            "depth",
            preview_fps=preview_fps,
            preview_width=preview_width,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# Minimum fraction of pixels that must be valid for a depth frame to be displayed.
# Frames below this threshold (blank / missing depth) are dropped and the last
# good image is held instead.
_DEPTH_MIN_VALID_FRACTION = 0.05
_DEPTH_PREVIEW_MIN_MM = 500.0
_DEPTH_PREVIEW_MAX_MM = 4000.0


def _mjpeg_gen(
    mgr: CameraManager,
    device_id: str,
    kind: str,
    preview_fps: int = 10,
    preview_width: int = 640,
):
    last_ts = -1
    last_good_depth_img: np.ndarray | None = None
    min_interval_s = 1.0 / max(preview_fps, 1)
    last_emit_s = 0.0

    while True:
        now_s = time.monotonic()
        sleep_s = min_interval_s - (now_s - last_emit_s)
        if sleep_s > 0:
            time.sleep(min(sleep_s, 0.05))
            continue

        frame = mgr.latest_frame(device_id)
        data: bytes | None = None

        if frame is None:
            if device_id not in mgr.active_device_ids():
                return
            img = _placeholder(640, 480, f"{device_id}: not started")
            last_ts = -1
        elif frame.host_timestamp_us == last_ts:
            time.sleep(0.005)
            continue
        else:
            last_ts = frame.host_timestamp_us
            if kind == "color":
                if frame.color_jpeg is not None:
                    data = frame.color_jpeg
                    img = None
                else:
                    img = frame.color
            else:
                depth = frame.depth
                depth_preview = _resize_depth_preview(depth, preview_width)
                valid = depth_preview[depth_preview > 0]
                valid_fraction = valid.size / max(depth_preview.size, 1)

                if valid_fraction < _DEPTH_MIN_VALID_FRACTION:
                    # Blank or mostly-zero frame (missing depth capture from SDK).
                    # Hold the last good image so the stream doesn't flash black.
                    if last_good_depth_img is None:
                        time.sleep(0.005)
                        continue
                    img = last_good_depth_img
                else:
                    norm = np.clip(
                        (depth_preview.astype(np.float32) - _DEPTH_PREVIEW_MIN_MM)
                        / (_DEPTH_PREVIEW_MAX_MM - _DEPTH_PREVIEW_MIN_MM),
                        0,
                        1,
                    )
                    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    last_good_depth_img = img

        if data is None:
            _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = jpeg.tobytes()
        last_emit_s = time.monotonic()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(data)).encode()
            + b"\r\n\r\n" + data + b"\r\n"
        )


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 80, 80), 2, cv2.LINE_AA)
    return img


def _resize_depth_preview(depth: np.ndarray, preview_width: int) -> np.ndarray:
    if depth.size == 0 or preview_width <= 0:
        return depth
    h, w = depth.shape[:2]
    if w <= preview_width:
        return depth
    preview_height = max(1, int(round(h * preview_width / w)))
    return cv2.resize(
        depth,
        (preview_width, preview_height),
        interpolation=cv2.INTER_NEAREST,
    )
