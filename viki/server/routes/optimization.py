"""
viki.server.routes.optimization
-------------------------------
Endpoints for converting raw skeleton recordings into prepared data:
listing raw recordings and applying Savitzky-Golay smoothing (raw -> prepared).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

import viki.config as config
from viki.optimization.preparation.processor import PreparationPipeline
from viki.server.deps import get_processor
from viki.server.smooth_viz import smooth_trajectory_stream
from viki.viz.smooth_viz_shared import SmoothVizConfig

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}

router = APIRouter(prefix="/optimization", tags=["optimization"])
logger = logging.getLogger(__name__)


class SmoothRequest(BaseModel):
    filename: str
    window_length: int = 7
    polyorder: int = 2


@router.get("/recordings")
async def list_recordings(
    page: int = 0,
    limit: int = 10,
    processor: PreparationPipeline = Depends(get_processor),
):
    """
    List raw skeleton recordings (rec-*.npz), paginated.

    Parameters
    ----------
    page : int, default=0
        Page number (zero-based).
    limit : int, default=10
        Number of recordings per page.

    Returns
    -------
    dict
        {"recordings": list[str]} – list of filenames.
    """
    return {"recordings": processor.list_recordings(page=page, page_size=limit)}


@router.post("/smooth")
async def smooth_recording(
    req: SmoothRequest,
    processor: PreparationPipeline = Depends(get_processor),
):
    """
    Apply Savitzky-Golay smoothing to a raw recording, producing a prepared
    (cln-*.npz) file with smoothed landmarks and end-effector poses.

    Parameters
    ----------
    req : SmoothRequest
        Filename, window length, and polynomial order.

    Returns
    -------
    dict
        {"status": "success", "path": str} – path to the prepared file.

    Raises
    ------
    HTTPException 404
        If file not found.
    HTTPException 400
        If smoothing parameters are invalid.
    HTTPException 500
        If an internal error occurs.
    """
    try:
        path, _ = processor.smooth_recording(
            req.filename,
            window_length=req.window_length,
            polyorder=req.polyorder,
        )
        return {"status": "success", "path": path}
    except FileNotFoundError:
        raise HTTPException(404, f"Recording {req.filename} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Smoothing failed")
        raise HTTPException(500, f"Smoothing failed: {str(e)}")


@router.get("/smoothed-recordings")
async def list_smoothed_recordings(page: int = 0, limit: int = 10):
    smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in smoothed_dir.glob("cln-*.npz")], reverse=True)
    start = page * limit
    end = start + limit
    return {"recordings": files[start:end]}


@router.get("/smooth-stream")
async def smooth_viz_stream(
    filename: str,
    show_raw: bool = True,
    show_smooth: bool = True,
    axes_length: float = 1.0,
    center_on: str = "world",
):
    smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
    npz_path = smoothed_dir / filename
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"Smoothed recording not found: {filename}")
    cfg = SmoothVizConfig(
        show_raw=show_raw,
        show_smooth=show_smooth,
        axes_length=axes_length,
        center_on=center_on,
    )
    return StreamingResponse(
        smooth_trajectory_stream(npz_path, cfg=cfg),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )


@router.get("/smooth-plot")
async def smooth_plot(filename: str):
    """
    Return a PNG comparing raw and smoothed wrist trajectories.

    Parameters
    ----------
    filename : str
        Prepared (cln-*.npz) file name.

    Returns
    -------
    Response
        PNG image.

    Raises
    ------
    HTTPException 404
        If file not found.
    """
    smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
    npz_path = smoothed_dir / filename
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"Smoothed recording not found: {filename}")

    with np.load(npz_path) as data:
        positions = data["positions"]
        timestamps = data["timestamps"]
        raw_points = data.get("raw_points")
        landmark_ids = data.get("landmark_ids")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t_sec = (timestamps - timestamps[0]) / 1_000_000

    labels = ["X", "Y", "Z"]
    colors_raw = ["#e74c3c", "#e67e22", "#3498db"]
    colors_smooth = ["#2ecc71", "#1abc9c", "#9b59b6"]

    for i, (ax, label, cr, cs) in enumerate(zip(axes, labels, colors_raw, colors_smooth)):
        ax.plot(t_sec, positions[:, i], color=cs, linewidth=2, label="Smoothed" if i == 0 else None)
        if raw_points is not None and landmark_ids is not None:
            wrist_col = int(np.where(landmark_ids == 0)[0][0])
            ax.plot(t_sec, raw_points[:, wrist_col, i], color=cr, linewidth=1, alpha=0.5, label="Raw" if i == 0 else None)
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Smoothing comparison — {filename}", fontsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
