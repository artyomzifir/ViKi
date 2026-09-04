"""
viki.server.routes.recording
----------------------------
Record a synchronised RGB-D scene into a new episode directory
(:class:`viki.cameras.record.SceneRecorder`). The recording runs in a background
thread; poll the returned job id.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from viki import config
from viki.cameras.hw_sync import HardwareSyncError
from viki.cameras.manager import CameraManager
from viki.server import jobs
from viki.server.deps import get_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/record", tags=["recording"])

# Set by POST /record/stop to end the in-progress capture early (Stop button).
# Cleared at the start of every recording.
_stop_evt = threading.Event()


def _build_cloud(ep, report, log, opts: dict | None = None) -> dict:
    """Queue-worker entry: build the episode's point cloud, reporting progress."""
    from viki.perception.cloud import build_cloud

    o = opts or {}
    log(f"building point cloud for {ep.id}")
    out = build_cloud(
        ep,
        stride=o.get("stride"),
        voxel=o.get("voxel"),
        bbox=o.get("bbox"),
        max_points=o.get("max_points"),
        bg_subtract=o.get("bg_subtract"),
        bg_tol_mm=o.get("bg_tol_mm"),
        report=report,
    )
    log(f"cloud done -> {out}")
    return {"cloud": out}


class CloudOpts(BaseModel):
    stride: int | None = None
    voxel: float | None = None
    max_points: int | None = None
    bbox: list[float] | None = None
    bg_subtract: bool | None = None
    bg_tol_mm: float | None = None


class RecordRequest(BaseModel):
    seconds: float = 10.0
    fps: int = 15
    task: str = ""
    demonstrator: str = ""
    hand: str = "right"
    dataset: str | None = None
    cloud: CloudOpts | None = None
    allow_amber: bool = False  # start on an 'amber' validation verdict (explicit)
    force: bool = False         # skip the whole setup-artifact gate (debug)


@router.post("/start")
async def start_recording(req: RecordRequest, mgr: CameraManager = Depends(get_manager)):
    if not mgr.active_device_ids():
        raise HTTPException(400, "no cameras are active — start cameras first")

    # This gate is not bypassed by ``force``: a two-Kinect take recorded in
    # standalone/software-sync mode is invalid input, not a debug variation.
    try:
        mgr.require_hardware_sync_ready()
    except HardwareSyncError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Setup-artifact gate (spec §7): the active preset must have fresh extrinsics,
    # a world anchor, a background plate and a non-red validation report whose
    # hash still matches the extrinsics.
    if not req.force:
        from viki.calibration import artifacts as _artifacts
        from viki.calibration.presets import current_active

        preset = current_active()
        if not preset:
            raise HTTPException(409, "no active calibration preset — calibrate and activate one first")
        ok, why = _artifacts.record_ready(preset, allow_amber=req.allow_amber)
        if not ok:
            raise HTTPException(409, f"calibration not record-ready: {why}")

    logger.info(
        "recording: dataset=%s cameras=%s max=%.0fs fps=%d task=%r cloud=%s",
        req.dataset or "(flat)", sorted(mgr.active_device_ids()), req.seconds, req.fps,
        req.task, "on" if req.cloud is not None else "default",
    )
    meta = {"task": req.task, "demonstrator": req.demonstrator, "hand": req.hand}
    episodes_dir = None if req.dataset else getattr(config, "EPISODES_DIR", "data/episodes")
    cloud_opts = req.cloud.model_dump() if req.cloud is not None else {}
    _stop_evt.clear()

    def _job():
        from viki.cameras.record import SceneRecorder

        rec = SceneRecorder(
            mgr, dataset=req.dataset, episodes_dir=episodes_dir, meta=meta
        )
        ep = rec.record(req.seconds, fps=req.fps, stop_event=_stop_evt)
        # Build the coloured point cloud straight away (params from the request,
        # else the CLOUD_* config keys) so the episode is viewer-ready without a
        # manual step; on its own 'cloud' lane so it runs in parallel with any
        # model run. The perception / model run stays a separate, explicit action.
        cloud_job = jobs.submit(
            "cloud",
            lambda report, log: _build_cloud(ep, report, log, cloud_opts),
            episode=ep.id,
            lane="cloud",
        )
        return {"episode": str(ep.root), "dataset": req.dataset, "cloud_job": cloud_job}

    return {"job_id": jobs.submit("record", _job, queued=False)}


@router.post("/stop")
async def stop_recording():
    """End the in-progress capture now (the recorder still finalises the file)."""
    _stop_evt.set()
    logger.info("recording: stop requested")
    return {"status": "stopping"}


@router.get("/jobs/{job_id}")
async def recording_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
