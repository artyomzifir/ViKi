"""
viki.server.routes.recording
----------------------------
Record a synchronised RGB-D scene into a new episode directory
(:class:`viki.cameras.record.SceneRecorder`). The recording runs in a background
thread; poll the returned job id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from viki import config
from viki.cameras.manager import CameraManager
from viki.server import jobs
from viki.server.deps import get_manager

router = APIRouter(prefix="/record", tags=["recording"])


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
        report=report,
    )
    log(f"cloud done -> {out}")
    return {"cloud": out}


class CloudOpts(BaseModel):
    stride: int | None = None
    voxel: float | None = None
    max_points: int | None = None
    bbox: list[float] | None = None


class RecordRequest(BaseModel):
    seconds: float = 10.0
    fps: int = 15
    task: str = ""
    demonstrator: str = ""
    hand: str = "right"
    dataset: str | None = None
    cloud: CloudOpts | None = None


@router.post("/start")
async def start_recording(req: RecordRequest, mgr: CameraManager = Depends(get_manager)):
    if not mgr.active_device_ids():
        raise HTTPException(400, "no cameras are active — start cameras first")

    meta = {"task": req.task, "demonstrator": req.demonstrator, "hand": req.hand}
    episodes_dir = None if req.dataset else getattr(config, "EPISODES_DIR", "data/episodes")
    cloud_opts = req.cloud.model_dump() if req.cloud is not None else {}

    def _job():
        from viki.cameras.record import SceneRecorder

        rec = SceneRecorder(
            mgr, dataset=req.dataset, episodes_dir=episodes_dir, meta=meta
        )
        ep = rec.record(req.seconds, fps=req.fps)
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


@router.get("/jobs/{job_id}")
async def recording_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
