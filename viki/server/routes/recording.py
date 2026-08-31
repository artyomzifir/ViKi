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


class RecordRequest(BaseModel):
    seconds: float = 10.0
    fps: int = 15
    task: str = ""
    demonstrator: str = ""
    hand: str = "right"
    dataset: str | None = None


@router.post("/start")
async def start_recording(req: RecordRequest, mgr: CameraManager = Depends(get_manager)):
    if not mgr.active_device_ids():
        raise HTTPException(400, "no cameras are active — start cameras first")

    meta = {"task": req.task, "demonstrator": req.demonstrator, "hand": req.hand}
    episodes_dir = None if req.dataset else getattr(config, "EPISODES_DIR", "data/episodes")

    def _job():
        from viki.cameras.record import SceneRecorder

        rec = SceneRecorder(
            mgr, dataset=req.dataset, episodes_dir=episodes_dir, meta=meta
        )
        ep = rec.record(req.seconds, fps=req.fps)
        return {"episode": str(ep.root), "dataset": req.dataset}

    return {"job_id": jobs.submit("record", _job, queued=False)}


@router.get("/jobs/{job_id}")
async def recording_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
