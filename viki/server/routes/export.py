"""viki.server.routes.export — labelled episodes -> LeRobot dataset (stub, paper §3.9)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki.server import jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    episodes: list[str]
    out_dir: str
    fps: int = 15


@router.post("")
async def start_export(req: ExportRequest):
    if not req.episodes:
        raise HTTPException(400, "no episodes given")

    logger.info("export: %d episode(s) -> %s @ %d fps", len(req.episodes), req.out_dir, req.fps)

    def _job():
        from viki.export import export_dataset

        return export_dataset(req.episodes, req.out_dir, fps=req.fps)

    return {"job_id": jobs.submit("export", _job)}


@router.get("/jobs/{job_id}")
async def export_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
