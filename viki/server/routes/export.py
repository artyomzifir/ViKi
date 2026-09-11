"""viki.server.routes.export — retargeted episodes -> a dataset (paper §3.9)."""

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
    # "trajectory" needs nothing beyond numpy; "lerobot" needs the optional
    # extra and episodes that have been replayed and screened.
    format: str = "trajectory"
    name: str | None = None
    fps: int = 15


@router.post("")
async def start_export(req: ExportRequest):
    if not req.episodes:
        raise HTTPException(400, "no episodes given")

    if req.format not in {"trajectory", "lerobot"}:
        raise HTTPException(422, "format must be 'trajectory' or 'lerobot'")
    logger.info("export: %d episode(s) -> %s (%s)", len(req.episodes), req.out_dir, req.format)

    def _job():
        if req.format == "lerobot":
            from viki.export import export_dataset

            return export_dataset(req.episodes, req.out_dir, fps=req.fps)
        from viki.export import export_trajectories

        return export_trajectories(req.episodes, req.out_dir, name=req.name)

    return {"job_id": jobs.submit("export", _job)}


@router.get("/jobs/{job_id}")
async def export_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
