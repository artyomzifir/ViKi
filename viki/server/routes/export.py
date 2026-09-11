"""viki.server.routes.export — retargeted episodes -> a dataset (paper §3.9)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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

            return {"out_dir": export_dataset(req.episodes, req.out_dir, fps=req.fps)}
        from viki.export import export_trajectories

        out = export_trajectories(req.episodes, req.out_dir, name=req.name)
        # Hand the UI what was actually written rather than only where: a
        # bundle that silently skipped half its episodes should say so.
        manifest = json.loads((Path(out) / "dataset.json").read_text())
        return {
            "out_dir": out,
            "name": manifest["name"],
            "episode_count": manifest["episode_count"],
            "total_frames": manifest["total_frames"],
            "robots": manifest["robots"],
            "skipped": manifest["skipped"],
            "screening": manifest["screening"],
        }

    return {"job_id": jobs.submit("export", _job)}


@router.get("/jobs/{job_id}")
async def export_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
