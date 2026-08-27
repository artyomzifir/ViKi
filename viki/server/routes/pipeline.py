"""
viki.server.routes.pipeline
---------------------------
Offline stage endpoints (extract / prepare / retarget). Groups the existing
optimization (raw -> prepared) and dataset (prepared -> plan .h5) routers under
one ``/api`` include, plus an ``extract`` job.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki.contracts import Episode
from viki.server import jobs
from viki.server.routes import dataset, optimization

router = APIRouter()
router.include_router(optimization.router)
router.include_router(dataset.router)

_ep_router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class ExtractRequest(BaseModel):
    episode: str  # path to episodes/<id>
    backend: str | None = None


@_ep_router.post("/extract")
async def extract(req: ExtractRequest):
    ep = Episode(root=Path(req.episode))
    if not ep.raw_dir.is_dir():
        raise HTTPException(404, f"no raw frames for episode {req.episode}")

    def _job():
        from viki.perception.extract import extract_episode

        return extract_episode(ep, backend=req.backend)

    return {"job_id": jobs.submit("extract", _job)}


@_ep_router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j


@_ep_router.get("/jobs")
async def job_list():
    return {"jobs": jobs.all_jobs()}


router.include_router(_ep_router)
