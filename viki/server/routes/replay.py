"""viki.server.routes.replay — plan.h5 -> replay.h5 (stub stage, paper §3.8)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki.contracts import Episode
from viki.server import jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/replay", tags=["replay"])


class ReplayRequest(BaseModel):
    episode: str
    driver: str = "dryrun"  # dryrun | ur3
    max_resolves: int = 0


@router.post("")
async def start_replay(req: ReplayRequest):
    ep = Episode(root=Path(req.episode))
    if not ep.plan_h5.exists():
        raise HTTPException(404, f"no plan.h5 for episode {req.episode}; run retarget first")

    logger.info("replay: episode=%s driver=%s max_resolves=%d", ep.id, req.driver, req.max_resolves)

    def _job():
        from viki.replay import replay_episode

        return replay_episode(ep, driver=req.driver, max_resolves=req.max_resolves)

    return {"job_id": jobs.submit("replay", _job, episode=ep.id)}


@router.get("/jobs/{job_id}")
async def replay_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
