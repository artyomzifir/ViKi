"""
viki.server.routes.pipeline
---------------------------
Offline stage endpoints: the episode-oriented API the web UI uses — list
episodes, run a stage as a job, and fetch an episode's 3-D geometry for the
viewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki import config
from viki.contracts import Episode
from viki.episode import read_status
from viki.server import jobs

router = APIRouter()

_ep = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _episode(path_or_id: str) -> Episode:
    p = Path(path_or_id)
    if not p.is_absolute() and not p.exists():
        p = Path(getattr(config, "EPISODES_DIR", "data/episodes")) / path_or_id
    ep = Episode(root=p)
    if not ep.root.is_dir():
        raise HTTPException(404, f"no episode at {path_or_id}")
    return ep


# ── listing ───────────────────────────────────────────────────────────


@_ep.get("/episodes")
async def list_episodes():
    root = Path(getattr(config, "EPISODES_DIR", "data/episodes"))
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        ep = Episode(root=d)
        meta = json.loads(ep.meta_path.read_text()) if ep.meta_path.exists() else {}
        out.append(
            {
                "id": ep.id,
                "path": str(d),
                "task": (meta.get("labels") or {}).get("task", meta.get("task", "")),
                "stages": read_status(ep).get("stages", {}),
                "has": {
                    "raw": ep.raw_dir.is_dir(),
                    "rec": ep.rec_npz.exists(),
                    "cln": ep.cln_npz.exists(),
                    "plan": ep.plan_h5.exists(),
                    "replay": ep.replay_h5.exists(),
                },
            }
        )
    return {"episodes": out}


# ── stage jobs ────────────────────────────────────────────────────────


class _EpReq(BaseModel):
    episode: str
    backend: str | None = None
    robot: str | None = None
    window: int = 7
    polyorder: int = 2


@_ep.post("/extract")
async def extract(req: _EpReq):
    ep = _episode(req.episode)

    def _job():
        from viki.perception.extract import extract_episode

        return extract_episode(ep, backend=req.backend)

    return {"job_id": jobs.submit("extract", _job)}


@_ep.post("/prepare")
async def prepare(req: _EpReq):
    ep = _episode(req.episode)

    def _job():
        from viki.prepare.run import prepare_episode

        return prepare_episode(ep, req.window, req.polyorder)

    return {"job_id": jobs.submit("prepare", _job)}


@_ep.post("/retarget")
async def retarget(req: _EpReq):
    ep = _episode(req.episode)

    def _job():
        from viki.retarget.run import retarget_episode

        return retarget_episode(ep, robot=req.robot)

    return {"job_id": jobs.submit("retarget", _job)}


@_ep.get("/jobs/{job_id}")
async def job_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j


@_ep.get("/jobs")
async def job_list():
    return {"jobs": jobs.all_jobs()}


# ── geometry for the 3-D viewer ──────────────────────────────────────


@_ep.get("/episode/{ep_id}/geometry")
async def geometry(ep_id: str, include_raw: int = 0):
    ep = _episode(ep_id)
    out: dict = {"id": ep.id, "cameras": {}, "n_frames": 0}

    extr_path = ep.raw_dir / "extrinsics.json"
    if extr_path.exists():
        from viki.render.robot_viz_shared import camera_gaze_dir, camera_world_pos

        for dev, e in json.loads(extr_path.read_text()).items():
            rvec, tvec = e.get("rvec"), e.get("tvec")
            if rvec is None or tvec is None:
                continue
            out["cameras"][dev] = {
                "pos": camera_world_pos(rvec, tvec).tolist(),
                "forward": camera_gaze_dir(rvec).tolist(),
            }

    if ep.cln_npz.exists():
        with np.load(ep.cln_npz) as d:
            out["n_frames"] = int(len(d["positions"]))
            out["fps"] = float(
                1e6 / max(np.median(np.diff(d["timestamps"].astype(float))), 1.0)
            )
            out["wrist_traj"] = np.asarray(d["positions"], np.float32).tolist()
            out["palm_rot"] = (
                np.asarray(d["rotations"], np.float32).reshape(-1, 9).tolist()
            )
            out["valid"] = np.asarray(d["valid"], bool).tolist()

    if include_raw and ep.rec_npz.exists():
        with np.load(ep.rec_npz) as d:
            devs = [str(x) for x in d["device_ids"]]
            pts = np.asarray(d["points"], np.float32)
            raw: dict[str, list] = {}
            for dev in sorted(set(devs)):
                mask = np.array([x == dev for x in devs])
                cloud = pts[mask].reshape(-1, 3)
                cloud = cloud[np.isfinite(cloud).all(axis=1)]
                raw[dev] = cloud[::3].tolist()  # decimate
            out["raw_points"] = raw

    return out


router.include_router(_ep)
