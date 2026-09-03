"""
viki.server.routes.pipeline
---------------------------
Offline stage endpoints: the episode-oriented API the web UI uses — list
episodes, run a stage as a job, and fetch an episode's 3-D geometry for the
viewer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from viki import config, datasets
from viki.contracts import Episode, cln_pose_keys
from viki.episode import read_status
from viki.server import jobs

logger = logging.getLogger(__name__)
router = APIRouter()

_ep = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _episode(path_or_id: str) -> Episode:
    p = Path(path_or_id)
    if not p.is_dir():
        # bare id: look in the legacy flat dir, then every dataset
        cand = datasets.episodes_root() / path_or_id
        if not cand.is_dir():
            for ds in datasets.datasets_root().glob("*/"):
                if (ds / path_or_id).is_dir():
                    cand = ds / path_or_id
                    break
        p = cand
    ep = Episode(root=p)
    if not ep.root.is_dir():
        raise HTTPException(404, f"no episode at {path_or_id}")
    return ep


# ── listing ───────────────────────────────────────────────────────────


@_ep.get("/episodes")
async def list_episodes(dataset: str | None = None):
    """All episodes (grouped datasets + legacy flat dir), or one dataset's."""
    return {"episodes": datasets.list_episodes(dataset)}


# ── stage jobs ────────────────────────────────────────────────────────


class _EpReq(BaseModel):
    episode: str
    model: str | None = None
    robot: str | None = None
    window: int = 7
    polyorder: int = 2


class _PerceiveReq(BaseModel):
    episodes: list[str]
    opts: dict = {}


class _ModelReq(BaseModel):
    model: str


@_ep.post("/perceive")
async def perceive(req: _PerceiveReq):
    """Queue the full perception stage (extract → fuse → smooth → EE → gripper
    → cln.npz) for one or more episodes. When ``opts.build_cloud`` is set the
    point cloud is queued as a **separate** job on the ``cloud`` lane so it
    computes in parallel with the model run."""
    logger.info("perceive: %d episode(s) %s opts=%s",
                len(req.episodes), [str(e).split('/')[-1] for e in req.episodes], req.opts)
    ids: list[str] = []
    for ep_ref in req.episodes:
        ep = _episode(ep_ref)
        opts = dict(req.opts)
        # The cloud is built right after recording; Extract only rebuilds it when
        # the "regenerate cloud" box is ticked (e.g. to widen the workspace AABB).
        want_cloud = bool(opts.pop("build_cloud", False))
        cloud_stride = opts.pop("cloud_stride", None)
        cloud_bbox = opts.pop("cloud_bbox", None)
        cloud_voxel = opts.pop("cloud_voxel", None)

        def _job(report, log, ep=ep, opts=opts):
            from viki.perception.run import perceive_episode

            log(f"perceive {ep.id}")
            return perceive_episode(ep, opts, report)

        ids.append(jobs.submit("perceive", _job, episode=ep.id))

        if want_cloud:
            def _cloud_job(report, log, ep=ep, stride=cloud_stride,
                           bbox=cloud_bbox, voxel=cloud_voxel):
                from viki.perception.cloud import build_cloud

                log(f"cloud {ep.id} (regenerate, bbox={bbox})")
                return build_cloud(ep, stride=stride, bbox=bbox, voxel=voxel, report=report)

            ids.append(jobs.submit("cloud", _cloud_job, episode=ep.id, lane="cloud"))
    return {"job_ids": ids}


@_ep.get("/models")
async def list_models():
    from viki.perception.backends.registry import list_models as _lm

    return _lm()


@_ep.post("/models/download")
async def download_model(req: _ModelReq):
    logger.info("model download requested: %s", req.model)

    def _job(report, log):
        from viki.perception.backends.registry import download

        return download(req.model, report, log)

    return {"job_id": jobs.submit("download", _job, episode=req.model)}


@_ep.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(409, "job already running or finished")
    return {"status": "cancelled"}


@_ep.post("/extract")
async def extract(req: _EpReq):
    ep = _episode(req.episode)
    logger.info("extract: episode=%s model=%s", ep.id, req.model)

    def _job():
        from viki.perception.extract import extract_episode

        return extract_episode(ep, model=req.model)

    return {"job_id": jobs.submit("extract", _job, episode=ep.id)}


@_ep.post("/cloud")
async def cloud(req: _EpReq):
    ep = _episode(req.episode)
    logger.info("cloud: episode=%s", ep.id)

    def _job(report, log):
        from viki.perception.cloud import build_cloud

        return build_cloud(ep, report=report)

    return {"job_id": jobs.submit("cloud", _job, episode=ep.id, lane="cloud")}


@_ep.post("/prepare")
async def prepare(req: _EpReq):
    ep = _episode(req.episode)
    logger.info("prepare: episode=%s sg=%s/%s", ep.id, req.window, req.polyorder)

    def _job():
        from viki.prepare.run import prepare_episode

        return prepare_episode(ep, req.window, req.polyorder)

    return {"job_id": jobs.submit("prepare", _job, episode=ep.id)}


@_ep.post("/retarget")
async def retarget(req: _EpReq):
    ep = _episode(req.episode)
    logger.info("retarget: episode=%s robot=%s", ep.id, req.robot)

    def _job():
        from viki.retarget.run import retarget_episode

        return retarget_episode(ep, robot=req.robot)

    return {"job_id": jobs.submit("retarget", _job, episode=ep.id)}


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
async def geometry(ep_id: str, include_raw: int = 0, frame: int | None = None):
    ep = _episode(ep_id)
    out: dict = {"id": ep.id, "cameras": {}, "n_frames": 0}

    meta = json.loads(ep.meta_path.read_text()) if ep.meta_path.exists() else {}
    preset_name = meta.get("calibration_preset")
    if preset_name:
        try:
            from viki.calibration import presets as _presets

            board = _presets.read_detail(preset_name).get("board")
            if board:
                out["board"] = board
        except Exception:  # noqa: BLE001
            pass
    out["workspace_bbox"] = list(getattr(config, "CLOUD_WORKSPACE_BBOX", []) or [])
    out["t_world_display"] = _world_display(ep)

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
            p_key, r_key = cln_pose_keys(
                d.files, getattr(config, "PERCEPTION_HAND_POSE_SOURCE", "landmarks")
            )
            out["n_frames"] = int(len(d["positions"]))
            out["fps"] = float(
                1e6 / max(np.median(np.diff(d["timestamps"].astype(float))), 1.0)
            )
            out["wrist_traj"] = np.asarray(d[p_key], np.float32).tolist()
            out["palm_rot"] = (
                np.asarray(d[r_key], np.float32).reshape(-1, 9).tolist()
            )
            out["valid"] = np.asarray(d["valid"], bool).tolist()
            if "hand_fit_capsule_radii" in d.files:
                out["hand_capsule_radii"] = np.asarray(d["hand_fit_capsule_radii"], np.float32).tolist()

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

    if frame is not None:
        out["frame"] = int(frame)
        grid = None
        if ep.cln_npz.exists():
            with np.load(ep.cln_npz) as d:
                grid = np.asarray(d["timestamps"], np.float64)
                n = len(d["smoothed_points"])
                if 0 <= frame < n:
                    out["fused_skeleton"] = _nan_rows(d["smoothed_points"][frame])
                    out["landmark_ids"] = np.asarray(d["landmark_ids"], int).tolist()
                    out["gripper"] = bool(d["gripper"][frame])
                    out["frame_valid"] = bool(d["valid"][frame])
                    if "hand_fit_capsules" in d.files and frame < len(d["hand_fit_capsules"]):
                        hc = np.asarray(d["hand_fit_capsules"][frame], np.float32)  # (C, 2, 3)
                        out["hand_capsules"] = (
                            None if not np.isfinite(hc).any() else hc.tolist()
                        )
        if ep.rec_npz.exists():
            with np.load(ep.rec_npz) as d:
                devs = np.array([str(x) for x in d["device_ids"]])
                ts = np.asarray(d["timestamps"], np.float64)
                pts = np.asarray(d["points"], np.float32)
                conf = np.asarray(d["confidence"], np.float32)
                lm_ids = np.asarray(d["landmark_ids"], int).tolist()
                t_us = float(grid[frame]) if grid is not None and frame < len(grid) else None
                per_cam: dict[str, dict] = {}
                for dev in sorted(set(devs.tolist())):
                    idx = np.where(devs == dev)[0]
                    if idx.size == 0:
                        continue
                    row = (idx[int(np.argmin(np.abs(ts[idx] - t_us)))] if t_us is not None
                           else idx[min(frame, idx.size - 1)])
                    per_cam[dev] = {
                        "points": _nan_rows(pts[row]),
                        "confidence": [None if not np.isfinite(c) else float(c)
                                       for c in conf[row]],
                        "landmark_ids": lm_ids,
                    }
                out["per_camera"] = per_cam

    return out


def _nan_rows(arr) -> list:
    """(K,3) array → nested list with non-finite values as ``None`` (JSON-safe)."""
    a = np.asarray(arr, float)
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a]


def _world_display(ep) -> list:
    """The episode's ``T_world_display`` (4x4 nested list) from
    ``raw/world_anchor.json``, or identity. Data in the geometry response and
    the cloud are in the RIG (reference-camera) frame; the viewer applies this
    for presentation only."""
    p = ep.raw_dir / "world_anchor.json"
    if p.exists():
        try:
            m = json.loads(p.read_text()).get("T_world_display")
            if m and len(m) == 4:
                return [[float(v) for v in row] for row in m]
        except (ValueError, OSError, TypeError):
            pass
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


# ── coloured point cloud (Viewer tab) ────────────────────────────────


@_ep.get("/episode/{ep_id}/cloud")
async def cloud_meta(ep_id: str):
    ep = _episode(ep_id)
    p = ep.cloud_dir / "meta.json"
    if not p.exists():
        raise HTTPException(404, "no cloud; run the 'cloud' stage first")
    meta = json.loads(p.read_text())
    meta.setdefault("t_world_display", _world_display(ep))
    return meta


@_ep.get("/episode/{ep_id}/cloud/{frame}")
async def cloud_frame(ep_id: str, frame: int):
    ep = _episode(ep_id)
    p = ep.cloud_dir / f"{frame:06d}.bin"
    if not p.exists():
        raise HTTPException(404, f"no cloud frame {frame}")
    return FileResponse(p, media_type="application/octet-stream")


router.include_router(_ep)
