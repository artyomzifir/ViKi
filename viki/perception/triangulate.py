"""
viki.perception.triangulate
---------------------------
Stage 2 of the multi-view triangulation task: recover each hand joint in the
world (rig) frame from the 2-D observations of two or more cameras, using depth
as a **soft** extra residual — never as the single anchor depth the mono lift
uses.

Pipeline per joint, per frame:

1. keep views whose per-landmark score clears ``TRI_MIN_SCORE``;
2. enumerate every camera pair (2-3 cams ⇒ no RANSAC); drop a pair whose rays to
   the candidate subtend less than ``TRI_MIN_RAY_DEG`` (ill-conditioned);
3. linear DLT on the pair + cheirality (in front of both cameras);
4. score the hypothesis against all kept views — inlier count by reprojection
   error, summed score, median reprojection error, ray angle;
5. take the best hypothesis, then refine non-linearly over its inliers:

       min_X  Σ_c w_c ρ(‖π_c(X) − u_c‖²)  +  λ_d Σ_c w^d_c ρ_d((z_c(X) − d_c − δ)²)

   (``scipy.optimize.least_squares``, ``loss`` from ``TRI_LOSS``).

**Distortion discipline:** the 2-D observations are undistorted **once**
(`cv2.undistortPoints(..., P=K)`) before the DLT, and everything downstream is a
plain pinhole projection. Distortion is never re-applied inside the residual.

**Depth as soft evidence:** residual ``z_c(X) − d_c − δ`` where ``δ`` is the
skin→joint-centre offset (depth sees the skin, the joint sits inside the
finger), per joint type. Depth weight is knocked down by local depth spread and
dropped entirely for invalid samples — one bad silhouette pixel must not move a
good multi-view point.

**``quality`` is a transparent score, not an inverse covariance.** It is
``inlier_fraction · reproj_term · ray_angle_term`` and is *not* calibrated
against ground truth; do not treat it as a variance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from viki import config
from viki.contracts import HAND_LM_COUNT

logger = logging.getLogger(__name__)

# skin → joint-centre offset (m), by MediaPipe landmark index. Fingertips sit
# closest to the surface; knuckles / wrist have more tissue over the joint.
_FINGERTIPS = (4, 8, 12, 16, 20)
_KNUCKLES = (1, 2, 5, 6, 9, 10, 13, 14, 17, 18)


def _delta_for(lm: int, base: float) -> float:
    if lm in _FINGERTIPS:
        return base
    if lm in _KNUCKLES:
        return base * 1.6
    if lm == 0:  # wrist — depth landmark is unreliable anyway
        return base * 2.0
    return base * 1.3


class _Cam:
    __slots__ = ("id", "K", "dist", "T_wc", "P", "C", "size")

    def __init__(self, cid: str, meta: dict):
        self.id = cid
        self.K = np.asarray(meta["K"], float)
        self.dist = np.asarray(meta.get("dist", [0] * 5), float).reshape(-1)
        self.T_wc = np.asarray(meta["T_wc"], float)          # world ← camera
        T_cw = np.linalg.inv(self.T_wc)                       # world → camera
        self.P = self.K @ T_cw[:3, :4]                        # 3×4, world → pixel
        self.C = self.T_wc[:3, 3]                             # camera centre (world)
        self.size = tuple(meta.get("image_size", (0, 0)))

    def undistort(self, uv) -> np.ndarray:
        p = np.asarray(uv, float).reshape(1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist, P=self.K).reshape(2)

    def project(self, X) -> tuple[np.ndarray, float]:
        x = self.P @ np.append(np.asarray(X, float), 1.0)
        return x[:2] / x[2], float(x[2])       # pixel, camera-frame depth


def _dlt(Pa, ua, Pb, ub) -> np.ndarray:
    A = np.stack([
        ua[0] * Pa[2] - Pa[0], ua[1] * Pa[2] - Pa[1],
        ub[0] * Pb[2] - Pb[0], ub[1] * Pb[2] - Pb[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _ray_deg(Ca, Cb, X) -> float:
    da = X - Ca
    db = X - Cb
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(da, db) / (na * nb), -1, 1))))


class TriConfig:
    def __init__(self, overrides: dict[str, object] | None = None):
        """Triangulation knobs from config, or an explicit named-profile map."""
        values = dict(overrides or {})

        def number(attr: str, config_key: str, default: float) -> float:
            return float(values.get(attr, getattr(config, config_key, default)))

        self.min_score = number("min_score", "TRI_MIN_SCORE", 0.30)
        self.min_ray_deg = number("min_ray_deg", "TRI_MIN_RAY_DEG", 5.0)
        self.reproj_inlier_px = number(
            "reproj_inlier_px", "TRI_REPROJ_INLIER_PX", 4.0,
        )
        self.depth_lambda = number("depth_lambda", "TRI_DEPTH_LAMBDA", 0.10)
        self.depth_delta_m = number("depth_delta_m", "TRI_DEPTH_DELTA_M", 0.010)
        self.depth_spread_scale_m = number(
            "depth_spread_scale_m", "TRI_DEPTH_SPREAD_SCALE_M", 0.020,
        )
        self.loss = str(values.get("loss", getattr(config, "TRI_LOSS", "soft_l1")))
        self.ray_ref_deg = number("ray_ref_deg", "TRI_RAY_REF_DEG", 20.0)
        cams = values.get(
            "geometry_cameras", getattr(config, "TRI_GEOMETRY_CAMERAS", []),
        ) or []
        self.geometry_cameras = list(cams)

    def as_dict(self) -> dict[str, object]:
        return {
            "min_score": self.min_score,
            "min_ray_deg": self.min_ray_deg,
            "reproj_inlier_px": self.reproj_inlier_px,
            "depth_lambda": self.depth_lambda,
            "depth_delta_m": self.depth_delta_m,
            "depth_spread_scale_m": self.depth_spread_scale_m,
            "loss": self.loss,
            "ray_ref_deg": self.ray_ref_deg,
            "geometry_cameras": self.geometry_cameras,
        }


def triangulate_joint(views: list[dict], cams: dict[str, _Cam], lm: int, cfg: TriConfig):
    """``views`` = ``[{camera_id, uv, score, depth_m, depth_valid, depth_spread_m}]``
    for ONE joint in ONE frame. Returns a dict or ``None`` (a gap)."""
    usable = [
        v for v in views
        if v["camera_id"] in cams and v["score"] >= cfg.min_score
        and np.isfinite(v["uv"]).all()
    ]
    if len(usable) < 2:
        return None
    for v in usable:
        v["_uvu"] = cams[v["camera_id"]].undistort(v["uv"])

    best = None  # (key tuple, X, inlier_ids)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            ca, cb = cams[a["camera_id"]], cams[b["camera_id"]]
            X = _dlt(ca.P, a["_uvu"], cb.P, b["_uvu"])
            _, za = ca.project(X)
            _, zb = cb.project(X)
            if za <= 0 or zb <= 0:
                continue
            if _ray_deg(ca.C, cb.C, X) < cfg.min_ray_deg:
                continue
            errs, inl = [], []
            for v in usable:
                uvp, z = cams[v["camera_id"]].project(X)
                e = float(np.linalg.norm(uvp - v["_uvu"]))
                errs.append(e)
                if z > 0 and e <= cfg.reproj_inlier_px:
                    inl.append(v)
            key = (len(inl), sum(v["score"] for v in inl), -float(np.median(errs)))
            if best is None or key > best[0]:
                best = (key, X, inl)

    if best is None or len(best[2]) < 2:
        return None
    X0, inliers = best[1], best[2]

    # ── non-linear refine over inliers ──
    def resid(X):
        r = []
        for v in inliers:
            c = cams[v["camera_id"]]
            uvp, z = c.project(X)
            w = max(v["score"], 1e-3) ** 0.5
            r += [w * (uvp[0] - v["_uvu"][0]), w * (uvp[1] - v["_uvu"][1])]
            if v["depth_valid"] and np.isfinite(v["depth_m"]):
                spread = v["depth_spread_m"] if np.isfinite(v["depth_spread_m"]) else 0.0
                wd = v["score"] * float(np.exp(-spread / cfg.depth_spread_scale_m))
                dscale = (cfg.depth_lambda * max(wd, 1e-4)) ** 0.5
                # express the depth error as an equivalent pixel disparity
                # (f · Δz / z) so it shares units and f_scale with reprojection
                f = 0.5 * (c.K[0, 0] + c.K[1, 1])
                dz = z - v["depth_m"] - _delta_for(lm, cfg.depth_delta_m)
                r.append(dscale * f * dz / max(z, 1e-3))
        return np.asarray(r, float)

    sol = least_squares(resid, X0, loss=cfg.loss, f_scale=cfg.reproj_inlier_px, max_nfev=60)
    X = sol.x

    errs = [float(np.linalg.norm(cams[v["camera_id"]].project(X)[0] - v["_uvu"])) for v in inliers]
    ray = max(
        (_ray_deg(cams[inliers[a]["camera_id"]].C, cams[inliers[b]["camera_id"]].C, X)
         for a in range(len(inliers)) for b in range(a + 1, len(inliers))),
        default=0.0,
    )
    mean_err = float(np.mean(errs))
    quality = (
        (len(inliers) / max(len(usable), 1))
        * float(np.clip(1.0 - mean_err / (2 * cfg.reproj_inlier_px), 0.0, 1.0))
        * float(np.clip(ray / cfg.ray_ref_deg, 0.0, 1.0))
    )
    return {
        "xyz": X.astype(np.float32),
        "quality": float(quality),
        "n_views": int(len(inliers)),
        "n_candidate_views": int(len(usable)),
        "ray_deg": float(ray),
        "reproj_px": mean_err,
    }


# ── episode driver ──────────────────────────────────────────────────────


def triangulate_episode(
    raw: Path, *, write: bool = True, cfg: TriConfig | None = None,
) -> dict:
    """Read ``raw/observations.npz`` (Stage 1) and write ``raw/joints3d.npz`` —
    per synced frame, the world-frame joints that triangulated + their quality.
    Frames/joints without ≥2 usable views are gaps (NaN, quality 0)."""
    from viki.perception.observations import read_observations

    obs = read_observations(raw)
    if obs is None or not len(obs.get("camera_id", [])):
        raise FileNotFoundError(f"no observations.npz in {raw}")
    cfg = cfg or TriConfig()
    meta_cams = obs["cameras"]
    want = set(cfg.geometry_cameras) if cfg.geometry_cameras else set(meta_cams)
    cams = {c: _Cam(c, meta_cams[c]) for c in meta_cams if c in want and meta_cams[c].get("K")}
    if len(cams) < 2:
        raise ValueError(f"need ≥2 geometry cameras with intrinsics, have {sorted(cams)}")

    cam_ids = obs["camera_id"].astype(str)
    frame_idx = obs["frame_index"].astype(int)
    uv = obs["uv"]
    score = obs["lm_score"]
    depth = obs["depth_m"]
    dvalid = obs["depth_valid"]
    dspread = obs["depth_spread_m"]

    # Group by SYNCED-FRAME INDEX, not host timestamp: the recorder is
    # index-aligned across cameras (mp4 frame i of every camera == one synced
    # group), whereas host_timestamp_us carries each camera's per-frame offset
    # from the tick, so two cameras' rows for the same group never share it.
    frames = sorted(set(frame_idx.tolist()))
    frame_rows: dict[int, list[int]] = {f: [] for f in frames}
    for r, f in enumerate(frame_idx.tolist()):
        frame_rows[f].append(r)
    T = len(frames)

    # canonical timestamp per synced group = the sync tick from raw/timestamps.json
    sync_us: list[int] = []
    try:
        import json as _json

        _ts = _json.loads((raw / "timestamps.json").read_text())
        _sync = [int(e["sync_us"]) for e in _ts if "sync_us" in e]
        sync_us = [_sync[f] if 0 <= f < len(_sync) else int(f) for f in frames]
    except Exception:  # noqa: BLE001
        sync_us = list(frames)
    out_xyz = np.full((T, HAND_LM_COUNT, 3), np.nan, np.float32)
    out_q = np.zeros((T, HAND_LM_COUNT), np.float32)
    out_nv = np.zeros((T, HAND_LM_COUNT), np.int8)
    out_ray = np.zeros((T, HAND_LM_COUNT), np.float32)
    out_reproj = np.full((T, HAND_LM_COUNT), np.nan, np.float32)

    nviews_hist = np.zeros(4, int)  # joints solved with 0/1/2/3+ views
    for ti, f in enumerate(frames):
        rows = [r for r in frame_rows[f] if cam_ids[r] in cams]
        for lm in range(HAND_LM_COUNT):
            views = [{
                "camera_id": cam_ids[r], "uv": uv[r, lm], "score": float(score[r, lm]),
                "depth_m": float(depth[r, lm]), "depth_valid": bool(dvalid[r, lm]),
                "depth_spread_m": float(dspread[r, lm]),
            } for r in rows]
            res = triangulate_joint(views, cams, lm, cfg)
            if res is None:
                nviews_hist[min(3, sum(1 for v in views
                                       if v["score"] >= cfg.min_score and np.isfinite(v["uv"]).all()))] += 1
                continue
            out_xyz[ti, lm] = res["xyz"]
            out_q[ti, lm] = res["quality"]
            out_nv[ti, lm] = res["n_views"]
            out_ray[ti, lm] = res["ray_deg"]
            out_reproj[ti, lm] = res["reproj_px"]
            nviews_hist[min(3, res["n_views"])] += 1

    summary = {
        "n_frames": T,
        "cameras": sorted(cams),
        "joints_total": int(T * HAND_LM_COUNT),
        "joints_solved": int(np.isfinite(out_xyz).all(axis=2).sum()),
        "nviews_hist_0_1_2_3plus": nviews_hist.tolist(),
        "reproj_px_median": float(np.nanmedian(out_reproj)) if np.isfinite(out_reproj).any() else None,
        "reproj_px_p95": float(np.nanpercentile(out_reproj, 95)) if np.isfinite(out_reproj).any() else None,
        "quality_median": float(np.median(out_q[out_q > 0])) if (out_q > 0).any() else 0.0,
        "config": cfg.as_dict(),
    }
    if write:
        np.savez(
            raw / "joints3d.npz", schema=np.int32(1),
            timestamps=np.asarray(sync_us, np.int64),
            xyz=out_xyz, quality=out_q, n_views=out_nv,
            ray_deg=out_ray, reproj_px=out_reproj,
            cameras=np.array(sorted(cams)),
        )
        import json as _json

        (raw / "joints3d_summary.json").write_text(_json.dumps(summary, indent=2))
        logger.info("triangulate %s: %d/%d joints solved, reproj med %.2f px",
                    raw.parent.name, summary["joints_solved"], summary["joints_total"],
                    summary["reproj_px_median"] or -1)
    return summary
