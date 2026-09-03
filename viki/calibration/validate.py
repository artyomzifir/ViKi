"""
viki.calibration.validate
-------------------------
Pre-record cloud-agreement gate (spec §6).

After calibration, build one point cloud **per camera** from a live empty scene,
in the rig (reference-camera) frame, and check that the cameras actually agree:
nearest-neighbour distance between each pair, plus an ICP whose *found* rigid
correction should be near zero — a large ICP translation with a small residual
is the exact signature of two mis-registered extrinsics.

Pure over the assembled clouds so the verdict logic is testable without cameras.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# spec §6 defaults (mm / deg). Overridable by the caller from config.
GREEN = {"nn_median_mm": 15.0, "icp_translation_mm": 20.0, "icp_rotation_deg": 2.0}
AMBER = {"nn_median_mm": 30.0, "icp_translation_mm": 50.0, "icp_rotation_deg": 5.0}
_MIN_PAIR_POINTS = 200
_SENSOR_NOISE_MM = 8.0  # below this an NN median is more likely a metric bug


def _crop(xyz: np.ndarray, aabb) -> np.ndarray:
    if not aabb or len(aabb) != 6:
        return xyz
    x0, x1, y0, y1, z0, z1 = aabb
    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0)
         & (xyz[:, 1] <= y1) & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1))
    return xyz[m]


def _icp(src: np.ndarray, dst: np.ndarray, iters: int = 30):
    """Trimmed point-to-point ICP src→dst. Returns ``(R, t, residual_median_m)``
    for the *total* correction found (identity = the clouds already agree)."""
    tree = cKDTree(dst)
    R_tot = np.eye(3)
    t_tot = np.zeros(3)
    cur = src
    resid = np.inf
    for _ in range(iters):
        d, idx = tree.query(cur, k=1)
        keep = d <= np.percentile(d, 70)
        P, Q = cur[keep], dst[idx[keep]]
        cP, cQ = P.mean(0), Q.mean(0)
        H = (P - cP).T @ (Q - cQ)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cQ - R @ cP
        cur = cur @ R.T + t
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        new_resid = float(np.median(np.linalg.norm(cur - dst[tree.query(cur, k=1)[1]], axis=1)))
        if abs(resid - new_resid) < 1e-5:
            resid = new_resid
            break
        resid = new_resid
    return R_tot, t_tot, resid


def _geodesic_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _pair_verdict(p: dict, green: dict, amber: dict) -> str:
    def within(lim):
        return (p["nn_median_mm"] <= lim["nn_median_mm"]
                and p["icp_translation_mm"] <= lim["icp_translation_mm"]
                and p["icp_rotation_deg"] <= lim["icp_rotation_deg"])
    if within(green):
        return "green"
    if within(amber):
        return "amber"
    return "red"


def pairwise_agreement(
    clouds: dict[str, np.ndarray],
    *,
    aabb=None,
    green: dict | None = None,
    amber: dict | None = None,
) -> dict:
    """``clouds[dev]`` is an (N,3) rig-frame point cloud. Returns
    ``{"verdict", "pairs": [...]}`` matching ``validation_report.json``."""
    green = {**GREEN, **(green or {})}
    amber = {**AMBER, **(amber or {})}
    devs = sorted(clouds)
    cropped = {d: _crop(np.asarray(clouds[d], float).reshape(-1, 3), aabb) for d in devs}

    pairs: list[dict] = []
    order = {"green": 0, "amber": 1, "red": 2}
    worst = "green" if len(devs) >= 2 else "red"

    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            a, b = cropped[devs[i]], cropped[devs[j]]
            n = min(len(a), len(b))
            if n < _MIN_PAIR_POINTS:
                pairs.append({
                    "a": devs[i], "b": devs[j], "n_points": int(n),
                    "skipped": True, "reason": "too few points in the workspace box",
                })
                worst = "red"
                continue
            d_ab = cKDTree(b).query(a, k=1)[0]
            d_ba = cKDTree(a).query(b, k=1)[0]
            nn_median = float(max(np.median(d_ab), np.median(d_ba)))
            nn_p90 = float(max(np.percentile(d_ab, 90), np.percentile(d_ba, 90)))
            R, t, resid = _icp(a, b)
            p = {
                "a": devs[i], "b": devs[j], "n_points": int(n),
                "nn_median_mm": round(nn_median * 1e3, 2),
                "nn_p90_mm": round(nn_p90 * 1e3, 2),
                "icp_translation_mm": round(float(np.linalg.norm(t)) * 1e3, 2),
                "icp_rotation_deg": round(_geodesic_deg(R), 3),
                "icp_residual_mm": round(resid * 1e3, 2),
            }
            p["verdict"] = _pair_verdict(p, green, amber)
            if p["nn_median_mm"] < _SENSOR_NOISE_MM:
                p["note"] = ("NN median below sensor noise — likely a metric bug, "
                             "not a perfect calibration")
            pairs.append(p)
            if order[p["verdict"]] > order[worst]:
                worst = p["verdict"]

    return {"verdict": worst, "pairs": pairs}
