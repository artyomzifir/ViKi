"""
viki.calibration.validate
-------------------------
Pre-record cloud-agreement gate (spec §6).

After calibration, build one point cloud **per camera** from a live empty scene,
in the rig (reference-camera) frame, and check that the cameras actually agree:
nearest-neighbour distance between each pair, plus an ICP whose *found* rigid
correction should be near zero.

**Only what both cameras actually observed is scored.** Both metrics compare a
point to the nearest point of the other cloud, so a surface only one camera has
is scored as disagreement no matter how good the extrinsics are. On a
wide-baseline pair that dominates: on the 0.60 m / 83° RealSense+Kinect rig,
half the RealSense cloud is outside the Kinect's view and the one-way NN medians
come out 42.8 mm vs 8.7 mm on a calibration two independent methods agree on to
0.32 mm.

Visibility is therefore tested in two steps, because the frustum alone is not
enough. A point can sit squarely inside the other camera's field of view and
still be something it never measured — a stereo dropout over low texture, a
shadow, a surface out of range. Measured on that rig: blank a contiguous region
of the RealSense depth image and 22.6 % of the Kinect's in-frustum points land
on pixels with no reading at all, carrying an NN median of **41.2 mm** against
**6.7 mm** for the points the RealSense genuinely saw. So the mask is frustum
*plus* a lookup in the other camera's depth image: it must have a reading there,
and that reading must not be well in front of the point (occlusion).

When the surviving overlap is too small to mean anything the pair is reported
``unknown``/``skipped``, never ``red``: not scorable is not the same as wrong,
and the recording gate treats them differently.

A caveat the mask does not remove: an empty scene is close to a single plane,
and sliding along a plane or spinning about its normal are ICP's unconstrained
directions. Of the 217 mm correction that pair scored unmasked, 215.8 mm lay
*in* the plane and the 12.5° rotation was about an axis 2.4° off its normal. So
a large ICP correction is evidence of mis-registration only in the component
that leaves the shared surface.

Pure over the assembled clouds (plus plain camera-model dicts) so the verdict
logic is testable without cameras.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# spec §6 defaults (mm / deg). Overridable by the caller from config.
GREEN = {"nn_median_mm": 15.0, "icp_translation_mm": 20.0, "icp_rotation_deg": 2.0}
AMBER = {"nn_median_mm": 30.0, "icp_translation_mm": 50.0, "icp_rotation_deg": 5.0}
_MIN_PAIR_POINTS = 200
_MIN_OVERLAP_FRAC = 0.15  # below this a pair is unscorable, not wrong
_OCCLUSION_TOL_MM = 50.0  # the other camera's reading may sit this far in front


def _crop(xyz: np.ndarray, aabb) -> np.ndarray:
    if not aabb or len(aabb) != 6:
        return xyz
    x0, x1, y0, y1, z0, z1 = aabb
    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0)
         & (xyz[:, 1] <= y1) & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1))
    return xyz[m]


def _visible_mask(xyz: np.ndarray, cam: dict) -> np.ndarray:
    """Which rig-frame points camera ``cam`` actually observed.

    ``cam`` describes the **depth** camera — that is the sensor a cloud comes
    from, and its field of view is not the colour camera's::

        {"T_ref_cam": 4x4, "K": 3x3, "width": int, "height": int,
         "depth_mm": (H, W) array | None}

    Without ``depth_mm`` this is the frustum alone; lens distortion is ignored
    on purpose, since a pinhole frustum errs on the side of keeping points.
    With it, the point must also land on a pixel that carries a reading, and
    that reading must not sit more than ``_OCCLUSION_TOL_MM`` in front of the
    point — a camera pointed at something it never measured has not seen it.
    A reading *behind* the point is left in: that is real disagreement, not
    absence.
    """
    return _visibility(xyz, cam)[0]


def _visibility(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, dict]:
    """:func:`_visible_mask` plus a count of why each point was dropped.

    The counts go into the report: when a pair comes back red, the one number
    that matters is *which* exclusion the disagreeing points escaped, and
    guessing at that from a median costs a rig session per guess.
    """
    T = np.asarray(cam["T_ref_cam"], float)
    p = (np.asarray(xyz, float) - T[:3, 3]) @ T[:3, :3]  # rig → camera
    z = p[:, 2]
    infront = z > 1e-6
    K = np.asarray(cam["K"], float)
    zs = np.where(infront, z, 1.0)
    u = p[:, 0] / zs * K[0, 0] + K[0, 2]
    v = p[:, 1] / zs * K[1, 1] + K[1, 2]
    inside = (infront & (u >= 0) & (u <= float(cam["width"]) - 1)
              & (v >= 0) & (v <= float(cam["height"]) - 1))
    why = {"outside_view": int((~inside).sum())}

    depth = cam.get("depth_mm")
    if depth is None:
        why.update(no_reading=None, occluded=None)
        return inside, why
    dm = np.asarray(depth)
    h, w = dm.shape[:2]
    ui = np.clip(np.rint(u), 0, w - 1).astype(np.intp)
    vi = np.clip(np.rint(v), 0, h - 1).astype(np.intp)
    seen_m = dm[vi, ui].astype(np.float64) / 1000.0
    has_reading = np.isfinite(seen_m) & (seen_m > 0)
    not_occluded = seen_m >= z - _OCCLUSION_TOL_MM / 1000.0
    why["no_reading"] = int((inside & ~has_reading).sum())
    why["occluded"] = int((inside & has_reading & ~not_occluded).sum())
    return inside & has_reading & not_occluded, why


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
    frusta: dict[str, dict] | None = None,
    aabb=None,
    green: dict | None = None,
    amber: dict | None = None,
    min_overlap: float | None = None,
) -> dict:
    """``clouds[dev]`` is an (N,3) rig-frame point cloud. Returns
    ``{"verdict", "pairs": [...]}`` matching ``validation_report.json``.

    ``frusta[dev]`` is that camera's depth visibility model (see
    :func:`_visible_mask`); when both cameras of a pair have one, each cloud is
    masked to what the *other* camera actually observed before scoring, and the
    pair carries a ``shared_view`` block. Without them the pair is scored whole
    — the old behaviour, which charges each camera for what the other never saw.

    A pair is ``skipped`` with verdict ``unknown`` when it cannot be scored:
    too few points, or a masked overlap below ``min_overlap`` of either cloud
    (default :data:`_MIN_OVERLAP_FRAC`).
    """
    green = {**GREEN, **(green or {})}
    amber = {**AMBER, **(amber or {})}
    frusta = frusta or {}
    min_overlap = _MIN_OVERLAP_FRAC if min_overlap is None else float(min_overlap)
    devs = sorted(clouds)
    cropped = {d: _crop(np.asarray(clouds[d], float).reshape(-1, 3), aabb) for d in devs}

    pairs: list[dict] = []
    order = {"green": 0, "amber": 1, "unknown": 2, "red": 3}
    worst = "green" if len(devs) >= 2 else "unknown"

    def _worse(v: str) -> None:
        nonlocal worst
        if order[v] > order[worst]:
            worst = v

    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            da, db = devs[i], devs[j]
            a_all, b_all = cropped[da], cropped[db]
            fa, fb = frusta.get(da), frusta.get(db)
            shared = None
            if fa is not None and fb is not None:
                mask_a, why_a = _visibility(a_all, fb)
                mask_b, why_b = _visibility(b_all, fa)
                a, b = a_all[mask_a], b_all[mask_b]
                frac_a = len(a) / len(a_all) if len(a_all) else 0.0
                frac_b = len(b) / len(b_all) if len(b_all) else 0.0
                shared = {
                    "a_points": int(len(a)), "b_points": int(len(b)),
                    "a_frac": round(frac_a, 3), "b_frac": round(frac_b, 3),
                    "a_dropped": why_a, "b_dropped": why_b,
                }
            else:
                a, b = a_all, b_all
                frac_a = frac_b = 1.0

            n = min(len(a), len(b))
            reason = None
            if n < _MIN_PAIR_POINTS:
                reason = ("too few points in the cameras' shared field of view"
                          if shared else "too few points in the workspace box")
            elif shared and min(frac_a, frac_b) < min_overlap:
                reason = (
                    f"the cameras share too little of the workspace "
                    f"({100 * min(frac_a, frac_b):.0f}% of a cloud, "
                    f"{100 * min_overlap:.0f}% needed) — nothing to compare on"
                )
            if reason:
                row = {
                    "a": da, "b": db, "n_points": int(n), "skipped": True,
                    "reason": reason, "verdict": "unknown",
                }
                if shared:
                    row["shared_view"] = shared
                pairs.append(row)
                _worse("unknown")
                continue

            d_ab = cKDTree(b).query(a, k=1)[0]
            d_ba = cKDTree(a).query(b, k=1)[0]
            nn_median = float(max(np.median(d_ab), np.median(d_ba)))
            nn_p90 = float(max(np.percentile(d_ab, 90), np.percentile(d_ba, 90)))
            R, t, resid = _icp(a, b)
            p = {
                "a": da, "b": db, "n_points": int(n),
                "nn_median_mm": round(nn_median * 1e3, 2),
                # both directions, because the verdict takes the max and the
                # two differ by 5x when one camera is missing a surface — the
                # max alone never says which camera to look at.
                "nn_median_ab_mm": round(float(np.median(d_ab)) * 1e3, 2),
                "nn_median_ba_mm": round(float(np.median(d_ba)) * 1e3, 2),
                "nn_p90_mm": round(nn_p90 * 1e3, 2),
                "icp_translation_mm": round(float(np.linalg.norm(t)) * 1e3, 2),
                "icp_rotation_deg": round(_geodesic_deg(R), 3),
                "icp_residual_mm": round(resid * 1e3, 2),
            }
            if shared:
                p["shared_view"] = shared
            p["verdict"] = _pair_verdict(p, green, amber)
            pairs.append(p)
            _worse(p["verdict"])

    return {"verdict": worst, "pairs": pairs}
