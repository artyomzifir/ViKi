"""
viki.calibration.bundle
-----------------------
Multi-pose extrinsics: one joint solve over every camera pose **and** every
board pose, instead of a per-camera PnP composed through the board.

Why joint: with a single static board pose the per-camera problem is
degenerate in the board's out-of-plane tilt (the true pose and a mirrored one
give almost the same reprojection). Composing two such per-camera solves leaves
the two camera frames disagreeing by cm / a degree. When the board is seen in
several distinct poses, each board pose becomes a shared parameter that *every*
camera that sees it must explain at once — that couples the cameras and breaks
the tilt ambiguity.

Parameters (all as Rodrigues ``rvec`` + ``tvec``):
  * ``T_cam_ref[i]`` for each non-reference camera  — ref-frame point → camera i
  * ``T_ref_board[k]`` for each capture set          — board point → ref frame
The reference camera is the identity (fixes the gauge).

Residual: reprojection of every detected ChArUco corner in every camera that
sees it, Huber-robustified. Sparse Jacobian via ``jac_sparsity`` (a corner in
set *k* / camera *i* touches only those two parameter blocks), ``tr_solver='lsmr'``.

A ``cv2.stereoCalibrate`` on the same co-visible sets is run as an independent
check; a disagreement over 5 mm / 0.5° is logged as a warning, not silently
resolved.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from viki.calibration.geometry import robust_planar_pnp
from viki.calibration.samples import _charuco_board, _K

logger = logging.getLogger(__name__)

_HUBER_PX = 2.0
_STEREO_TRANS_WARN_MM = 5.0
_STEREO_ROT_WARN_DEG = 0.5
_DEGENERATE_TILT_DEG = 8.0  # board-pose rotations all within this ⇒ degenerate


# ── small SE(3) helpers (Rodrigues 6-vectors) ───────────────────────────


def _rt_to_T(rt: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rt[:3], float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rt[3:]
    return T


def _T_to_rt(T: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.asarray(T[:3, :3], float))
    return np.concatenate([r.reshape(3), np.asarray(T[:3, 3], float)])


def _apply(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ T[:3, :3].T + T[:3, 3]


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _dist(intr: dict) -> np.ndarray:
    return np.asarray(intr.get("dist_coeffs", [0, 0, 0, 0, 0]), float).reshape(-1)


# ── solve ──────────────────────────────────────────────────────────────


def solve_bundle(
    sets: list[dict],
    intrinsics: dict[str, dict],
    board_cfg: dict,
    *,
    reference_device: str | None = None,
) -> dict:
    """``sets`` is a list of ``{observations: {dev: {charuco_ids, charuco_corners}}}``.
    Returns ``{"reference_device", "devices": {dev: T_ref_cam 4x4 list},
    "solve": {...}}`` where ``T_ref_cam`` is camera → reference frame (the
    inverse of the ``T_cam_ref`` parameter), matching the ``extrinsics.json``
    schema.
    """
    if board_cfg.get("type") != "aruco":
        raise ValueError("bundle solve only supports ChArUco boards")
    board = _charuco_board(board_cfg)
    obj_all = np.asarray(board.getChessboardCorners(), np.float64)  # (N, 3)

    # observed corners per (set, dev)
    obs: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    devs_seen: set[str] = set()
    for s in sets:
        row: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for dev, o in (s.get("observations") or {}).items():
            ids = np.asarray(o.get("charuco_ids", []), int).reshape(-1)
            uv = np.asarray(o.get("charuco_corners", []), float).reshape(-1, 2)
            if ids.size < 4 or ids.size != len(uv) or int(ids.max(initial=-1)) >= len(obj_all):
                continue
            if dev not in intrinsics:
                continue
            row[dev] = (ids, uv)
            devs_seen.add(dev)
        if row:
            obs.append(row)
    if not obs:
        raise ValueError("no usable ChArUco observations")

    all_devs = sorted(devs_seen)
    ref = reference_device or _pick_reference(obs, all_devs)
    if ref not in devs_seen:
        raise ValueError(f"reference device {ref!r} has no observations")
    other_devs = [d for d in all_devs if d != ref]

    K = {d: _K(intrinsics[d]) for d in all_devs}
    D = {d: _dist(intrinsics[d]) for d in all_devs}

    # ── per-(set,dev) PnP for initialisation ──
    pnp: dict[tuple[int, str], np.ndarray] = {}  # (k, dev) -> T_cam_board (4x4)
    for k, row in enumerate(obs):
        for dev, (ids, uv) in row.items():
            try:
                rvec, tvec = robust_planar_pnp(obj_all[ids], uv, K[dev], D[dev], tag=f"{dev}/set{k}")
            except RuntimeError:
                continue
            pnp[(k, dev)] = _rt_to_T(np.concatenate([np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)]))

    # ── init camera poses T_cam_ref[i] from the richest co-visible set ──
    cam_rt: dict[str, np.ndarray] = {}
    for dev in other_devs:
        best = None  # (n_common, k)
        for k, row in enumerate(obs):
            if (k, dev) in pnp and (k, ref) in pnp:
                n = len(set(row[dev][0]) & set(row[ref][0]))
                if best is None or n > best[0]:
                    best = (n, k)
        if best is None:
            raise ValueError(
                f"camera {dev!r} never co-observes the board with the reference "
                f"{ref!r} — cannot chain it into the rig"
            )
        k = best[1]
        T_dev_ref = pnp[(k, dev)] @ np.linalg.inv(pnp[(k, ref)])  # ref→dev
        cam_rt[dev] = _T_to_rt(T_dev_ref)

    # ── init board poses T_ref_board[k] ──
    board_rt: list[np.ndarray | None] = [None] * len(obs)
    for k, row in enumerate(obs):
        if (k, ref) in pnp:
            board_rt[k] = _T_to_rt(pnp[(k, ref)])  # ref is identity ⇒ T_ref_board = T_refcam_board
            continue
        for dev in other_devs:
            if (k, dev) in pnp:
                T_ref_board = np.linalg.inv(_rt_to_T(cam_rt[dev])) @ pnp[(k, dev)]
                board_rt[k] = _T_to_rt(T_ref_board)
                break
    keep = [k for k, b in enumerate(board_rt) if b is not None]
    if len(keep) < len(obs):
        logger.warning("bundle: dropped %d set(s) with no placeable board pose", len(obs) - len(keep))
    obs = [obs[k] for k in keep]
    board_rt = [board_rt[k] for k in keep]
    n_sets = len(obs)
    if n_sets == 0:
        raise ValueError("no board poses could be initialised")

    # ── pack / unpack ──
    nb = 6 * len(other_devs)
    x0 = np.concatenate(
        [np.concatenate([cam_rt[d] for d in other_devs]) if other_devs else np.zeros(0),
         np.concatenate(board_rt)]
    )

    def cams_of(x):
        out = {ref: np.eye(4)}
        for j, d in enumerate(other_devs):
            out[d] = _rt_to_T(x[6 * j:6 * j + 6])
        return out

    def boards_of(x):
        return [_rt_to_T(x[nb + 6 * k:nb + 6 * k + 6]) for k in range(n_sets)]

    # residual row index bookkeeping for the sparsity pattern
    blocks: list[tuple[int, str, int]] = []  # (set_k, dev, n_corners)
    for k, row in enumerate(obs):
        for dev, (ids, uv) in row.items():
            blocks.append((k, dev, len(ids)))
    total_rows = 2 * sum(n for _, _, n in blocks)

    def residual(x):
        cams = cams_of(x)
        boards = boards_of(x)
        out = np.empty(total_rows)
        r = 0
        for k, row in enumerate(obs):
            Tb = boards[k]
            for dev, (ids, uv) in row.items():
                Xr = _apply(Tb, obj_all[ids])
                Xc = Xr if dev == ref else _apply(cams[dev], Xr)
                proj, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K[dev], D[dev])
                m = 2 * len(ids)
                out[r:r + m] = (proj.reshape(-1, 2) - uv).ravel()
                r += m
        return out

    # ── sparse Jacobian pattern ──
    js = lil_matrix((total_rows, x0.size), dtype=np.uint8)
    r = 0
    for k, dev, n in blocks:
        m = 2 * n
        cb = nb + 6 * k
        js[r:r + m, cb:cb + 6] = 1
        if dev != ref:
            j = other_devs.index(dev)
            js[r:r + m, 6 * j:6 * j + 6] = 1
        r += m
    js = js.tocsr()

    res = least_squares(
        residual, x0, jac_sparsity=js, tr_solver="lsmr",
        loss="huber", f_scale=_HUBER_PX, method="trf", max_nfev=200, xtol=1e-10,
    )

    cams = cams_of(res.x)
    boards = boards_of(res.x)

    # per-device reprojection RMS on the final estimate
    rms: dict[str, float] = {}
    per_dev_sq: dict[str, list[float]] = {d: [] for d in all_devs}
    for k, row in enumerate(obs):
        for dev, (ids, uv) in row.items():
            Xc = _apply(boards[k], obj_all[ids])
            if dev != ref:
                Xc = _apply(cams[dev], Xc)
            proj, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K[dev], D[dev])
            per_dev_sq[dev].extend(np.sum((proj.reshape(-1, 2) - uv) ** 2, axis=1).tolist())
    for d, sq in per_dev_sq.items():
        rms[d] = float(np.sqrt(np.mean(sq))) if sq else float("nan")

    # T_ref_cam = inverse of the T_cam_ref parameter
    devices = {ref: np.eye(4).tolist()}
    for d in other_devs:
        devices[d] = np.linalg.inv(cams[d]).tolist()

    board_rots = [b[:3, :3] for b in boards]
    degenerate = all(
        _geodesic_deg(board_rots[0], R) < _DEGENERATE_TILT_DEG for R in board_rots[1:]
    ) if len(board_rots) > 1 else True

    stereo = _stereo_check(obs, obj_all, K, D, cams, ref, other_devs, intrinsics)

    solve = {
        "method": "bundle",
        "rms_reproj_px": rms,
        "n_sets": n_sets,
        "n_points": int(sum(n for _, _, n in blocks)),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "degenerate": bool(degenerate),
        "stereo_check": stereo,
    }
    if degenerate:
        logger.warning(
            "bundle: all %d board poses are within %.0f° of each other — the solve "
            "is degenerate in the board tilt; collect sets at varied board angles",
            n_sets, _DEGENERATE_TILT_DEG,
        )
    logger.info("bundle: %d sets, %d cams, reproj RMS %s px%s",
                n_sets, len(all_devs), {d: round(v, 2) for d, v in rms.items()},
                "  [DEGENERATE]" if degenerate else "")
    return {"reference_device": ref, "devices": devices, "solve": solve}


def _pick_reference(obs: list[dict], devs: list[str]) -> str:
    """The device present in the most sets (ties → lexical)."""
    count = {d: 0 for d in devs}
    for row in obs:
        for d in row:
            count[d] += 1
    return max(devs, key=lambda d: (count[d], -devs.index(d)))


def _stereo_check(obs, obj_all, K, D, cams, ref, other_devs, intrinsics) -> dict:
    """Independent ``cv2.stereoCalibrate`` on the ref↔other pair with the most
    co-visible sets; compare to the bundle's relative pose."""
    if not other_devs:
        return {}
    # pick the pair with the most co-visible sets
    best_dev, best_sets = None, []
    for dev in other_devs:
        cov = [row for row in obs
               if dev in row and ref in row
               and len(set(row[dev][0]) & set(row[ref][0])) >= 6]
        if len(cov) > len(best_sets):
            best_dev, best_sets = dev, cov
    if best_dev is None or len(best_sets) < 2:
        return {"ran": False, "reason": "need ≥2 co-visible sets with ≥6 shared corners"}

    objp, ip_ref, ip_dev = [], [], []
    for row in best_sets:
        ids_r, uv_r = row[ref]
        ids_d, uv_d = row[best_dev]
        common = np.intersect1d(ids_r, ids_d)
        objp.append(obj_all[common].astype(np.float32))
        ip_ref.append(uv_r[np.searchsorted(ids_r, common)].astype(np.float32))
        ip_dev.append(uv_d[np.searchsorted(ids_d, common)].astype(np.float32))
    size = (
        int(intrinsics[ref].get("width", 1280)),
        int(intrinsics[ref].get("height", 720)),
    )
    try:
        _, _, _, _, _, R, T, *_ = cv2.stereoCalibrate(
            objp, ip_ref, ip_dev, K[ref], D[ref], K[best_dev], D[best_dev], size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    except cv2.error as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"stereoCalibrate failed: {exc}"}

    T_bundle = cams[best_dev]  # ref → dev
    d_rot = _geodesic_deg(R, T_bundle[:3, :3])
    d_trans_mm = float(np.linalg.norm(T.reshape(3) - T_bundle[:3, 3]) * 1e3)
    disagree = d_trans_mm > _STEREO_TRANS_WARN_MM or d_rot > _STEREO_ROT_WARN_DEG
    if disagree:
        logger.warning(
            "bundle vs stereoCalibrate (%s↔%s) disagree by %.1f mm / %.2f° — "
            "check the residual/Jacobian, not just one result",
            ref, best_dev, d_trans_mm, d_rot,
        )
    return {
        "ran": True, "pair": [ref, best_dev], "n_sets": len(best_sets),
        "delta_translation_mm": d_trans_mm, "delta_rotation_deg": d_rot,
        "agrees": not disagree,
    }
