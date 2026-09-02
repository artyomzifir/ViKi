"""
viki.perception.rs_offline
--------------------------
Replay a RealSense colour↔depth registration from the JSON the recorder stored
(``raw/<dev>_rs_calib.json``), with no device. The RealSense analogue of
:mod:`viki.perception.k4a_offline`.

``RealSenseCalibration`` satisfies :class:`viki.contracts.DepthProjector`
(``project_color_to_depth``) and provides the same extra hooks the offline
stages call on a Kinect calibration — ``depth3d_to_color3d`` and
``color_deproject_maps`` — so ``extract`` / ``cloud`` / ``hand_fit`` treat both
backends identically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from viki.cameras.rs_math import deproject_pixel, project_point

logger = logging.getLogger(__name__)


class RealSenseCalibration:
    """Colour↔depth reprojection from stored RealSense stream intrinsics + the
    depth→colour extrinsic. All maths is pinhole + Brown-Conrady, pure NumPy."""

    def __init__(self, color_intr: dict, depth_intr: dict, R_dc: np.ndarray, t_dc: np.ndarray):
        self._color = color_intr
        self._depth = depth_intr
        self._R = np.asarray(R_dc, dtype=np.float64).reshape(3, 3)  # depth→colour
        self._t = np.asarray(t_dc, dtype=np.float64).reshape(3)     # metres
        self._map_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------

    @classmethod
    def from_episode(cls, raw_dir, dev_id: str, meta: dict | None) -> "RealSenseCalibration | None":
        """Build from ``raw/<dev>_rs_calib.json``. ``None`` when the file is
        absent (caller falls back to the identity projector)."""
        raw_dir = Path(raw_dir)
        cam = ((meta or {}).get("cameras") or {}).get(dev_id, {}) or {}
        p = raw_dir / cam.get("rs_calib", f"{dev_id}_rs_calib.json")
        if not p.is_file():
            return None
        try:
            d = json.loads(p.read_text())
            ext = d["depth_to_color"]
            R = np.asarray(ext["rotation"], dtype=np.float64).reshape(3, 3).T  # col-major → M
            t = np.asarray(ext["translation"], dtype=np.float64)
            cal = cls(d["color"], d["depth"], R, t)
            logger.info("rs_offline[%s]: loaded colour↔depth calibration", dev_id)
            return cal
        except Exception as exc:  # noqa: BLE001 — never abort a stage over calib
            logger.warning("rs_offline[%s]: %s unusable (%s)", dev_id, p.name, exc)
            return None

    # ── projections ───────────────────────────────────────────────────

    def project_color_to_depth(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Colour pixel + expected range ``z`` (metres) → depth-image pixel."""
        p_col = deproject_pixel(self._color, float(u), float(v), float(z))
        p_dep = self._R.T @ (p_col - self._t)
        if p_dep[2] <= 0:
            return None
        uv = project_point(self._depth, p_dep)
        return (float(uv[0]), float(uv[1]))

    def depth3d_to_color3d(self, xyz_m) -> np.ndarray | None:
        """3-D point (metres, depth frame) → colour camera frame. The offline
        lift deprojects with the depth intrinsics but the recorded extrinsics are
        the colour camera's ChArUco pose."""
        p = np.asarray(xyz_m, dtype=np.float64).reshape(3)
        return self._R @ p + self._t

    def color_deproject_maps(self, dh: int, dw: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel ``(A, B)`` such that a depth pixel ``(u, v)`` at range
        ``z`` mm lands at ``z · A[v, u] + B[v, u]`` mm in the **colour** camera
        frame — the vectorised form ``cloud`` / ``hand_fit`` consume. ``A`` is the
        rotated deprojection ray, ``B`` the constant depth→colour offset."""
        key = (int(dh), int(dw))
        hit = self._map_cache.get(key)
        if hit is not None:
            return hit
        vs, us = np.mgrid[0:dh, 0:dw]
        rays = deproject_pixel(self._depth, us, vs, 1.0)      # (dh, dw, 3), depth frame
        A = rays @ self._R.T                                  # rotate into colour frame
        B = np.broadcast_to(self._t * 1000.0, (dh, dw, 3)).copy()  # metres → mm
        A = A.astype(np.float64)
        self._map_cache[key] = (A, B)
        logger.info("rs_offline: built %dx%d colour-deprojection maps", dw, dh)
        return A, B
