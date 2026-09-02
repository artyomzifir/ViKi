"""
viki.cameras.rs_math
--------------------
Pure-NumPy ports of librealsense's ``rs2_deproject_pixel_to_point`` and
``rs2_project_point_to_pixel`` (rsutil.h). No ``pyrealsense2`` needed — so the
offline stages can reproject a RealSense colour↔depth pixel from the intrinsics
JSON stored at record time, the same way :mod:`viki.perception.k4a_offline` does
for the Kinect.

An intrinsics dict is ``{"fx","fy","ppx","ppy","model","coeffs":[k1,k2,p1,p2,k3]}``.
``model`` is one of ``none`` / ``brown_conrady`` / ``modified_brown_conrady`` /
``inverse_brown_conrady``. Fisheye models (``ftheta``, ``kannala_brandt4``) are
not produced by a D4xx colour or depth stream; they fall through as ``none``
with a logged warning.

All functions accept scalars or broadcastable arrays for the pixel / point
coordinates, so ``deproject_pixel`` can be run over a whole depth image at once.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_BROWN = {"brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"}


def _coeffs(intr: dict) -> tuple[float, float, float, float, float]:
    c = list(intr.get("coeffs") or ())
    c += [0.0] * (5 - len(c))
    return c[0], c[1], c[2], c[3], c[4]


def deproject_pixel(intr: dict, u, v, depth):
    """Pixel ``(u, v)`` at range ``depth`` → 3-D point in that camera's frame.

    ``depth`` shares the returned units (pass metres, get metres; the ``z``
    component equals ``depth``).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = (u - intr["ppx"]) / intr["fx"]
    y = (v - intr["ppy"]) / intr["fy"]

    model = str(intr.get("model", "none")).lower()
    if model in _BROWN:
        k1, k2, p1, p2, k3 = _coeffs(intr)
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        ux = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        uy = y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        x, y = ux, uy
    elif model not in ("none", "", "distortion.none"):
        logger.warning("rs_math.deproject_pixel: unsupported model %r — treating as none", model)

    depth = np.asarray(depth, dtype=np.float64)
    return np.stack([depth * x, depth * y, depth * np.ones_like(x)], axis=-1)


def project_point(intr: dict, point):
    """3-D point in the camera frame → pixel ``(u, v)``. ``point`` last axis is xyz."""
    point = np.asarray(point, dtype=np.float64)
    x = point[..., 0] / point[..., 2]
    y = point[..., 1] / point[..., 2]

    model = str(intr.get("model", "none")).lower()
    if model in _BROWN:
        k1, k2, p1, p2, k3 = _coeffs(intr)
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xf, yf = x * f, y * f
        dx = xf + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = yf + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        x, y = dx, dy
    elif model not in ("none", "", "distortion.none"):
        logger.warning("rs_math.project_point: unsupported model %r — treating as none", model)

    u = x * intr["fx"] + intr["ppx"]
    v = y * intr["fy"] + intr["ppy"]
    return np.stack([u, v], axis=-1)


def extrinsic_matrix(rotation9, translation3) -> tuple[np.ndarray, np.ndarray]:
    """librealsense ``rs2_extrinsics`` → ``(R 3x3, t 3)`` with ``to = R @ from + t``.

    ``rs2_extrinsics.rotation`` is column-major, ``translation`` is in metres.
    """
    R = np.asarray(rotation9, dtype=np.float64).reshape(3, 3).T
    t = np.asarray(translation3, dtype=np.float64).reshape(3)
    return R, t
