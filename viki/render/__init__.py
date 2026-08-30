"""
viki.render
-----------
All visualisation: depth colourisation + undistortion (:mod:`viki.render.depth`),
MJPEG encoding (:mod:`viki.render.mjpeg`), and the matplotlib 3-D episode
skeleton view (:mod:`viki.render.skeleton3d`). Pure numpy/cv2/matplotlib — no
FastAPI, no camera, reusable from scripts.
"""

from viki.render.depth import DepthColorizer, DepthStabilizer, Undistorter  # noqa: F401
from viki.render.mjpeg import mjpeg_chunk, placeholder  # noqa: F401

__all__ = [
    "DepthColorizer",
    "DepthStabilizer",
    "Undistorter",
    "mjpeg_chunk",
    "placeholder",
]
