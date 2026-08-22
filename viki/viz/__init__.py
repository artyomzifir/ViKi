"""
viki.viz
--------
Pixel-level helpers shared by the streaming server: MJPEG encoding and
depth/colour image preparation. Pure functions over numpy arrays — no
FastAPI, no camera, no I/O — so they are reusable (e.g. by Phase 2 overlays)
and testable without hardware.
"""
