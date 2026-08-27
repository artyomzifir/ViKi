"""
viki.server.deps
----------------
FastAPI dependencies. The camera manager and calibrator are built once in the
app lifespan and stored on ``app.state``; handlers get them via ``Depends`` so
they don't reach into app state directly. There is no live pipeline / worker
anymore — extraction, preparation, retargeting, replay and export all run
offline over episode directories.
"""

from __future__ import annotations

from fastapi import Request

from viki.calibration.manager import CalibrationManager
from viki.cameras.manager import CameraManager


def get_manager(request: Request) -> CameraManager:
    """The application's CameraManager."""
    return request.app.state.manager


def get_calibrator(request: Request) -> CalibrationManager:
    """The application's CalibrationManager."""
    return request.app.state.calibrator
