"""
viki.server.deps
----------------
FastAPI dependencies. The manager and calibrator are created once in the
app lifespan and stored on ``app.state``; these resolve them so route
handlers receive them via ``Depends`` instead of reaching into app state.
"""

from __future__ import annotations

from fastapi import Request

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.skeleton_worker import SkeletonWorker
from viki.optimization.preparation.processor import PreparationPipeline


def get_manager(request: Request) -> CameraManager:
    """
    Dependency that returns the global CameraManager instance.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    CameraManager
        The application's camera manager.
    """
    return request.app.state.manager


def get_calibrator(request: Request) -> CalibrationManager:
    """
    Dependency that returns the global CalibrationManager instance.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    CalibrationManager
        The application's calibration manager.
    """
    return request.app.state.calibrator


def get_worker(request: Request) -> SkeletonWorker:
    """
    Dependency that returns the global SkeletonWorker instance.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    SkeletonWorker
        The background skeleton processing worker.
    """
    return request.app.state.skeleton_worker


def get_processor(request: Request) -> PreparationPipeline:
    """
    Dependency that returns the global PreparationPipeline instance.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    PreparationPipeline
        The skeleton data processor (used for retargeting, etc.).
    """
    return request.app.state.skeleton_processor

