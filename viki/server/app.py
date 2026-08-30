"""
viki.server.app
---------------
Application assembly only: lifespan resources, static files, router wiring.
Request logic lives in ``viki.server.routes``; the offline pipeline stages live
in ``viki.{cameras,perception,prepare,retarget,replay,export}`` and are driven
here only through thin job endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from viki.calibration.manager import CalibrationManager
from viki.cameras.manager import CameraManager
from viki.server.routes import (
    calibration,
    cameras,
    datasets,
    export,
    label,
    pipeline,
    recording,
    replay,
    skeleton,
    system,
)

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the two long-lived objects; stop cameras on shutdown."""
    app.state.manager = CameraManager()
    app.state.calibrator = CalibrationManager(app.state.manager)
    from viki.calibration import presets as _presets

    applied = _presets.apply_active_on_startup()
    if applied:
        logging.getLogger(__name__).info("active calibration preset: %s", applied)
    app.state.calibrator.load_all_extrinsics()
    yield
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Server", lifespan=lifespan)


@app.middleware("http")
async def _no_cache_frontend(request: Request, call_next):
    """The frontend is bind-mounted and has no build step / content hashing, so a
    browser that cached an older ``index.html`` keeps serving a stale bundle.
    Force revalidation on the page and its assets (ETag still makes it a cheap
    304 when nothing changed)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

router = APIRouter(prefix="/api", tags=["api"])
for mod in (cameras, calibration, skeleton, system, recording, pipeline, replay, label, export):
    router.include_router(mod.router)
router.include_router(datasets.router)
router.include_router(datasets.ep_router)
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend HTML page."""
    return (STATIC_DIR / "index.html").read_text()
