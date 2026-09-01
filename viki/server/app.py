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
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_log = logging.getLogger("viki.api")

# Endpoints hit by the frontend's poll loops — logging every call floods the log
# and drowns the events that matter. Uvicorn's access log still records them.
_QUIET_PATHS = (
    "/api/system/stats", "/api/health", "/api/pipeline/jobs",
    "/api/calibration/status/", "/api/calibration/samples_count/",
    "/api/record/jobs/", "/api/replay/jobs/", "/api/export/jobs/",
)


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
async def _observe(request: Request, call_next):
    """Two things: (1) force frontend revalidation — the static bundle is
    bind-mounted with no content hashing, so a stale ``index.html`` would keep
    serving old JS; (2) log every state-changing API call (method, path, status,
    duration) plus any 4xx/5xx, so the server log carries a record of what was
    done, not just uvicorn's access lines. Poll endpoints are skipped."""
    path = request.url.path
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _log.exception("%s %s -> unhandled exception (%.0f ms)",
                       request.method, path, (time.perf_counter() - t0) * 1e3)
        raise
    dt_ms = (time.perf_counter() - t0) * 1e3

    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"

    if path.startswith("/api/") and not any(path.startswith(q) for q in _QUIET_PATHS):
        action = request.method != "GET"
        if action or response.status_code >= 400:
            lvl = logging.WARNING if response.status_code >= 400 else logging.INFO
            _log.log(lvl, "%s %s -> %s (%.0f ms)",
                     request.method, path, response.status_code, dt_ms)
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
