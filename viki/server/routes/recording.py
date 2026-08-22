"""
viki.server.routes.recording
----------------------------
Endpoints for starting/stopping RGB-D recording in the background worker.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from viki.capture.manager import CameraManager
from viki.server.deps import get_manager, get_worker
from viki.server.skeleton_worker import SkeletonWorker

router = APIRouter(prefix="/record", tags=["recording"])

class RecordRequest(BaseModel):
    duration: float = 10.0
    fps: int = 15
    output_dir: str = "data/videos"

@router.post("/start")
async def start_recording(
    req: RecordRequest,
    mgr: CameraManager = Depends(get_manager),
    worker: SkeletonWorker = Depends(get_worker),
):
    """
    Start an RGB-D recording in the background worker.

    The recording will run for the specified duration and save raw colour
    and depth data to the output directory.

    Parameters
    ----------
    req : RecordRequest
        Duration (seconds), FPS, and output directory.

    Returns
    -------
    dict
        {"status": "recording started in background worker", "duration": float, "fps": int}

    Raises
    ------
    HTTPException 400
        If no cameras are currently active.
    """
    if not mgr.active_device_ids():
        raise HTTPException(status_code=400, detail="No cameras are currently active. Start cameras first.")
    
    worker.set_rgbd_recording(
        enabled=True, 
        duration=req.duration, 
        output_dir=req.output_dir
    )
    
    return {"status": "recording started in background worker", "duration": req.duration, "fps": req.fps}
