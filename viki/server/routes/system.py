"""
viki.server.routes.system
-------------------------
System endpoints: configuration management and server restart.
"""

from fastapi import APIRouter, HTTPException
import json
import os
import shutil
import subprocess
import time
import logging
from viki.config import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["system"])

# ── host load monitor (CPU / RAM / GPU) ──────────────────────────────────
# No psutil in the image — read /proc directly. CPU% needs two samples, so we
# cache the last /proc/stat jiffies snapshot between calls.
_cpu_prev: tuple[float, float] | None = None  # (idle, total)
_gpu_cache: tuple[float, list | None] = (0.0, None)  # (t, parsed) — nvidia-smi is slow


def _cpu_percent() -> float | None:
    global _cpu_prev
    try:
        with open("/proc/stat") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
    except OSError:
        return None
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)  # idle + iowait
    total = sum(parts)
    prev, _cpu_prev = _cpu_prev, (idle, total)
    if prev is None:
        return None
    d_total = total - prev[1]
    d_idle = idle - prev[0]
    if d_total <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total))), 1)


def _mem() -> dict | None:
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = float(v.strip().split()[0]) * 1024.0  # kB → bytes
    except OSError:
        return None
    total = info.get("MemTotal", 0.0)
    avail = info.get("MemAvailable", info.get("MemFree", 0.0))
    used = max(0.0, total - avail)
    return {
        "used": round(used),
        "total": round(total),
        "percent": round(100.0 * used / total, 1) if total else None,
    }


def _loadavg() -> list[float] | None:
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:
        return None


def _gpu() -> list | None:
    """nvidia-smi one-shot, cached ~2 s (spawning it per poll is wasteful)."""
    global _gpu_cache
    now = time.time()
    if now - _gpu_cache[0] < 2.0:
        return _gpu_cache[1]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2.0,
        )
        gpus = []
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                f = [c.strip() for c in line.split(",")]
                if len(f) < 5:
                    continue
                gpus.append({
                    "name": f[0],
                    "util": float(f[1]),
                    "mem_used": round(float(f[2]) * 1024 * 1024),
                    "mem_total": round(float(f[3]) * 1024 * 1024),
                    "temp": float(f[4]),
                })
        parsed = gpus or None
    except (OSError, subprocess.SubprocessError, ValueError):
        parsed = None
    _gpu_cache = (now, parsed)
    return parsed


@router.get("/system/stats")
async def system_stats():
    """Live host load: CPU %, RAM, GPU util + memory. For the top-bar monitor."""
    return {
        "cpu_percent": _cpu_percent(),
        "mem": _mem(),
        "loadavg": _loadavg(),
        "gpu": _gpu(),
    }

@router.get("/health")
async def health():
    """Liveness probe — the header status dot goes green when this answers."""
    return {"status": "ok"}


@router.get("/config")
async def get_config():
    """
    Get the current user configuration (from `user_configuration.json`).

    Returns
    -------
    dict
        The configuration object.

    Raises
    ------
    HTTPException 404
        If the user configuration file does not exist.
    """
    if not os.path.exists(USER_CONFIG_PATH):
        raise HTTPException(status_code=404, detail="User configuration file not found")
    with open(USER_CONFIG_PATH, "r") as f:
        return json.load(f)

@router.post("/config")
async def save_config(config: dict):
    """
    Save a new configuration to `user_configuration.json`.

    Parameters
    ----------
    config : dict
        Full configuration object.

    Returns
    -------
    dict
        {"status": "success"}

    Raises
    ------
    HTTPException 500
        If saving fails (e.g., permission error).
    """
    try:
        with open(USER_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/reset")
async def reset_config():
    """
    Reset the user configuration to the default by copying `default_configuration.json`.

    Returns
    -------
    dict
        {"status": "success"}

    Raises
    ------
    HTTPException 404
        If the default configuration file is missing.
    HTTPException 500
        If copying fails.
    """
    try:
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="Default configuration file not found")
        shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to reset config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restart")
async def restart_server():
    """
    Restart the server by exiting the process (container restarts automatically).

    This endpoint calls `os._exit(1)`, which terminates the Python process.
    With Docker's `restart: unless-stopped`, the container will restart.
    """
    logger.info("Restarting server via API request...")
    # os._exit(1) is used to kill the python process immediately.
    # Since the container is set to restart: unless-stopped, Docker will restart it.
    os._exit(1)
