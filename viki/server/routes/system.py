"""
viki.server.routes.system
-------------------------
System endpoints: configuration management and server restart.
"""

from fastapi import APIRouter, HTTPException
import json
import os
import shutil
import logging
from viki.config import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["system"])

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
