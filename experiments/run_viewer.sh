#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODE="${1:?usage: experiments/run_viewer.sh EPISODE [viewer options]}"
shift
: "${DISPLAY:?DISPLAY must point to the desktop X server}"
: "${XAUTHORITY:?XAUTHORITY must point to the desktop authorization file}"

docker run --rm --detach --name viki-mujoco-viewer \\
  --gpus all --memory 8g --cpus 8 \\
  -e NVIDIA_VISIBLE_DEVICES=all \\
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \\
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/viki-xauth -e MUJOCO_GL=glfw \\
  -e PYTHONPATH=/app \\
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \\
  -v "$XAUTHORITY":/tmp/viki-xauth:ro \\
  -v "$ROOT/experiments":/app/experiments:ro \\
  -v "$ROOT/data":/app/data:ro \\
  -v "$ROOT/models":/app/models:ro \\
  -w /app viki-experiments python3 experiments/viewer.py "$EPISODE" "$@"
