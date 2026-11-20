#!/usr/bin/env bash
# Native MuJoCo GUI, using the exact saved Shadow object plan.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DISPLAY:?DISPLAY must point to the desktop X server}"
: "${XAUTHORITY:?XAUTHORITY must point to the desktop authorization file}"
if docker container inspect viki-shadow-mujoco-gui >/dev/null 2>&1; then
  docker rm -f viki-shadow-mujoco-gui
fi
docker run --rm --detach --name viki-shadow-mujoco-gui \
  --gpus all --memory 8g --cpus 8 --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/viki-xauth -e MUJOCO_GL=glfw \
  -e PYTHONPATH=/app -e HOME=/tmp \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$XAUTHORITY":/tmp/viki-xauth:ro \
  -v "$ROOT/experiments":/app/experiments:ro \
  -v "$ROOT/viki":/app/viki:ro \
  -v "$ROOT/data":/app/data:ro \
  -v "$ROOT/models":/app/models:ro \
  -v "$ROOT/data/experiments/hocap-rl":/app/rl-output \
  -w /app viki-experiments python3 experiments/hocap_shadow_replay.py --gui "$@"
