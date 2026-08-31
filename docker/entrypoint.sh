#!/usr/bin/env bash
# docker/entrypoint.sh
# Runs inside the container before starting the server.
# Sets up permissions that cannot be baked into the image.

set -e

# DRI access for Kinect depth engine (OpenGL via llvmpipe)
chmod a+rw /dev/dri/renderD* 2>/dev/null || true
chmod a+rw /dev/dri/card*    2>/dev/null || true

# USB access for cameras
chmod a+rw /dev/bus/usb/*/*  2>/dev/null || true

# Increase USB DMA memory limit for multiple high-bandwidth devices (Kinect x2)
# errno=12 (ENOMEM) fix
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || true

# Wrap the main command so signals reach it (the FastAPI lifespan stops the
# cameras itself on shutdown).
"$@" &
PID=$!
trap 'kill -SIGINT $PID; wait $PID' SIGINT SIGTERM
wait $PID
