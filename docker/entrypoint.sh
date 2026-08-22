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

# Wrap the main command to catch signals and ensure graceful shutdown
"$@" &
PID=$!

# Forward SIGINT and SIGTERM to the child process
trap 'python3 /app/scripts/stop_cameras.py; kill -SIGINT $PID; wait $PID' SIGINT SIGTERM

# Wait for the child process to finish
wait $PID
