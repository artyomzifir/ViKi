#!/usr/bin/env bash
# docker/entrypoint.sh
# Runs inside the container before starting the server.
# Sets up permissions that cannot be baked into the image.

set -e

# Loosen node perms in case the host udev rules didn't fire (harmless if they
# did — the rules already set 0666). The container is no longer privileged, so
# host-kernel knobs (usbcore.usbfs_memory_mb) live in scripts/host_setup.sh now.
chmod a+rw /dev/dri/renderD* /dev/dri/card* 2>/dev/null || true
chmod a+rw /dev/bus/usb/*/*                 2>/dev/null || true

# Wrap the main command so signals reach it (the FastAPI lifespan stops the
# cameras itself on shutdown).
"$@" &
PID=$!
trap 'kill -SIGINT $PID; wait $PID' SIGINT SIGTERM
wait $PID
