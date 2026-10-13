#!/usr/bin/env bash
# docker/entrypoint.sh
# Runs inside the container before starting the server.
# Prepares the writable home shared by the non-root Compose services.

set -e

mkdir -p "$HOME"

# Wrap the main command so signals reach it (the FastAPI lifespan stops the
# cameras itself on shutdown).
"$@" &
PID=$!
trap 'kill -SIGINT $PID; wait $PID' SIGINT SIGTERM
wait $PID
