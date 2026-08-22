# docker/Dockerfile
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates gnupg2 apt-transport-https \
    git \
    libusb-1.0-0 libgl1 libglib2.0-0 \
    libgl1-mesa-dri mesa-vulkan-drivers \
    libgles2-mesa \
    libegl1-mesa \
    libglvnd0 \
    python3 python3-pip \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Intel RealSense SDK ───────────────────────────────────────────────────────
# Source: https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc \
        | gpg --dearmor | tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo jammy main" \
        | tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        librealsense2 librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Azure Kinect SDK (libk4a) ─────────────────────────────────────────────────
# Packages only exist for ubuntu/18.04 on Microsoft's repo.
# EULA must be pre-accepted via debconf for non-interactive install.
# Source: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1190
RUN echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' \
        | debconf-set-selections && \
    echo 'libk4a1.4 libk4a1.4/accept-eula boolean true' \
        | debconf-set-selections && \
    wget -qO /tmp/libk4a.deb \
        https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb && \
    wget -qO /tmp/libk4a-dev.deb \
        https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        /tmp/libk4a.deb /tmp/libk4a-dev.deb && \
    rm -f /tmp/libk4a*.deb && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# No pyk4a — we use our own ctypes wrapper over libk4a.so
WORKDIR /app

COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pip-tools typing_extensions && \
    pip-compile pyproject.toml -o requirements.txt && \
    pip install -r requirements.txt

RUN mkdir -p viki && touch viki/__init__.py
RUN pip install --no-deps -e .

# Copy real source (will be overwritten by bind mount at runtime anyway)
COPY ./viki/ ./viki/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Production target
FROM base AS backend
CMD ["/bin/bash"]

# Test target
FROM base AS test
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[dev]"
CMD ["/bin/bash"]
