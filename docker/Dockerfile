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

# The source tree is never baked into the image: docker-compose.yml bind-mounts
# ./viki -> /app/viki at runtime and PYTHONPATH=/app makes it importable. We only
# install an editable link (the stub package dir is enough) so the `viki` console
# script and package metadata exist; the real modules come from the bind mount.
RUN mkdir -p viki && touch viki/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

# ── Perception backend: RTMPose-Hand via rtmlib on ONNX Runtime (CUDA) ────────
# rtmlib hard-depends on opencv-python, which would fight the
# opencv-python-headless pinned in pyproject.toml — install it without deps and
# bring its real needs (onnxruntime-gpu, tqdm; cv2 comes from headless) explicitly.
#
# GPU: onnxruntime-gpu ships CUDA 12 wheels (sm_120 / Blackwell OK) and the
# nvidia-*-cu12 runtime wheels carry the CUDA/cuDNN .so files, so no CUDA base
# image. The driver is injected by nvidia-container-toolkit (`gpus: all`).
# `onnx` is for _pick_device()'s CUDA-EP probe session.
#
# CRITICAL: libonnxruntime_providers_cuda.so links libcublas.so.12 / libcudnn.so.9
# / libcudart.so.12 etc. by SONAME, and the nvidia wheels drop those under
# site-packages/nvidia/*/lib which is NOT on the default loader path — without
# the ldconfig entry below the CUDA EP fails to load and ORT *silently* falls
# back to CPU. (rtmpose.py also calls onnxruntime.preload_dlls() as a backstop.)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "onnxruntime-gpu>=1.22" "tqdm>=4.66" onnx \
        nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
        nvidia-cufft-cu12 nvidia-curand-cu12 && \
    pip install --no-deps rtmlib && \
    ls -d /usr/local/lib/python3.*/dist-packages/nvidia/*/lib \
        > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Production target
FROM base AS backend
CMD ["/bin/bash"]

# Test target
FROM base AS test
# "-e" keeps the project a link to the (bind-mounted) tree, not a copy in
# site-packages; "[dev]" pulls the test extras (pytest). Runtime deps are
# already satisfied from requirements.txt above, so this resolves from cache.
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[dev]"
CMD ["/bin/bash"]
