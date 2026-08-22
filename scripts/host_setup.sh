#!/usr/bin/env bash
# scripts/host_setup.sh
# Run once on the host machine (Ubuntu 24.04) before starting the container.
#
# What this does:
#   1. Installs Docker (if not present)
#   2. Adds user to docker group
#   3. udev rules for RealSense, Azure Kinect, and DRI
#   4. Adds user to plugdev + video groups
#
# Usage:
#   chmod +x scripts/host_setup.sh
#   sudo ./scripts/host_setup.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

[[ $EUID -ne 0 ]] && error "Run as root: sudo $0"

CURRENT_USER="${SUDO_USER:-$USER}"

# ── 1. Docker ─────────────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    info "Docker already installed: $(docker --version)"
else
    info "Installing Docker..."
    apt-get update -qq
    apt-get install -y --no-install-recommends ca-certificates curl gnupg

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    info "Docker installed: $(docker --version)"
fi

info "Adding '$CURRENT_USER' to docker group..."
usermod -aG docker "$CURRENT_USER"

# ── 2. udev rules: Intel RealSense ───────────────────────────────────────────
info "Installing RealSense udev rules..."
cat > /etc/udev/rules.d/99-realsense.rules << 'RULES'
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666", GROUP="plugdev"
RULES

# ── 3. udev rules: Azure Kinect DK ───────────────────────────────────────────
info "Installing Azure Kinect udev rules..."
cat > /etc/udev/rules.d/99-k4a.rules << 'RULES'
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097c", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097e", MODE="0666", GROUP="plugdev"
RULES

# ── 4. udev rules: DRI (GPU access for Kinect depth engine) ──────────────────
info "Installing DRI udev rules..."
cat > /etc/udev/rules.d/99-dri.rules << 'RULES'
SUBSYSTEM=="drm", MODE="0666"
RULES

udevadm control --reload-rules
udevadm trigger
info "udev rules installed."

# ── 5. Add user to plugdev + video ───────────────────────────────────────────
info "Adding '$CURRENT_USER' to plugdev, video, render groups..."
usermod -aG plugdev,video,render "$CURRENT_USER"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Done.${NC}"
warn "Log out and back in for group changes to take effect."
warn "Then reconnect your cameras and run: docker compose up"