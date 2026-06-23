#!/usr/bin/env bash
# scripts/host_setup_no_docker.sh
# Run once on the host machine before starting the container.
#
# Safe variant for machines where Docker is already installed/configured.
# This script DOES NOT install Docker, DOES NOT modify Docker config,
# and DOES NOT add the user to the docker group.
#
# What this does:
#   1. Adds user to plugdev, video, render groups
#   2. Installs udev rules for RealSense
#   3. Installs udev rules for Azure Kinect DK
#   4. Installs DRI udev rule for camera/GPU device access
#   5. Reloads udev rules
#
# Usage:
#   chmod +x scripts/host_setup_no_docker.sh
#   sudo ./scripts/host_setup_no_docker.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

[[ $EUID -ne 0 ]] && error "Run as root: sudo $0"

CURRENT_USER="${SUDO_USER:-$USER}"

# ── 1. User groups for camera / GPU device access ────────────────────────────
info "Adding '$CURRENT_USER' to plugdev, video, render groups..."
usermod -aG plugdev,video,render "$CURRENT_USER"

# ── 2. udev rules: Intel RealSense ───────────────────────────────────────────
info "Installing RealSense udev rules..."
cat > /etc/udev/rules.d/99-realsense.rules <<'RULES'
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666", GROUP="plugdev"
RULES

# ── 3. udev rules: Azure Kinect DK ───────────────────────────────────────────
info "Installing Azure Kinect udev rules..."
cat > /etc/udev/rules.d/99-k4a.rules <<'RULES'
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097c", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097e", MODE="0666", GROUP="plugdev"
RULES

# ── 4. udev rules: DRI ───────────────────────────────────────────────────────
# Needed by the Azure Kinect depth engine / OpenGL context inside Docker.
info "Installing DRI udev rules..."
cat > /etc/udev/rules.d/99-dri.rules <<'RULES'
SUBSYSTEM=="drm", MODE="0666"
RULES

# ── 5. Reload udev ───────────────────────────────────────────────────────────
info "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Done.${NC}"
warn "Docker was not installed or modified by this script."
warn "Log out and back in for group changes to take effect."
warn "Then reconnect your cameras and run: docker compose up --build"
