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

# Cleanup for the retracted 2026-09-20 workaround. libk4a performs the kernel
# driver detach itself; leaving this udev rule behind adds an unnecessary race
# during enumeration. Keep the migration here so upgraded hosts do not retain it.
OBSOLETE_K4A_UVC_RULE=/etc/udev/rules.d/99-k4a-no-uvcvideo.rules
if [ -e "$OBSOLETE_K4A_UVC_RULE" ]; then
  rm -f -- "$OBSOLETE_K4A_UVC_RULE"
  info "Removed obsolete Kinect uvcvideo-exclusion rule."
fi

# ── 4. udev rules: DRI (GPU access for Kinect depth engine) ──────────────────
info "Installing DRI udev rules..."
cat > /etc/udev/rules.d/99-dri.rules << 'RULES'
SUBSYSTEM=="drm", MODE="0666"
RULES

udevadm control --reload-rules
udevadm trigger
info "udev rules installed."

# ── 4b. USB DMA memory limit for the Kinect pair ─────────────────────────────
# Two high-bandwidth Kinects hit the default 16 MB usbfs cap (ENOMEM, errno 12).
# Persist it here instead of poking /sys from the container (that needs
# privileged / CAP_SYS_ADMIN, which the container no longer has).
# usbcore is built into the Ubuntu kernel, not a module, so an options line in
# modprobe.d is inert — the value survives a reboot only from the kernel command
# line. Set the cmdline; keep the modprobe.d file for a kernel that does build
# usbcore as a module, and write the live value so this boot benefits too.
info "Setting usbcore.usbfs_memory_mb=1000 (cmdline + live)..."
echo "options usbcore usbfs_memory_mb=1000" > /etc/modprobe.d/viki-usbfs.conf
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || \
    warn "could not set usbfs_memory_mb for the running kernel; the cmdline below covers the next boot"

GRUB_FILE=/etc/default/grub
if [ -f "$GRUB_FILE" ]; then
  if grep -q "usbcore.usbfs_memory_mb=" "$GRUB_FILE"; then
    sed -i 's/usbcore\.usbfs_memory_mb=[0-9]*/usbcore.usbfs_memory_mb=1000/g' "$GRUB_FILE"
  else
    sed -i 's/^\(GRUB_CMDLINE_LINUX_DEFAULT="\)/\1usbcore.usbfs_memory_mb=1000 /' "$GRUB_FILE"
  fi
  if command -v update-grub >/dev/null 2>&1; then
    update-grub >/dev/null 2>&1 && info "GRUB updated; the limit persists from the next boot."
  else
    warn "update-grub not found — run your bootloader update manually."
  fi
  grep -n "GRUB_CMDLINE_LINUX_DEFAULT" "$GRUB_FILE"
else
  warn "$GRUB_FILE not found — set usbcore.usbfs_memory_mb=1000 on the kernel cmdline yourself."
fi

LIVE_USBFS=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo "?")
if [ "$LIVE_USBFS" != "1000" ]; then
  warn "usbfs_memory_mb is $LIVE_USBFS for the running kernel (want 1000)."
  warn "Two Kinects on the default 16 MB hit ENOMEM on URB submit. Reboot before a two-camera run."
fi

# ── 5. Add user to plugdev + video ───────────────────────────────────────────
info "Adding '$CURRENT_USER' to plugdev, video, render groups..."
usermod -aG plugdev,video,render "$CURRENT_USER"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Done.${NC}"
warn "Log out and back in for group changes to take effect."
warn "Then reconnect your cameras and run: docker compose up"
