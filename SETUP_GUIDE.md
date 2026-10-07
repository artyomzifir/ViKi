# ViKi — Setup Guide

This guide covers the complete one-time setup for running ViKi on a fresh Ubuntu machine with the reference hardware configuration used during development.

**Reference hardware:**
- Host: any x86_64 PC running Ubuntu 22.04, with a dedicated GPU or Intel integrated graphics
- Cameras: Intel RealSense D435i × 1, Azure Kinect DK × 2
- Sync cable: 3.5mm mono (TS) audio cable

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Host setup script](#2-host-setup-script)
3. [USB memory limit for Azure Kinect](#3-usb-memory-limit-for-azure-kinect)
4. [X11 access for Kinect depth engine](#4-x11-access-for-kinect-depth-engine)
5. [USB wiring for two Azure Kinects](#5-usb-wiring-for-two-azure-kinects)
6. [Hardware sync cable](#6-hardware-sync-cable)
7. [Running ViKi](#7-running-viki)
8. [Camera settings reference](#8-camera-settings-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Requirements

### Software

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 LTS x86_64 | Other distros untested |
| Docker Engine | 24+ | Installed by `host_setup.sh` |
| Docker Compose plugin | v2 | Installed by `host_setup.sh` |

### Hardware

| Device | Role | USB requirement |
|---|---|---|
| Intel RealSense D435i | Observation camera (policy input) | USB 3.0 (5 Gbps) |
| Azure Kinect DK | Kinematics extraction | USB 3 SuperSpeed; **one independent host controller per Kinect** |
| 3.5mm mono cable | Hardware sync between Kinects | Any length under ~3m |

> ### Two-Kinect USB-controller issue: resolved on the current rig
>
> Two Kinects sharing the Intel Alder Lake-S PCH xHCI previously hard-locked this
> host. The second Kinect now uses an independent Renesas µPD720201 PCIe xHCI;
> dual-camera vendor and ViKi captures work on this wiring. For this rig, use
> **one independent host controller per Kinect**. Two ports or hubs under one
> controller do not provide that isolation. The old failure and its recovery
> procedure are retained as history in
> [`docs/camera_freeze_investigation.md`](docs/camera_freeze_investigation.md).

### USB controller requirement

Each Kinect streams colour and depth using SuperSpeed USB. The old failure was
at the **shared host controller**, not merely a shared hub. This rig runs one
Kinect on the Intel PCH xHCI (`0000:00:14.0`) and the other on the independent
Renesas PCIe xHCI (`0000:08:00.0`). Both cameras enumerate at `5000M` and work;
a `10000M` link is not required by this measured setup.

Check the current wiring with:

```bash
lsusb -t
lspci -nn | grep -i 'USB controller'
```

The current working topology has one Kinect beneath Bus 002 (`xhci_hcd`, Intel)
and one beneath Bus 004 (`xhci-pci-renesas`, Renesas). The PCI devices above are
different controllers. Do not infer isolation from different sockets, hubs, or
USB bus numbers alone: one xHCI can expose both USB 2 and USB 3 buses. For more
Kinects on this rig, budget one independent controller per camera.

---

## 2. Host setup script

Run once as root on the host machine. This installs Docker, sets up udev rules for all cameras, and adds your user to the necessary groups.

```bash
chmod +x scripts/host_setup.sh
sudo ./scripts/host_setup.sh
```

What it does:

- Installs Docker Engine and Docker Compose plugin (if not present)
- Creates `/etc/udev/rules.d/99-realsense.rules` — grants RealSense USB access without root
- Creates `/etc/udev/rules.d/99-k4a.rules` — grants Azure Kinect USB access without root
- Creates `/etc/udev/rules.d/99-dri.rules` — grants DRI device access (`MODE=0666`) needed by the Kinect depth engine OpenGL context inside Docker
- Adds your user to groups: `docker`, `plugdev`, `video`, `render`

**After the script completes: log out and back in.** Group membership changes require a new login session to take effect.

---

## 3. USB memory limit for Azure Kinect

The Linux kernel's default USB DMA memory limit (16 MB) is insufficient for two Azure Kinects streaming simultaneously. Without this fix, the second device fails with `LIBUSB_ERROR_IO (errno=12, ENOMEM)`.

This must be set as a kernel boot parameter via GRUB — setting it at runtime inside Docker is unreliable.

**Edit GRUB:**

```bash
sudo nano /etc/default/grub
```

Find this line:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Replace with:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000"
```

**Apply and reboot:**

```bash
sudo update-grub
sudo reboot
```

**Verify after reboot:**

```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

Expected output: `1000`

> Source: [Azure Kinect SDK issue #485](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/485)

---

## 4. X11 access for Kinect depth engine

The Azure Kinect depth engine is a closed-source binary that processes raw IR sensor data using OpenGL shaders (via the llvmpipe software rasterizer inside Docker). It needs access to the host X11 display.

Add this line to `~/.bashrc` so it runs automatically at every login:

```bash
echo 'xhost +local: > /dev/null 2>&1' >> ~/.bashrc
source ~/.bashrc
```

If you skip this step, Kinect start will fail with:

```
Authorization required, but no authorization protocol specified
depth engine create and initialize failed with error code: 204
```

---

## 5. USB wiring for two Azure Kinects

1. Connect one Kinect to a motherboard Intel port and the other to the Renesas PCIe card.
2. Run `lsusb -t` and `lspci -nn | grep -i 'USB controller'`; confirm each camera is beneath a different PCI xHCI controller (see [Requirements](#1-requirements)).
3. Recheck after moving cables. Device indices and USB bus numbers can change; physical controller isolation is the invariant.

Each Azure Kinect DK appears as two USB devices:
- A `Video` class device (color camera, `uvcvideo` driver)
- A `Vendor Specific Class` device (depth MCU, no kernel driver — accessed directly by libk4a)

---

## 6. Hardware sync cable

Without hardware sync, two Kinects interfere with each other's IR depth projectors, producing corrupted depth frames when both are aimed at the same scene.

**Cable:** standard 3.5mm mono (TS) audio cable. Stereo (TRS) also works. Keep it under ~3m.

**Wiring:**
```
Kinect A  SYNC OUT ──── cable ──── SYNC IN  Kinect B
(master)                                   (subordinate)
```

> The SYNC IN / SYNC OUT ports are on the back of the Kinect under a small removable plastic cover.

**Startup in ViKi UI:**

Set the physical roles in `KINECT_SYNC` (`data/user_configuration.json`) to
match the cable direction. Clicking **Start** on either Kinect starts the entire
rig automatically: every subordinate first, then the master.

With two or more connected Kinects, standalone mode is forbidden. ViKi checks
the SDK-reported SYNC IN/SYNC OUT jack state and refuses to start a partial or
miswired rig. It also verifies that the per-device K4A timestamp offset
stays stable within 500 µs across buffered frames. Recording has a second gate and cannot
bypass this check with the debug `force` option. Clicking **Stop** on either
Kinect stops the whole rig.

---

## 7. Running ViKi

First run (builds the Docker image, takes ~5 minutes):

```bash
docker compose up --build
```

Subsequent runs:

```bash
docker compose up
```

Open `http://localhost:8000` in your browser.

To open a debug terminal inside the container:

```bash
docker compose run --rm terminal
```

To stop:

```bash
Ctrl+C
# or
docker compose down
```

---

## 8. Camera settings reference

### Intel RealSense D435i

| Setting | Supported values | Recommended |
|---|---|---|
| Resolution | 640×480, 1280×720, 1920×1080 | 1280×720 |
| FPS | 15, 30 | 30 |
| Depth mode | — (always aligned to color) | — |

### Azure Kinect DK

| Setting | Supported values | Recommended |
|---|---|---|
| Resolution | 1280×720, 1920×1080, 2048×1536 | 1280×720 |
| FPS | 5, 15, 30 | 30 |
| Depth mode | see below | NFOV_UNBINNED |

**Depth mode comparison:**

| Mode | Resolution | Range | Notes |
|---|---|---|---|
| `NFOV_UNBINNED` | 640×576 | up to ~3.86m | Best accuracy, narrow FOV — recommended for tabletop |
| `NFOV_2X2BINNED` | 320×288 | up to ~5.46m | Faster, lower resolution |
| `WFOV_UNBINNED` | 1024×1024 | up to ~2.21m | Wide FOV, fisheye distortion |
| `WFOV_2X2BINNED` | 512×512 | up to ~2.88m | Wide FOV, lower resolution |

For manipulation tasks at 0.5–1.5m: `NFOV_UNBINNED` gives the best depth accuracy.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Kinect fails: `depth engine error 204` | X11 not accessible | Run `xhost +local:` before `docker compose up` |
| Kinect fails: `depth engine error 207 — OpenGL 4.4 context creation failed` | DRI not accessible or wrong driver | Check `/dev/dri` exists in container; ensure `GALLIUM_DRIVER=llvmpipe` is set |
| Second Kinect fails: `LIBUSB_ERROR_IO errno=12` | USB DMA memory limit | Apply GRUB `usbfs_memory_mb=1000` fix and reboot |
| Second Kinect fails: `LIBUSB_ERROR_BUSY` | USB resource conflict | Check that each Kinect is on its own PCI xHCI controller, then retry after stopping the rig |
| Kinect fails: `k4a_device_open failed` after stop/start | USB not released yet | Wait 2–3 seconds after stopping before restarting |
| RealSense fails: `Couldn't resolve requests` | Unsupported resolution/fps | Use 640×480 or 1280×720; avoid 4K |
| RealSense: very low framerate or timeouts | USB 2.0 port or shared hub | Connect to a USB 3.0 port directly on the motherboard |
| Depth stream black in UI | ID conflict resolved, but browser cached old state | Hard-refresh the page (Ctrl+Shift+R) |
| `Authorization required` in logs | `xhost` not set | Add `xhost +local:` to `~/.bashrc` |
| Container exits with code 139 (segfault) | Kinect buffer read issue | Update to latest `kinect.py` which uses `ctypes.string_at` for safe buffer copy |
