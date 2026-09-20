# Multi-Kinect host freeze investigation

Status: open; the original `uvcvideo` explanation is retracted.

## Observed failure

On 2026-09-20 the host required four hard resets while the two-camera Azure
Kinect rig was in use. The last run started both cameras at 2048x1536/30 fps
and continued to serve API requests and save calibration pairs through
20:45:56. The system journal had stopped at 20:45:32 and the next boot began at
20:47:34. There was no Python exception, container OOM, kernel OOM, NVIDIA Xid,
MCE, thermal event, watchdog report, lockup report, or panic in the retained
logs. Docker reported `OOMKilled=false`.

The host remained reachable remotely for a short time after local interaction
failed. This makes an initial USB/xHCI failure more likely than an immediate
whole-host failure.

Both Kinects (`2-1` and `2-3`), the keyboard, mouse, and all other USB devices
share the machine's only USB controller, Intel Alder Lake-S PCH xHCI
`0000:00:14.0`. A controller failure can therefore remove both cameras and all
local input together. The cameras being on different motherboard sockets does
not provide controller isolation.

## Retracted `uvcvideo` hypothesis

Commit `099be56` treated a Kinect colour interface bound to `uvcvideo` as a
second owner conflicting with libk4a. That interpretation was incorrect.
libk4a's Linux USB path explicitly checks for an active kernel driver, detaches
it with `libusb_detach_kernel_driver()`, and then claims the interface. The
2026-09-20 application log shows this normal lifecycle: all four colour
interfaces were visible before opening the first device, and only the other
device's two interfaces remained before opening the second.

The same UVC probe messages also occurred on a boot that ended with a clean
shutdown. They identify Kinect enumeration, not the cause of the later freeze.
The custom `99-k4a-no-uvcvideo.rules` rule, runtime warning, and troubleshooting
claim have therefore been removed. The host copy of that rule must also be
deleted before the next hardware test; rerunning `scripts/host_setup.sh` now
performs that one-time cleanup.

## Confirmed software defect

ViKi encoded `K4A_WAIT_RESULT_TIMEOUT` as `1`; the SDK defines `FAILED=1` and
`TIMEOUT=2`. Consequently, a normal subordinate timeout while waiting for the
master was logged as `k4a_device_get_capture failed (result=2)`, while an actual
terminal stream failure would have been retried as a timeout.

The corrected lifecycle is fail-closed:

- timeouts drop one frame and keep the stream alive;
- `K4A_WAIT_RESULT_FAILED` stops the worker and records `last_error`;
- failed workers are excluded from the active-camera set;
- stale frames are not served after failure;
- a teardown that has not joined within eight seconds remains registered, and
  ViKi refuses to open a second handle for the same device.

This prevents an SDK-level stream failure from becoming a repeated call loop
against an already unhealthy USB controller. It does not prove that ViKi
caused the original xHCI failure.

## Next hardware isolation run

Run with remote SSH monitoring so local USB-input loss can be distinguished
from a whole-host lockup:

1. One Kinect physically connected, 1280x720/30 fps.
2. Two Kinects, 1280x720/15 fps, then 30 fps.
3. Two Kinects, 2048x1536/30 fps.
4. Repeat with one Kinect on an independent PCIe USB controller.
5. If needed, repeat on the installed 6.17 kernel instead of 7.0.

Persist kernel, libk4a trace, one-second CPU/memory/I/O samples, and network
liveness during each run. If the host becomes unreachable rather than merely
losing local USB, enable pstore/kdump or netconsole before further reproduction.
