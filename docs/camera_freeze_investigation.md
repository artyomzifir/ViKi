# Multi-Kinect host freeze investigation

Status: root cause captured; the original `uvcvideo` explanation is retracted.

## Observed failure

On 2026-09-20 the host required four hard resets while the two-camera Azure
Kinect rig was in use. The last run started both cameras at 2048x1536/30 fps
and continued to serve API requests and save calibration pairs through
20:45:56. The system journal had stopped at 20:45:32 and the next boot began at
20:47:34. There was no Python exception, container OOM, kernel OOM, NVIDIA Xid,
MCE, thermal event, watchdog report, lockup report, or panic in the retained
logs. Docker reported `OOMKilled=false`.

The host remained reachable remotely for a short time after local interaction
failed. EFI pstore from the later reproductions now proves that the failure is
inside the host's xHCI kernel path, rather than merely being correlated with
USB traffic.

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

## Confirmed kernel failure

The 21:34 reproduction on Ubuntu kernel `7.0.0-28-generic` produced a complete
EFI pstore trace. A libusb thread in the ViKi `uvicorn` process issued an
USBDEVFS URB cancellation. The kernel then waited forever for the global xHCI
spinlock:

```
native_queued_spin_lock_slowpath
  xhci_urb_dequeue
    usb_hcd_unlink_urb
      usb_kill_urb
        usbdev_do_ioctl
```

The NMI watchdog reported a hard lockup on CPU 2. Two seconds later the MCE
broadcast could not stop CPUs 8-9 and the kernel panicked. This is why the
desktop and USB input failed first, remote access survived briefly, and no
ordinary journal error preceded the reset.

The archived pstore records explain the earlier resets as well:

- three other runs locked in the same xHCI spinlock from `uvicorn`, two in
  `xhci_urb_dequeue` and one in `xhci_urb_enqueue`;
- one run retained only an NVIDIA Xid 16/vblank-stall trace before the final
  MCE timeout, so its initiating CPU lock is not recoverable;
- none were container OOMs or Python exceptions.

The final reproduction happened while frames were flowing normally at
1280x720/30 fps, roughly two minutes after a 15-to-30 fps reopen. It was not
caused by the unit-test container: that container did not receive USB devices,
finished normally 33 seconds before the watchdog report, and the blocked task
in pstore belongs to the web server's libusb thread.

After the panic, a warm reboot did not power-cycle the Kinects. Both colour
devices remained unresponsive: UVC control requests timed out with `-110`, USB
U1 transitions failed, and probe ended with `-71`. Physically disconnecting
both cameras removed that poisoned device state.

This is a kernel/host-controller deadlock exposed by normal libusb URB
cancellation. Application fail-closed handling is still useful, but cannot
make the affected xHCI kernel path safe. Repeated stop/start operations merely
increase exposure to URB teardown; avoiding them is not a complete fix because
the latest lockup occurred during steady streaming.

## Next hardware isolation run

Do not reconnect the Kinects on `7.0.0-28-generic`. The machine already has
`6.17.0-40-generic`, its initramfs, and the matching NVIDIA 610.43.02 DKMS
modules installed. Boot that kernel, power-cycle the cameras, and run with
remote SSH monitoring:

1. One Kinect physically connected, 1280x720/30 fps.
2. Two Kinects, 1280x720/15 fps, then 30 fps.
3. Two Kinects, 2048x1536/30 fps.
4. Repeat with one Kinect on an independent PCIe USB controller.
5. Do not return to a 7.x kernel until the relevant xHCI regression is fixed or
   the same matrix has passed on a patched build.

Persist kernel, libk4a trace, one-second CPU/memory/I/O samples, and network
liveness during each run. Pstore is working and must remain enabled; retain its
dump after any further failure.

## Verified follow-up: kernel 6.17 is **not** a fix (2026-09-20 22:05)

The isolation matrix above was run and stopped at step 2. Result, with timestamps
from the journal:

| step | configuration | outcome |
|---|---|---|
| 1 | one Kinect, 1280x720/30, standalone | **stable** — started 22:03:50, streamed, host healthy |
| 2 | two Kinects, 1280x720/15, HW-synced rig | **host hard-locked** — rig started 22:04:16, journal stopped 22:05:01 |

The crashed boot ran `6.17.0-40-generic` (Ubuntu 6.17.13), confirmed from its own
`Linux version` banner, so **the kernel downgrade does not remove the failure**.
The 7.x-regression reading is therefore too narrow: 6.17 hard-locks too, roughly
45 seconds after the second camera is opened.

The last kernel line before silence is the second Kinect being claimed:

```
usb 2-1.1: Found UVC 1.00 device Azure Kinect 4K Camera (045e:097d)
usb 2-1.1: usbfs: process 20294 (uvicorn) did not claim interface 0 before use
```

and, on the first camera moments earlier, the isochronous endpoint being clamped:

```
usb 2-3.1: Isoc endpoint with wBytesPerInterval of 1024 ... ep 129: setting to 944
```

No pstore record survived this one, so there is no stack for it; the diagnosis
rests on the four archived dumps from the 7.x runs plus this reproduction.

**Caveat on the trigger.** The sequence was start-one → stop → start-both inside
30 seconds, so this run cannot separate "two cameras on one controller" from
"URB teardown immediately followed by re-enumeration". Both are on the suspect
path. A clean next run should power-cycle, then open two cameras once, with no
prior stop.

### What this leaves as the actual fix

The remaining explanation is the one the earlier research already pointed at and
the kernel version does not change: **both Kinects, the keyboard and the mouse
share the one Intel Alder Lake-S PCH xHCI controller**. Microsoft's multi-Kinect
guidance calls for separate USB root ports, and working multi-camera rigs use a
discrete PCIe USB card; Azure Kinect issue #1905 reports this same full-system
lockup with no useful log. Different sockets on the case are not different
controllers.

So:

1. **Fix:** give the second Kinect its own host controller — a PCIe USB 3.x card
   with a dedicated lane. That is the only measure that addresses the shared
   controller, and it is what the vendor documents for this configuration.
2. **Meanwhile, one Kinect is usable** — step 1 above passed. Single-camera
   capture, calibration and playback can proceed on this host today.
3. **Reduce exposure, do not mistake it for a cure:** every stop/start cancels
   isochronous URBs, which is the path the archived dumps deadlock on. The
   fail-closed lifecycle from `3894e71` helps ViKi not pile calls onto an
   already-sick controller, but cannot make that kernel path safe.
4. Keep pstore enabled; it captured four of the five dumps and is the only
   instrument that has ever produced a stack here.
