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

### What this leaves (superseded — see the external review below)

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

## External review, 2026-09-20: the "separate controller" conclusion is overstated

A second review supplied upstream references that change the reading above. The
bug reports are cited as received, not independently reproduced here; the
repository claims below were re-verified against `v0.0.2`.

### Why the downgrade changed nothing

**kernel bugzilla #221103** — `xhci_hcd: System lockup`, opened February 2026 and
still live in August 2026: full hard-lock, no stack, no panic, no journal.
Trigger is usbfs open/close under load; the root cause given is the xHC coming
out of runtime-suspend too fast, its registers reading back `0xffffffff`, then
`Controller not ready at resume -19` → HC died. **Affected range 6.12–7.0 covers
both kernels tested here**, which explains the 6.17 reproduction directly: the
downgrade never left the window.

That also supplies a mechanism for our stack. If the controller stops answering
MMIO, whoever holds `xhci->lock` spins in `xhci_handshake` on a long timeout
while every other CPU piles into `native_queued_spin_lock_slowpath` — which is
what the archived dumps show, and it matches the desktop and USB input dying
first while SSH lived a little longer.

Confirmed workaround in that report: **`usbcore.autosuspend=-1`** plus
`power/control=on` on the controller. One cmdline line, one run — and it is the
cheapest test available, so it comes before any hardware purchase.

Related isochronous reports: a linux-usb thread (August 2026) on repeated URB
cancel against a UVC status interrupt endpoint desynchronising the isoc ring on
Intel xHCI — old bug, not a regression, partially helped by `uvcvideo
quirks=0x80` (FIX_BANDWIDTH); and **#220748**, xHCI ignoring `start_frame` and
always assuming `URB_ISO_ASAP`. Our own log line
`Isoc endpoint with wBytesPerInterval of 1024 ... setting to 944` sits in that
same territory.

### Why "separate controller" is not yet the answer

**Azure Kinect #1905** was cited above as evidence for the shared-controller
theory. Read properly it weakens it: that report is on Windows 11, and the
reporter states the lockups occur across *different ports and different
controllers*. If the same hang reaches a Windows stack, the problem is closer to
how the Kinect drives isochronous streams than to Linux xHCI specifically. (That
report also ran the cameras on USB-C power with no mains PSU, an independent
source of instability, and the repository was archived in August 2024 with no
vendor answer.)

So the honest formulation is: **a separate controller removes one of two
suspects, and not the one all four archived dumps sit on.** It is still worth
doing — the vendor documents it, and #1401 recommends cards with a real
controller per port (Renesas µPD720202, FL1100) — but it is insurance, not a
diagnosis.

### The decisive experiment, still unrun and still free

Step 2 above did start-one → stop → start-both inside 30 seconds, so it never
separated "two cameras on one controller" from "URB teardown immediately before
re-enumeration". Until this is run, no purchase is justified by evidence:

1. Unplug both cameras; add `usbcore.autosuspend=-1` to the cmdline; reboot.
2. `echo on > /sys/bus/pci/devices/0000:00:14.0/power/control`.
3. Reconnect both cameras, on their mains PSUs.
4. Open **once**: subordinate, then master, 1280x720/30. No stop/start afterwards.
5. Hold 30 minutes under load with SSH monitoring, pstore and netconsole armed
   (#221103 needed netconsole on the runs where pstore stayed empty).

Passes → the killer is teardown/re-enumeration, and the operating discipline
below fixes it outright. Fails → two cameras on one controller is real and the
PCIe card is justified. Fails on one camera → the controller or board is faulty.

### Repository findings from the same review (verified on `v0.0.2`)

- **`color_format=K4A_IMAGE_FORMAT_COLOR_BGRA32`** (`kinect.py:423`). BGRA32 is
  not a native device mode; at 2048x1536 the camera emits **MJPEG only**, and the
  SDK converts on the host CPU. So JPEG is already in the path — switching to
  `COLOR_MJPG` with lazy decode loses no data at all and drops a constant
  full-rate conversion for both cameras inside the process that runs the libusb
  event thread. #221103 names CPU load as a reproduction condition, so this is
  the second one-line change worth trying.
- **Manual exposure is never set** — `color_control` appears nowhere in `viki/`.
  The multi-camera sync documentation requires manual exposure, because auto
  "causes dynamic timing changes that push cameras out of sync", and it should be
  set once rather than per capture. This is a plausible explanation for the
  timestamp-offset spread recorded in the accuracy audit (median 0.503 ms, P90
  4.5 ms, >10 ms on 4.23% of frames) where a hard trigger should give a near
  constant ~160 us: the Kinect timestamp is the centre of exposure, so a floating
  exposure moves it even under a perfect trigger.
- `timeout_ms` defaults to **1000** (`kinect.py:327`) while its docstring says
  5000 (`kinect.py:318`) — tight for a subordinate waiting on the master's first
  pulses.
- `depth_delay_off_color_usec=0` is hardcoded (`kinect.py:428`); fine for two
  cameras, needs to move into config for a third.
- **Correction (2026-09-20 22:40): the ENOMEM dismissal above was wrong.**
  `scripts/host_setup.sh` persisted the limit through
  `/etc/modprobe.d/viki-usbfs.conf`, but **`usbcore` is built into the Ubuntu
  kernel, not a module**, so that file is inert — only the kernel command line
  sets it. The value was 1000 on this machine solely because the cmdline had
  been edited by hand. A boot without that parameter falls back to the default
  **16 MB**, which is below the documented minimum for two Kinects and is exactly
  the ENOMEM-on-URB-submit condition the script's own comment warns about.
  Verified live in the 22:37 recovery-mode boot: `usbfs_memory_mb = 16`.
  `host_setup.sh` now writes the GRUB cmdline and warns when the running kernel
  disagrees. Any two-camera measurement taken while this reads 16 is measuring a
  degraded rig, not the defect under study. The subordinate-before-master start
  order is per documentation.
- Already fixed on this branch: the `K4A_WAIT_RESULT_TIMEOUT` enum. The review
  read `main` (`c39782b`), which still carries the old value; `3894e71` is not
  pushed there.

### Operating discipline, independent of the cause

- **Take camera lifecycle out of the web UI.** Every HTTP start/stop cancels
  isochronous URBs, and a recording session does dozens. Open the cameras once
  per session and keep them streaming; let the UI toggle only whether the writer
  persists frames. Same features, one teardown instead of many.
- **Make a freeze cheap.** `record.py` keeps `_timestamps` in a RAM list
  (line 69) and writes the manifest at the end (line 284), so a lock-up mid
  episode loses the lot — an mp4 without its moov atom is unreadable. Write
  timestamps as JSONL with periodic flush, write the manifest up front with
  `complete: false` and flip it at the end, and cut episodes into 20-30 s
  segments so a freeze costs one segment rather than the session. (`sensor_meta`
  is already written up front.)

## Controlled run, 2026-09-20 22:43 — the two cheap hypotheses are dead

Same kernel as the four archived dumps, one variable changed. Conditions, all
verified from `/proc/cmdline` and sysfs before the run:

| | |
|---|---|
| kernel | `7.0.0-28-generic` — the one the dumps came from |
| `usbcore.autosuspend` | **-1** (the #221103 workaround) |
| `usbcore.usbfs_memory_mb` | 1000 |
| controller runtime PM | `auto`, but `runtime_suspended_time = 0` for the whole boot |
| method | **one** open of both cameras, no prior stop, no re-enumeration |

Timeline:

```
22:43:39  POST start rig, 1280x720/30
22:43:41  both active, hardware_sync verified, offset 144 us (expected 160, spread 33)
22:43:41  kernel journal stops
22:43:46  the 1 Hz sampler is still writing, api=200 — userspace outlives the journal
22:45:18  next boot
```

The rig came up **healthy** — a good sync, offset inside tolerance — and the host
was gone about two seconds later. No pstore record. On 6.17 the same step took
45 seconds; here it took two.

### What this eliminates

1. **#221103 / runtime-suspend is not our bug.** Its confirmed workaround was in
   force and did not prevent the lock-up. Independently, the controller never
   entered runtime suspend at any point in that boot
   (`runtime_suspended_time = 0`), so the "xHC resumes too fast, registers read
   0xffffffff" mechanism never had an opportunity to fire. The affected-range
   argument that explained away the 6.17 reproduction is therefore withdrawn:
   both kernels crash, and not for that reason.
2. **URB teardown is not required.** This run contained no stop, no
   re-enumeration and no second open — the whole rig came up in one transaction,
   subordinate then master. The caveat that had stood since the first matrix run
   is now resolved: teardown is not necessary to trigger the failure.

### What survives

**Two Kinects streaming concurrently through one xHCI controller.** It is the
only suspect left standing after the two above were removed, and it is the one
Microsoft's multi-camera guidance addresses directly. Note this does not promote
it to proven: #1905 still reports the same hang across different controllers on
Windows, so a discrete PCIe card is the best-supported next action rather than a
guaranteed cure.

Also worth recording: one camera has now streamed without incident in two
separate boots, and the failure has never once occurred with a single camera.

Caveat, as always: one run per configuration. This is elimination by single
counter-example, which is sound for killing a hypothesis and weak for confirming
one.

## "But recording used to work" — what actually differs (2026-09-20, 23:00)

The operator's objection is fair and worth answering with evidence: two-camera
recording ran fine in early September. Everything that could have changed since
was checked, and almost nothing did.

| | September takes | tonight |
|---|---|---|
| kernel | `7.0.0-28-generic` — running from 2026-09-02 | `7.0.0-28-generic` (and 6.17, which also died) |
| colour | 1280x720 @ 30 | 1280x720 @ 30 |
| depth | 640x576, NFOV_UNBINNED | 640x576, NFOV_UNBINNED |
| cameras | two, HW-synced | two, HW-synced |
| NVIDIA driver | 610.43.02, installed 2026-07-24 | same |
| `web` container GPU env | `gpus: all`, caps `compute,utility` | same until 21:59 tonight |

So: **not a kernel regression, not a settings change, not a driver update.** The
kernel now blamed was already running during the successful recordings. Package
upgrades in the window are routine security updates, nothing touching USB, xHCI
or NVIDIA. The only committed change to `docker-compose.yml` since is an
optional `sam2` tools-profile service, which `up` never starts. The
`NVIDIA_DRIVER_CAPABILITIES` extension to `graphics,display` was made at 21:59
tonight — *after* four of the evening's freezes — so it cannot explain them
either. The one camera-touching commit in the window, `1b399d3`, adds a pure
calibration-maths helper and does not go near the USB start path.

### What is different: nothing ever ran for long

Every recording on disk, measured from the mp4s:

```
2026-09-01  1920x1080  599 frames  ~20 s   2 cameras
2026-09-02  1280x720   610         ~20 s   3 cameras
2026-09-02  1280x720   898         ~30 s   3 cameras
2026-09-03  1280x720   899/898/199/660/898 ~7-30 s   2 cameras
2026-09-04  1280x720   898/897/898 ~30 s   2 cameras
2026-09-09  1280x720   897         ~30 s   2 cameras
```

**The longest two-camera stream ever recorded on this rig is 30 seconds**, and
almost every take sits exactly at that cap. Tonight's 6.17 failure arrived at
**45 seconds** — past the duration of every take ever made here.

So "it used to work" is consistent with a latent fault that short takes simply
outran: record, stop, twenty to thirty seconds at a time, never a long hold. The
difference tonight is the usage pattern, not the configuration — the rig was
held open indefinitely instead of being released after half a minute.

### Where that explanation stops

It does not cover the 7.0 run, which died about **two seconds** after both
cameras came up. A pure duration threshold cannot produce both a 2 s and a 45 s
failure. The honest shape is a probabilistic hazard, concentrated near start-up
and accumulating with time, rather than a fixed cliff — which also fits five
freezes tonight at widely different intervals.

**Practical consequence.** Short takes are not a fix, but they are measurably
what the rig has survived. Until the controller question is settled, keeping
two-camera streaming to sub-30-second bursts matches every successful recording
in the dataset, and the segmented-recording change (still unimplemented, see
above) would cap the loss when a burst does not survive.

## The duration theory is dead too — and that points at hardware

The operator objected that calibration used to run for a long time, and that the
first freezes today came during calibration. Correct on both counts, and the
journal still holds the evidence: it reaches back to 2026-09-02.

**Boot `-20`, 2026-09-09.** Kinects plugged in at 14:35:36, calibration preset
`skrip` written at 14:50, boot continued until 15:27:45. That is roughly fifteen
minutes of two-camera work — a full calibration session — followed by another
half hour of uptime. On `7.0.0-28-generic`, at 1280x720/30, on USB ports `2-1`
and `2-3`.

Tonight the same rig dies in two seconds. So the "short takes outran a latent
fault" explanation is withdrawn: this rig has demonstrably streamed two cameras
for a quarter of an hour.

### Everything software-side is identical

Checked and matching between the working Sept 9 session and tonight:

- kernel `7.0.0-28-generic` — the same boot-to-boot;
- colour 1280x720/30, depth NFOV_UNBINNED;
- USB topology: both Kinects on `2-1` and `2-3`, the very same ports;
- NVIDIA driver 610.43.02 from July;
- `web` container GPU environment, until a change made at 21:59 tonight that
  postdates four of the freezes;
- no camera-code change reaching the USB start path;
- and the `uvcvideo` / pipewire noise is *equally present in both* — 345
  uvcvideo lines and 280 `spa.v4l2` errors during the successful Sept 9 session.
  That is now a third independent reason to consider that hypothesis closed.

### What that leaves

No software or configuration variable distinguishes a rig that calibrated for
fifteen minutes from one that dies in two seconds. What remains is the physical
layer degrading between 2026-09-09 and 2026-09-20: a cable, a connector, a hub,
a camera's external power supply, the PSU, or the controller itself. The
operator's instinct that this "is not natural" is the right reading of it.

### The next test is cheap and decisive

Run **each camera alone** for fifteen minutes, one after the other, at
1280x720/30:

- one of them kills the host → that camera, its cable or its PSU is the fault,
  and it is identified without buying anything;
- both survive alone but the pair dies → the shared path (controller, hub, or
  total power draw) is implicated, and the PCIe card becomes the justified buy;
- both survive alone *and* together → the fault is intermittent and the search
  moves to what differs on the failing runs.

Single-camera operation has never once failed here, across every boot tested, so
this test is also the low-risk one.

## Reduced load buys time, it does not fix it (2026-09-20 23:10)

Last run of the evening. Both cameras, one open, no prior stop, on
`7.0.0-28-generic` with `usbcore.autosuspend=-1`:

- colour **MJPG** 1280x720 (what the device streams anyway — see the
  image-format constant fix)
- **15 fps** instead of 30
- depth **NFOV_2X2BINNED** (320x288) instead of NFOV_UNBINNED (640x576)

Roughly a fivefold cut in isochronous bandwidth. The rig came up with sync
verified and an offset of 100 us. Started 23:10:37, journal ended 23:14:14 —
**3.5 minutes**, against 2 seconds at full rate.

So load is a *modulator*, not the cause: less of it postpones the lock-up
proportionally without removing it. That is consistent with everything else —
one camera never fails, two always eventually do.

### Decision

Two Kinects are retired on this host. The rig moves to **one Kinect plus a
RealSense D435i** for the second view, which loses hardware sync on that pair
but keeps the machine alive. The limitation is recorded in `README.md`,
`SETUP_GUIDE.md` and `bugs.md` so it is not rediscovered. A PCIe USB card with a
dedicated controller per port remains the untested candidate fix.
