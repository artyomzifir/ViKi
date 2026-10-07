#!/usr/bin/env python3
"""
Record a hardware-synced Kinect pair with the vendor binary, one process each.

The current rig gives each Kinect an independent USB host controller;
ViKi's two-camera capture works on that wiring. This vendor recorder is a
raw-MKV diagnostic alternative, not a workaround for the old shared-controller
host freeze. It uses one process per camera and writes MJPEG straight to MKV.

Ordering is not a preference. The subordinate must be streaming and waiting
before the master starts emitting pulses, or it never sees a trigger — so
subordinates go first and the master goes last, per the vendor's own guidance.

    docker compose run --rm terminal python3 scripts/record_k4a_pair.py \
        --seconds 20 --out data/k4a_raw/<name>

Roles come from the physical sync jacks, because device indices change on
replug; the subordinate delay comes from ``KINECT_SYNC``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RECORDER = "k4arecorder"


def _roles() -> tuple[int, list[int], int]:
    """(master_index, subordinate_indices, subordinate_delay_us).

    Roles come from the **sync jacks**, not from configuration. Device indices
    follow USB enumeration order and change on every replug, so a config that
    named ``kinect_1`` as master is describing yesterday's enumeration. The
    cable is the ground truth: the camera with something in SYNC OUT drives, the
    one with something in SYNC IN follows. Getting this backwards costs a whole
    run — the subordinate refuses to start with "failure to detect presence of
    sync in cable".
    """
    import ctypes

    from viki.cameras.kinect import K4ADevice, _load_libk4a

    lib = _load_libk4a()
    lib.k4a_device_open.argtypes = [ctypes.c_uint32, ctypes.POINTER(K4ADevice)]
    lib.k4a_device_get_sync_jack.argtypes = [
        K4ADevice, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)
    ]

    masters, subs = [], []
    for i in range(int(lib.k4a_device_get_installed_count())):
        h = K4ADevice(None)
        if lib.k4a_device_open(ctypes.c_uint32(i), ctypes.byref(h)) != 0:
            raise SystemExit(
                f"device index {i} will not open — after a host freeze the Kinects "
                "stay wedged until USB *and* mains power are pulled"
            )
        sin, sout = ctypes.c_bool(), ctypes.c_bool()
        lib.k4a_device_get_sync_jack(h, ctypes.byref(sin), ctypes.byref(sout))
        lib.k4a_device_close(h)
        if sin.value:
            subs.append(i)
        elif sout.value:
            masters.append(i)

    if len(masters) != 1 or not subs:
        raise SystemExit(
            f"sync cable is not wired for a pair: SYNC OUT on {masters}, "
            f"SYNC IN on {subs}. Exactly one camera must drive (SYNC OUT) and at "
            "least one follow (SYNC IN)."
        )

    from viki import config

    delay = int((getattr(config, "KINECT_SYNC", {}) or {}).get("subordinate_delay_us", 160))
    return masters[0], subs, delay


def _cmd(index: int, path: Path, role: str, args, delay_us: int) -> list[str]:
    cmd = [
        RECORDER,
        "--device", str(index),
        "-l", str(int(args.seconds)),
        "-c", args.color_mode,
        "-d", args.depth_mode,
        "-r", str(int(args.rate)),
        "--imu", "OFF",
        "--external-sync", role,
    ]
    if role == "Subordinate":
        cmd += ["--sync-delay", str(int(delay_us))]
    if args.exposure_us:
        cmd += ["-e", str(int(args.exposure_us))]
    return cmd + [str(path)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="directory for the .mkv files")
    ap.add_argument("--color-mode", default="720p")
    ap.add_argument("--depth-mode", default="NFOV_UNBINNED")
    ap.add_argument("--rate", type=int, default=30, choices=(5, 15, 30))
    ap.add_argument("--exposure-us", type=int, default=None,
                    help="manual exposure; auto exposure walks the inter-camera offset")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to let the subordinate reach its wait state")
    args = ap.parse_args()

    master, subs, delay_us = _roles()
    out = Path(args.out or f"data/k4a_raw/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    started = time.time()
    try:
        for i in subs:                       # subordinates first — they wait for pulses
            p = out / f"kinect_{i}.mkv"
            cmd = _cmd(i, p, "Subordinate", args, delay_us)
            print("[sub ] " + " ".join(cmd), flush=True)
            procs.append((f"kinect_{i}", subprocess.Popen(cmd)))
        time.sleep(args.settle)

        p = out / f"kinect_{master}.mkv"     # master last — it drives the trigger
        cmd = _cmd(master, p, "Master", args, delay_us)
        print("[mstr] " + " ".join(cmd), flush=True)
        procs.append((f"kinect_{master}", subprocess.Popen(cmd)))

        rc = {name: proc.wait() for name, proc in procs}
    except KeyboardInterrupt:
        print("interrupted — stopping recorders", flush=True)
        for _, proc in procs:
            proc.send_signal(signal.SIGINT)
        rc = {name: proc.wait() for name, proc in procs}

    print(f"\nelapsed {time.time() - started:.1f}s -> {out}", flush=True)
    bad = {n: c for n, c in rc.items() if c != 0}
    for f in sorted(out.glob("*.mkv")):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
    if bad:
        print("non-zero exits:", bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
