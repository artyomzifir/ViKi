#!/usr/bin/env python3
"""
Record a hardware-synced Kinect pair with the vendor binary, one process each.

ViKi's own capture path hard-locks this host when both Kinects stream through a
single libusb context (``bugs.md`` row 1). ``k4arecorder`` is the documented
multi-device route: one process per camera, separate libusb contexts, MJPEG
written straight to MKV with no BGRA32 conversion, and a single teardown at the
end instead of dozens of HTTP start/stops.

Ordering is not a preference. The subordinate must be streaming and waiting
before the master starts emitting pulses, or it never sees a trigger — so
subordinates go first and the master goes last, per the vendor's own guidance.

    docker compose run --rm terminal python3 scripts/record_k4a_pair.py \
        --seconds 20 --out data/k4a_raw/<name>

Roles come from ``KINECT_SYNC`` in the configuration, so this and the ViKi
recorder cannot disagree about which camera is the master.
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
    """(master_index, subordinate_indices, subordinate_delay_us) from config."""
    from viki import config

    sync = getattr(config, "KINECT_SYNC", {}) or {}
    master = sync.get("master")
    subs = list(sync.get("subordinates") or [])
    if not master or not subs:
        raise SystemExit(
            "KINECT_SYNC has no master/subordinates — this script only drives a "
            "wired pair. Configure it, or record the single camera through ViKi."
        )

    def idx(dev: str) -> int:
        if not str(dev).startswith("kinect_"):
            raise SystemExit(f"not a Kinect device id: {dev!r}")
        return int(str(dev).split("_", 1)[1])

    return idx(master), [idx(d) for d in subs], int(sync.get("subordinate_delay_us", 160))


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
