#!/usr/bin/env python3
"""
Measure rig time-sync from recorded episodes (triangulation task, Stage 0).

Reads every ``data/datasets/**/raw/timestamps.json`` and reports, per device,
the frame-offset-from-tick distribution and — the number that matters for
multi-view triangulation — the INTER-CAMERA gap per synced group
(``|offset_a - offset_b|``), which maps to a spatial error ``dx ~= v * dt``.

Run after wiring the Kinect hardware sync and re-recording; expect the
inter-camera P95 to fall from ~20-30 ms (software-only) to single-digit ms.

    docker compose run --rm terminal python3 scripts/measure_sync.py
"""
from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np

GEOM = ("kinect_0", "kinect_1")
WINDOWS_MS = (2.0, 5.0, 16.7)


def _p(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def analyse(ep: Path) -> dict | None:
    rows = json.loads((ep / "raw" / "timestamps.json").read_text())
    if not rows:
        return None
    meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}
    devs = sorted({d for r in rows for d in (r.get("offsets_us") or {})})
    off = {d: np.array([r["offsets_us"].get(d, np.nan) for r in rows], float) for d in devs}
    out = {"episode": ep.name, "task": meta.get("task"), "n": len(rows),
           "kinect_sync": "wired" if (meta.get("kinect_sync") or {}) else "software-only",
           "per_device": {}, "pairs": {}}
    for d in devs:
        o = off[d][np.isfinite(off[d])]
        out["per_device"][d] = {
            "mean_ms": round(o.mean() / 1e3, 2),
            "p95_abs_ms": round(_p(np.abs(o), 95) / 1e3, 2),
            "max_abs_ms": round(float(np.abs(o).max()) / 1e3, 2),
        }
    for a, b in combinations(devs, 2):
        m = np.isfinite(off[a]) & np.isfinite(off[b])
        gap = np.abs(off[a][m] - off[b][m])
        if not gap.size:
            continue
        out["pairs"][f"{a}|{b}"] = {
            "median_ms": round(float(np.median(gap)) / 1e3, 2),
            "p95_ms": round(_p(gap, 95) / 1e3, 2),
            "max_ms": round(float(gap.max()) / 1e3, 2),
            **{f"frac_gt_{w}ms": round(float((gap > w * 1e3).mean()), 3) for w in WINDOWS_MS},
        }
    return out


def main() -> None:
    eps = sorted(Path(p).parent.parent
                 for p in glob.glob("data/datasets/**/raw/timestamps.json", recursive=True))
    reports = [r for r in (analyse(e) for e in eps) if r]
    print(json.dumps(reports, indent=2))
    print("\nSUMMARY — inter-camera gap for the geometry pair")
    for r in reports:
        key = next((k for k in r["pairs"] if set(k.split("|")) == set(GEOM)), None) \
            or next(iter(r["pairs"]), None)
        if not key:
            continue
        p = r["pairs"][key]
        print(f"  {r['episode']:24s} {key:20s} median {p['median_ms']:6.2f}  "
              f"P95 {p['p95_ms']:6.2f}  max {p['max_ms']:6.2f} ms  ({r['kinect_sync']})")


if __name__ == "__main__":
    main()
