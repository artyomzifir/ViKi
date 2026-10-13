#!/usr/bin/env python3
"""
tools/hocap_download.py
-----------------------
Fetch ONE HO-Cap archive (a subject, or the shared ``calibration`` / ``models``
/ ``poses`` / ``labels`` bundle) into ``data/raw_datasets/hocap/`` and verify
the downloaded size against the server's ``Content-Length``.

HO-Cap — arXiv:2406.06843, https://irvlutd.github.io/HOCap — is **CC BY 4.0**.
This is a plain HTTP downloader: stdlib only, resumable, **no HOCap-Toolkit
import**. The archive links below are the public distribution URLs published in
the toolkit's ``config/hocap_recordings.yaml``.

    python tools/hocap_download.py calibration
    python tools/hocap_download.py subject_1
    python tools/hocap_download.py labels --dest data/raw_datasets/hocap

Nothing is unzipped automatically (pass ``--extract`` to unzip in place). The
converter (``viki hocap-import``) then runs against the extracted tree.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Published in IRVLUTD/HO-Cap :: config/hocap_recordings.yaml (CC BY 4.0 data).
HOCAP_ARCHIVES: dict[str, str] = {
    "models": "https://utdallas.box.com/shared/static/con44iqej33weg9f3rpxof61eh3x2x21.zip",
    "calibration": "https://utdallas.box.com/shared/static/nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip",
    "subject_1": "https://utdallas.box.com/shared/static/w0voy9bixtxyclo52841xyamock2lxpt.zip",
    "subject_2": "https://utdallas.box.com/shared/static/j498kxxrkvaf674tvmt4su4ad0bz9s9f.zip",
    "subject_3": "https://utdallas.box.com/shared/static/shklq33yaoozh9gm681nxwnq0o3y0y1d.zip",
    "subject_4": "https://utdallas.box.com/shared/static/dew68k7b3ya09t40818gpfxm95oa4yeq.zip",
    "subject_5": "https://utdallas.box.com/shared/static/mutor2a09kudze1yw173gsfetsru7ces.zip",
    "subject_6": "https://utdallas.box.com/shared/static/iyja7rdbjx2ksgjhmdu6mx3zqvaurdni.zip",
    "subject_7": "https://utdallas.box.com/shared/static/4g5qyig6i4uz1rgrzkcu9n4mhdjs74m2.zip",
    "subject_8": "https://utdallas.box.com/shared/static/khrb5guy8rdwnoqi4euk2w0mk5lslxkn.zip",
    "subject_9": "https://utdallas.box.com/shared/static/3x5yitydmbmwolq9bty5dd2udu5v52fc.zip",
    "poses": "https://utdallas.box.com/shared/static/2lofbp2yd005d8o213ns77mdrtxg8eep.zip",
    "labels": "https://utdallas.box.com/shared/static/ayd4st2wo588z2yqbuxalptxnz2qxlj5.zip",
}

_CHUNK = 4 * 1024 * 1024


def _remote_size(url: str) -> int | None:
    """Total archive size in bytes. Tries HEAD, then a 1-byte range GET whose
    ``Content-Range: bytes 0-0/<total>`` gives the size (Box's shared/static
    links 404 on HEAD but honour range requests)."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as r:
            cl = r.headers.get("Content-Length")
            if cl is not None:
                return int(cl)
    except Exception:  # noqa: BLE001
        pass
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=90
        ) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                total = cr.rsplit("/", 1)[-1].strip()
                return int(total) if total.isdigit() else None
    except Exception:  # noqa: BLE001
        return None
    return None


def download(name: str, dest_dir: Path, *, extract: bool = False) -> Path:
    if name not in HOCAP_ARCHIVES:
        raise SystemExit(
            f"unknown archive {name!r}; choose from: {', '.join(HOCAP_ARCHIVES)}"
        )
    url = HOCAP_ARCHIVES[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}.zip"

    expected = _remote_size(url)
    have = out.stat().st_size if out.exists() else 0
    headers = {}
    mode = "wb"
    if have and expected and have < expected:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"resuming {name}.zip at {have / 1e6:.1f} MB / {expected / 1e6:.1f} MB")
    elif have and expected and have == expected:
        print(f"{out} already complete ({have} bytes)")
        if extract:
            _unzip(out, dest_dir)
        return out

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(out, mode) as fh:
        got = have
        total = expected or (int(r.headers.get("Content-Length", 0)) + have)
        while True:
            chunk = r.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                pct = 100.0 * got / total
                print(f"\r  {name}.zip  {got / 1e6:8.1f} / {total / 1e6:8.1f} MB "
                      f"({pct:5.1f}%)", end="", flush=True)
    print()

    final = out.stat().st_size
    if expected is not None and final != expected:
        raise SystemExit(
            f"size mismatch for {out}: got {final} bytes, server reported {expected}. "
            "Re-run to resume."
        )
    print(f"OK  {out}  ({final} bytes"
          + (f", matches Content-Length {expected}" if expected is not None
             else ", server gave no Content-Length") + ")")

    if extract:
        _unzip(out, dest_dir)
    return out


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    print(f"extracting {zip_path.name} -> {dest_dir}/ ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print("done")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    p.add_argument("archive", choices=sorted(HOCAP_ARCHIVES),
                   help="which HO-Cap archive to fetch")
    p.add_argument("--dest", default="data/raw_datasets/hocap",
                   help="download directory (default: data/raw_datasets/hocap)")
    p.add_argument("--extract", action="store_true", help="unzip in place after download")
    a = p.parse_args(argv)
    download(a.archive, Path(a.dest), extract=a.extract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
