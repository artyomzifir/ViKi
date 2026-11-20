"""Resume pinned HO-Cap pick-and-place archive downloads and verify hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


PLAN_PATH = Path(__file__).with_name("hocap_pnp_plan.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text())

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with (root / "download.log").open("a") as stream:
            stream.write(line + "\n")

    for index, item in enumerate(plan["files"], 1):
        destination = root / item["name"]
        partial = root / (item["name"] + ".part")
        expected = int(item["size"])
        if destination.exists():
            if destination.stat().st_size == expected and sha256(destination) == item["sha256"]:
                log(f"[{index}/{len(plan['files'])}] verified {destination.name}")
                continue
            raise RuntimeError(f"Existing file has wrong size or hash: {destination}")
        if partial.exists() and partial.stat().st_size > expected:
            raise RuntimeError(f"Partial file exceeds expected size: {partial}")
        remaining = expected - (partial.stat().st_size if partial.exists() else 0)
        if shutil.disk_usage(root).free < remaining + 8_000_000_000:
            raise RuntimeError(f"Insufficient disk reserve before {destination.name}")
        url = (
            "https://huggingface.co/datasets/JWRoboticsVision/HO-Cap-Dataset/resolve/"
            + plan["revision"] + "/" + item["name"]
        )
        log(f"[{index}/{len(plan['files'])}] downloading {destination.name}")
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
             "--retry-delay", "10", "--connect-timeout", "30", "--speed-time", "60",
             "--speed-limit", "1024", "--continue-at", "-", "--silent",
             "--show-error", "--output", str(partial), url],
            check=True,
        )
        if partial.stat().st_size != expected or sha256(partial) != item["sha256"]:
            raise RuntimeError(f"Downloaded file failed size/hash verification: {partial}")
        partial.replace(destination)
        log(f"[{index}/{len(plan['files'])}] complete {destination.name}")
    log("ALL DOWNLOADS COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path.home() / "Downloads/HO-Cap-PnP-raw"
    )
    download(parser.parse_args().out.expanduser())
