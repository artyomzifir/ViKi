"""Retarget newly recovered cube tracks in isolated, resumable episodes."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from threading import Lock

from scripts.batch_cube_scene_retarget import plan_quality, scene_quality


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPARISON = DATA / "experiments/track-recovery-comparison.json"
MODELS = DATA / "experiments/track-recovery-full"
OUTPUT = DATA / "experiments/recovered-retarget"
DATASET = DATA / "datasets/40mm-pnp"
MODEL = Path("intermediates/segmentation/sam2.1_hiera_small/object_models.npz")
RECORD_LOCK = Lock()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"unexpected existing link: {destination}")
    elif destination.exists():
        raise ValueError(f"scratch input is not a symlink: {destination}")
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def record(episode: str, status: str, **details: object) -> None:
    event = {"episode": episode, "status": status, **details}
    line = json.dumps(event, sort_keys=True)
    with RECORD_LOCK:
        with (OUTPUT / "status.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def process(row: dict) -> str:
    name = row["episode"]
    source = DATASET / name
    scratch = OUTPUT / name
    link(source / "cln.npz", scratch / "cln.npz")
    link(source / "raw", scratch / "raw")
    link(MODELS / f"{name}.npz", scratch / MODEL)
    quality = scene_quality(scratch / MODEL)
    if not quality["accepted"]:
        record(name, "model_rejected", quality=quality)
        return "model_rejected"
    plan = scratch / "plan.h5"
    if not plan.is_file():
        log = OUTPUT / f"{name}-retarget.log"
        with log.open("w") as stream:
            result = subprocess.run(
                ["viki", "retarget", str(scratch.relative_to(ROOT)),
                 "--robot", "ur10"],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            record(name, "retarget_failed", returncode=result.returncode,
                   log=str(log.relative_to(ROOT)))
            return "retarget_failed"
    quality = plan_quality(plan)
    status = "accepted" if quality["accepted"] else "plan_rejected"
    record(name, status, quality=quality)
    return status


def main() -> None:
    comparison = json.loads(COMPARISON.read_text())
    selected = [row for row in comparison["episodes"]
                if row["before"] != "accepted" and row["after"] == "accepted"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        counts: Counter[str] = Counter(pool.map(process, selected))
    summary = {"selected": len(selected), "counts": dict(counts)}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
