"""Resume the reviewed cube-scene, object-aware retarget and relative-track batch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "batch_extract_2026-10-09.json"
PROMPTS = DATA / "2026-10-06_08-55-49-cube-prompts.json"
REPORT = DATA / "batch_cube_pipeline_2026-10-09.jsonl"
LOGS = DATA / "batch_cube_pipeline_logs"
MODEL = Path("intermediates/segmentation/sam2.1_hiera_small/object_models.npz")
EXPECTED_COLOR = {2: "blue", 3: "red", 4: "green"}


def record(episode: str, stage: str, status: str, **details: object) -> None:
    event = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "episode": episode,
             "stage": stage, "status": status, **details}
    with REPORT.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def first_rgb(video: Path) -> np.ndarray:
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True,
    )
    if len(completed.stdout) != 1920 * 1080 * 3:
        raise ValueError(f"unexpected first-frame size for {video}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(1080, 1920, 3)


def matches_color(rgb: np.ndarray, color: str) -> bool:
    red, green, blue = [float(value) for value in rgb]
    return {
        "blue": blue > 60 and blue > 1.5 * green and blue > 1.5 * red,
        "red": red > 60 and red > 1.8 * green and red > 1.8 * blue,
        "green": green > 20 and green > 2 * red and green > 1.15 * blue,
    }[color]


def prompt_colors_match(episode: Path, prompts: dict) -> tuple[bool, dict]:
    samples = {}
    for camera, annotations in prompts["cameras"].items():
        frame = first_rgb(episode / "raw" / f"{camera}.mp4")
        for annotation in annotations:
            object_id = annotation["object_id"]
            if object_id not in EXPECTED_COLOR:
                continue
            x_pixel, y_pixel = annotation["points"][0]
            patch = frame[y_pixel - 5:y_pixel + 6, x_pixel - 5:x_pixel + 6]
            median = np.median(patch, axis=(0, 1))
            samples[f"{camera}:{object_id}"] = {
                "rgb": median.astype(int).tolist(),
                "match": matches_color(median, EXPECTED_COLOR[object_id]),
            }
    return all(sample["match"] for sample in samples.values()), samples


def scene_quality(model_path: Path) -> dict:
    with np.load(model_path, allow_pickle=False) as model:
        ids = model["object_ids"].tolist()
        labels = model["object_labels"].tolist()
        blue_indices = [index for index, label in enumerate(labels)
                        if label == "manipulated_object"]
        if len(blue_indices) != 1 or len(ids) != 3:
            return {"accepted": False, "reason": "wrong object labels/count"}
        blue_index = blue_indices[0]
        confidence = model["track_confidence"][:, blue_index]
        positions = model["translation_world"][:, blue_index]
        travel_m = float(np.linalg.norm(positions[-1] - positions[0]))
        median_confidence = float(np.nanmedian(confidence))
        p10_confidence = float(np.nanpercentile(confidence, 10))
        return {
            "accepted": bool(
                np.isfinite(positions).all()
                and 0.2 <= travel_m <= 0.6
                and median_confidence >= 0.65
                and p10_confidence >= 0.5
            ),
            "blue_travel_m": travel_m,
            "blue_confidence_median": median_confidence,
            "blue_confidence_p10": p10_confidence,
        }


def plan_quality(plan_path: Path) -> dict:
    with h5py.File(plan_path) as plan:
        metrics = json.loads(plan["metrics_json"][()].decode())
        status = plan["solver_status"][()].decode()
        error_mm = np.asarray(plan["position_error_m"], dtype=float) * 1000.0
    close_start, carry_start, release_start, open_end = (
        metrics["object_grasp"]["phase_bounds"]
    )
    if not 0 <= close_start < carry_start < release_start < open_end <= len(error_mm):
        raise ValueError("invalid cube-grasp phase bounds in plan")
    critical_error = error_mm[close_start:open_end]
    position_rmse_mm = float(metrics["position_rmse_mm"])
    position_p95_mm = float(metrics["position_p95_mm"])
    critical_rmse_mm = float(np.sqrt(np.mean(critical_error ** 2)))
    critical_p95_mm = float(np.percentile(critical_error, 95))
    return {
        "accepted": bool(
            status == "converged"
            and np.isfinite(position_rmse_mm)
            and np.isfinite(position_p95_mm)
            and np.isfinite(critical_rmse_mm)
            and np.isfinite(critical_p95_mm)
            and position_rmse_mm <= 20.0
            and critical_rmse_mm <= 15.0
            and critical_p95_mm <= 20.0
        ),
        "solver_status": status,
        "position_rmse_mm": position_rmse_mm,
        "position_p95_mm": position_p95_mm,
        "critical_position_rmse_mm": critical_rmse_mm,
        "critical_position_p95_mm": critical_p95_mm,
    }


def run_stage(episode: Path, stage: str, service: str, *args: str) -> bool:
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{episode.name}-{stage}.log"
    command = ["docker", "compose", "run", "--rm", service, *args]
    record(episode.name, stage, "running", log=str(log.relative_to(ROOT)))
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    record(episode.name, stage, "done" if result.returncode == 0 else "error",
           returncode=result.returncode, log=str(log.relative_to(ROOT)))
    return result.returncode == 0


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    prompts = json.loads(PROMPTS.read_text())
    accepted_extract = set()
    for relative_episode in manifest["episodes"]:
        episode = ROOT / relative_episode
        status_path = episode / "status.json"
        if status_path.is_file():
            stages = json.loads(status_path.read_text()).get("stages", {})
            if stages.get("extract", {}).get("done") and stages.get("prepare", {}).get("done"):
                accepted_extract.add(episode.name)
    for relative_episode in manifest["episodes"]:
        episode = ROOT / relative_episode
        if episode.name not in accepted_extract:
            record(episode.name, "extract", "skipped", reason="quality gate or unfinished")
            continue
        if shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
            record(episode.name, "batch", "stopped", reason="less than 8 GiB free")
            break
        try:
            colors_ok, samples = prompt_colors_match(episode, prompts)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            record(episode.name, "prompt_preflight", "needs_review", error=str(error))
            continue
        if not colors_ok:
            record(episode.name, "prompt_preflight", "needs_review", samples=samples)
            continue
        model_path = episode / MODEL
        if not model_path.is_file():
            if not run_stage(episode, "scene", "sam2", "scene", relative_episode,
                             "--prompts", str(PROMPTS.relative_to(ROOT)), "--objects", "3"):
                continue
        try:
            quality = scene_quality(model_path)
        except (OSError, ValueError, KeyError) as error:
            record(episode.name, "scene_quality", "needs_review", error=str(error))
            continue
        record(episode.name, "scene_quality",
               "accepted" if quality["accepted"] else "needs_review", **quality)
        if not quality["accepted"]:
            continue
        if not (episode / "plan.h5").is_file():
            if not run_stage(episode, "retarget", "cli", "retarget", relative_episode,
                             "--robot", "ur10"):
                continue
        try:
            quality = plan_quality(episode / "plan.h5")
        except (OSError, ValueError, KeyError) as error:
            record(episode.name, "plan_quality", "needs_review", error=str(error))
            continue
        record(episode.name, "plan_quality",
               "accepted" if quality["accepted"] else "needs_review", **quality)
        if not quality["accepted"]:
            continue
        relative_path = episode / "intermediates/object_relative.npz"
        if not relative_path.is_file():
            run_stage(episode, "object_relative", "cli", "object-relative", relative_episode)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record("batch", "driver", "error", error=repr(error))
        sys.exit(1)
