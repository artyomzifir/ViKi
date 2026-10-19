"""Run the HO-Cap PnP hand-pose benchmark from flat mirror archives.

Stage one sequence at a time. Detect eight views once, then reuse the selected
per-camera observations for the two-view comparison without a second inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from viki.benchmark.evaluate import evaluate_episode
from viki.benchmark.hocap import import_sequence, render_report
from viki.contracts import Episode
from viki.episode import mark_stage
from viki.perception.extract import extract_episode
from viki.perception.observations import calib_id
from viki.perception.profiles import HOCAP_MEDIAPIPE_TRI_V1, HOCAP_STABLE_FUSED_HAND_V4
from viki.prepare.run import prepare_episode

LOGGER = logging.getLogger(__name__)
MODEL = "mediapipe"
PAIR_BY_HAND = {
    "right": ("043422252387", "105322251225"),
    "left": ("105322251564", "037522251142"),
}
ALL_CAMERAS = (
    "105322251564", "043422252387", "037522251142", "105322251225",
    "108222250342", "117222250549", "046122250168", "115422250549",
)
FRAME_FILE = re.compile(
    r"^(?P<serial>\d{12})-(?P<frame>\d{6})\."
    r"(?P<kind>color\.jpg|depth\.png|label\.npy)$"
)


def _archive_file(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"unreadable archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def stage_sequence(archives: Path, sequence_id: str, target: Path,
                   cameras: tuple[str, ...], max_frames: int | None = None) -> tuple[str, str]:
    """Map the mirror's flat tar naming to the importer's directory schema."""
    subject, sequence = sequence_id.split("-", 1)
    selected = set(cameras)
    source_archive = archives / f"{sequence_id}.tar"
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    calib = target / "calibration"
    with tarfile.open(archives / "calibration.tar") as source:
        for member in source:
            if not member.isfile():
                continue
            if member.name.startswith("intrinsics.") and member.name.endswith(".yaml"):
                destination = calib / "intrinsics" / member.name.removeprefix("intrinsics.")
            elif member.name.startswith("extrinsics.") and member.name.endswith(".yaml"):
                destination = calib / "extrinsics" / member.name.removeprefix("extrinsics.")
            else:
                continue
            _archive_file(source, member, destination)
    with tarfile.open(archives / "sequence_labels.tar") as source:
        name = f"{sequence_id}.meta.yaml"
        _archive_file(source, source.getmember(name), target / subject / sequence / "meta.yaml")

    counts = {"color.jpg": 0, "depth.png": 0, "label.npy": 0}
    with tarfile.open(source_archive) as source:
        for member in source:
            if not member.isfile():
                continue
            match = FRAME_FILE.fullmatch(member.name)
            if match is None:
                continue
            serial, frame, kind = match.group("serial", "frame", "kind")
            if max_frames is not None and int(frame) >= max_frames:
                continue
            if kind != "label.npy" and serial not in selected:
                continue
            suffix = {"color.jpg": "color", "depth.png": "depth", "label.npy": "label"}[kind]
            extension = {"color.jpg": "jpg", "depth.png": "png", "label.npy": "npz"}[kind]
            destination = target / subject / sequence / serial / f"{suffix}_{frame}.{extension}"
            _archive_file(source, member, destination)
            counts[kind] += 1
    if min(counts.values()) == 0:
        raise ValueError(f"incomplete archive {sequence_id}: {counts}")
    LOGGER.info("staged %s: %s", sequence_id, counts)
    return subject, sequence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def _discard_media(episode_dir: Path) -> None:
    raw = episode_dir / "raw"
    for video in raw.glob("*.mp4"):
        video.unlink()
    for depth in raw.glob("*_depth"):
        shutil.rmtree(depth)


def reuse_camera_tracks(source: Episode, target: Episode,
                        cameras: tuple[str, ...], profile: str = HOCAP_STABLE_FUSED_HAND_V4) -> None:
    """Preserve exact detector/depth outputs, filtering only row camera IDs."""
    selected = set(cameras)
    with np.load(source.rec_npz, allow_pickle=False) as data:
        row_count = len(data["device_ids"])
        mask = np.isin(data["device_ids"], cameras)
        rec_count = int(mask.sum())
        payload = {
            key: data[key][mask] if data[key].ndim and len(data[key]) == row_count
            and key != "landmark_ids" else data[key]
            for key in data.files
        }
    if not set(payload["device_ids"]).issubset(selected):
        raise ValueError("filtered rec contains an unselected camera")
    np.savez_compressed(target.rec_npz, **payload)

    source_raw, target_raw = source.raw_dir, target.raw_dir
    with np.load(source_raw / "observations.npz", allow_pickle=False) as data:
        row_count = len(data["camera_id"])
        mask = np.isin(data["camera_id"], cameras)
        observations = {
            key: data[key][mask] if data[key].ndim and len(data[key]) == row_count
            else data[key]
            for key in data.files
        }
    np.savez(target_raw / "observations.npz", **observations)
    meta = json.loads((source_raw / "observations_meta.json").read_text())
    meta["cameras"] = {camera: meta["cameras"][camera] for camera in cameras
                       if camera in meta["cameras"]}
    fingerprint = calib_id(target_raw)
    for camera in meta["cameras"]:
        meta["cameras"][camera]["calib_id"] = fingerprint
    _write_json(target_raw / "observations_meta.json", meta)
    mark_stage(target, "extract", frames=rec_count, model=MODEL,
               track_lm="all", profile=profile,
               reused_from=str(source.root))


def run_batch(archives: Path, plan_path: Path, out_root: Path, *,
              max_frames: int | None = None, limit: int | None = None,
              views: tuple[str, ...] = ("eight", "pair"),
              retain_media: bool = False,
              profile: str = HOCAP_STABLE_FUSED_HAND_V4) -> None:
    if "pair" in views and "eight" not in views:
        raise ValueError("two-view reuse requires the eight-view run")
    LOGGER.info("benchmark model=%s device=cpu profile=%s", MODEL, profile)
    plan = json.loads(plan_path.read_text())
    sequences = plan["sequences"][:limit]
    status_path = out_root / "batch_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {
        "dataset": plan["dataset"], "revision": plan["revision"],
        "model": MODEL, "inference_device": "cpu", "profile": profile,
        "pair_by_hand": {hand: list(pair) for hand, pair in PAIR_BY_HAND.items()},
        "sequences": {},
    }
    if status["model"] != MODEL or status["profile"] != profile:
        raise ValueError("batch_status.json belongs to a different model/profile")
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root = out_root.parent / "_hocap_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    for index, item in enumerate(sequences, 1):
        sequence_id = item["id"]
        record = status["sequences"].setdefault(sequence_id, {})
        pending = [
            view for view in views
            if record.get(view, {}).get("status") != "complete"
            or not (out_root / record[view]["episode_name"] / "gt/evaluation.json").is_file()
        ]
        if not pending:
            LOGGER.info("[%s/%s] already complete: %s", index, len(sequences), sequence_id)
            continue
        if shutil.disk_usage(out_root).free < 10_000_000_000:
            raise RuntimeError("less than 10 GB free; stopping before staging next sequence")
        LOGGER.info("[%s/%s] %s: %s", index, len(sequences), sequence_id, pending)
        with tempfile.TemporaryDirectory(prefix=f"{sequence_id}-", dir=staging_root) as staging:
            root = Path(staging)
            subject, sequence = stage_sequence(
                archives, sequence_id, root, ALL_CAMERAS, max_frames=max_frames,
            )
            import yaml
            meta_path = root / subject / sequence / "meta.yaml"
            serials = tuple(str(s) for s in yaml.safe_load(meta_path.read_text())["realsense"]["serials"])
            if set(serials) != set(ALL_CAMERAS):
                raise ValueError(f"unexpected camera set in {sequence_id}: {serials}")
            for view in ("eight", "pair"):
                if view not in pending:
                    continue
                cameras = serials if view == "eight" else PAIR_BY_HAND[item["hand"]]
                started = time.monotonic()
                episode_dir: Path | None = None
                try:
                    result = import_sequence(
                        root, subject, sequence, out_root=out_root,
                        cameras=cameras, max_frames=max_frames,
                        write_media=(view == "eight"),
                    )
                    if result.skipped:
                        raise RuntimeError(result.skipped[0]["reason"])
                    episode_dir = result.episode_dir
                    render_report([result], out_path=episode_dir / "hocap_import.md")
                    episode = Episode(root=episode_dir)
                    actual_hand = json.loads((episode_dir / "meta.json").read_text())["hand"]
                    if actual_hand != item["hand"]:
                        raise ValueError(f"plan hand {item['hand']} != label hand {actual_hand}")
                    if view == "eight":
                        last: dict[str, int] = {}

                        def progress(stage: str, camera: str, frame: int, total: int) -> None:
                            if frame >= total or frame // 100 > last.get(camera, -1):
                                LOGGER.info("%s %s %s %s/%s", sequence_id, stage,
                                            camera, frame, total)
                                last[camera] = frame // 100

                        extract_episode(episode, model=MODEL, hand=actual_hand,
                                        profile=profile, report=progress)
                    else:
                        eight_record = record.get("eight", {})
                        if eight_record.get("status") != "complete":
                            raise RuntimeError("eight-view inference did not complete")
                        reuse_camera_tracks(
                            Episode(root=out_root / eight_record["episode_name"]),
                            episode, tuple(cameras), profile,
                        )
                    prepare_episode(episode, profile=profile)
                    score = evaluate_episode(episode_dir, expected_profile=(
                        profile if profile == HOCAP_STABLE_FUSED_HAND_V4 else None
                    ))
                    _write_json(episode_dir / "gt/evaluation.json", score)
                    score_key = ("hand_fit_joints" if profile == HOCAP_STABLE_FUSED_HAND_V4
                                 else "smoothed_observed")
                    measured = score.get("prepare", {}).get(score_key, {})
                    coverage = measured.get("coverage")
                    mpjpe = measured.get("mpjpe_mm")
                    if profile == HOCAP_STABLE_FUSED_HAND_V4 and (coverage is None or mpjpe is None):
                        raise ValueError("final hand-fit joint score is missing")
                    quality = "review" if coverage is None or coverage < 0.8 or mpjpe is None or mpjpe > 30 else "pass"
                    _write_json(episode_dir / "batch_run.json", {
                        "sequence_id": sequence_id, "view": view,
                        "source_archive": f"{sequence_id}.tar",
                        "source_revision": plan["revision"],
                        "model": MODEL, "inference_device": "cpu",
                        "profile": profile,
                        "score_key": score_key,
                        "reused_eight_view_detections": view == "pair",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "quality": quality,
                        "depth_consistency": result.acceptance["depth_consistency"],
                    })
                    record[view] = {
                        "status": "complete", "quality": quality, "score_key": score_key,
                        "episode_name": episode_dir.name,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "mpjpe_mm": mpjpe, "coverage": coverage,
                    }
                    LOGGER.info("complete %s %s: %s", sequence_id, view, record[view])
                except Exception as exc:
                    record[view] = {"status": "failed", "error": repr(exc),
                                    "traceback": traceback.format_exc()}
                    failures.append((sequence_id, view))
                    LOGGER.exception("failed %s %s", sequence_id, view)
                finally:
                    if episode_dir is not None and not retain_media:
                        _discard_media(episode_dir)
                    _write_json(status_path, status)
    LOGGER.info("batch finished; status: %s; failures: %s", status_path, failures)
    if failures:
        raise RuntimeError(f"{len(failures)} HO-Cap runs failed; see {status_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("/app/tools/hocap_pnp_plan.json"))
    parser.add_argument("--out", type=Path, default=Path("/app/data/datasets/hocap_stable_fused_v4"))
    parser.add_argument("--profile", choices=(HOCAP_STABLE_FUSED_HAND_V4, HOCAP_MEDIAPIPE_TRI_V1),
                        default=HOCAP_STABLE_FUSED_HAND_V4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--views", nargs="+", choices=("pair", "eight"), default=("eight", "pair"))
    parser.add_argument("--retain-media", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_batch(args.archives, args.plan, args.out, max_frames=args.max_frames,
              limit=args.limit, views=tuple(args.views), retain_media=args.retain_media,
              profile=args.profile)


if __name__ == "__main__":
    main()
