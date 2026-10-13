"""Stream one RGB view of right-hand HO-Cap pick-and-place sequences to MP4."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import tarfile
from pathlib import Path

import requests
import yaml


REPOSITORY = "JWRoboticsVision/HO-Cap-Dataset"
REVISION = "2b24836d5e51ad39e56ed4db3fc0c166e755332e"
CAMERA = "043422252387"
FPS = 30


def selected_sequences(labels_tar: Path) -> list[dict]:
    sequences = []
    with tarfile.open(labels_tar) as archive:
        for member in archive:
            if not member.name.endswith(".meta.yaml"):
                continue
            metadata = yaml.safe_load(archive.extractfile(member).read())
            if metadata.get("task_id") != 1 or metadata.get("mano_sides") != ["right"]:
                continue
            if CAMERA not in metadata["realsense"]["serials"]:
                raise ValueError(f"{member.name}: missing camera {CAMERA}")
            sequences.append(
                {
                    "id": member.name.removesuffix(".meta.yaml"),
                    "frames": int(metadata["num_frames"]),
                    "objects": metadata["object_ids"],
                }
            )
    return sorted(sequences, key=lambda item: item["id"])


def download_video(session: requests.Session, sequence: dict, output: Path) -> None:
    archive_name = f"{sequence['id']}.tar"
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{archive_name}"
    partial = output.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(partial),
    ]
    process = None
    frame_count = 0
    try:
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                for member in archive:
                    name = member.name
                    if not name.startswith(f"{CAMERA}-") or not name.endswith(".color.jpg"):
                        continue
                    frame_id = int(name.split("-", 1)[1].split(".", 1)[0])
                    if frame_id != frame_count:
                        raise ValueError(f"{archive_name}: expected frame {frame_count}, got {frame_id}")
                    if process is None:
                        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    process.stdin.write(archive.extractfile(member).read())
                    frame_count += 1
                    if frame_count % 300 == 0:
                        print(f"  {sequence['id']}: {frame_count}/{sequence['frames']} frames", flush=True)
                    if frame_count == sequence["frames"]:
                        break
        if frame_count != sequence["frames"] or process is None:
            raise ValueError(f"{archive_name}: found {frame_count}/{sequence['frames']} RGB frames")
        process.stdin.close()
        process.stdin = None
        _, errors = process.communicate()
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed for {archive_name}: {errors.decode(errors='replace')}")
        partial.replace(output)
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        partial.unlink(missing_ok=True)
        raise


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_index(output_dir: Path, entries: list[dict]) -> None:
    cards = []
    for entry in entries:
        name = html.escape(entry["file"])
        title = html.escape(entry["id"])
        objects = html.escape(", ".join(entry["objects"]))
        cards.append(
            f'<section><h2>{title}</h2><p>{entry["frames"] / FPS:.1f} s · {objects}</p>'
            f'<video controls preload="metadata" width="640" src="{name}"></video></section>'
        )
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>HO-Cap: right-hand pick-and-place</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:auto;padding:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:24px}'
        'section{border:1px solid #ccc;padding:12px}video{max-width:100%}</style>'
        f'<h1>HO-Cap: правая рука, pick-and-place — {len(entries)} видео</h1>'
        f'<p>Одна RGB-камера {CAMERA}, {FPS} fps. Архивы RGB-D не сохранялись.</p>'
        '<main>' + "".join(cards) + '</main></html>'
    )
    (output_dir / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_tar", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--only", action="append", help="download just one sequence ID (repeatable)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_sequences(args.labels_tar)
    if args.only:
        requested = set(args.only)
        selected = [item for item in selected if item["id"] in requested]
        if len(selected) != len(requested):
            raise ValueError(f"unknown sequence IDs: {sorted(requested - {item['id'] for item in selected})}")
    entries = []
    with requests.Session() as session:
        for sequence in selected:
            output = args.output_dir / f"{sequence['id']}.mp4"
            if not output.exists():
                print(f"Downloading {sequence['id']} ({sequence['frames']} frames)", flush=True)
                download_video(session, sequence, output)
            entry = {
                **sequence,
                "file": output.name,
                "bytes": output.stat().st_size,
                "sha256": file_hash(output),
            }
            entries.append(entry)
            print(f"Ready {sequence['id']} ({entry['bytes'] / 1e6:.1f} MB)", flush=True)
            (args.output_dir / "manifest.json").write_text(
                json.dumps(
                    {"repository": REPOSITORY, "revision": REVISION, "camera": CAMERA,
                     "fps": FPS, "videos": entries}, indent=2, ensure_ascii=False
                ) + "\n"
            )
            write_index(args.output_dir, entries)


if __name__ == "__main__":
    main()
