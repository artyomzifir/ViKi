"""Build a lossless, reproducible HO-Cap batch-results diary from episode JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EPISODE_FILES = (
    "gt/evaluation.json",
    "intermediates/segmentation/hocap-oracle-v1/evaluation.json",
    "acceptance.json",
    "batch_run.json",
)
STATUS_FILES = ("batch_status.json", "object_batch_status.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def build_report(dataset: Path) -> str:
    episodes = sorted(path for path in dataset.iterdir() if path.is_dir())
    hand_status, object_status = (_read(dataset / name) for name in STATUS_FILES)
    for label, status in (("hand", hand_status), ("object", object_status)):
        rows = list(status["sequences"].values())
        if len(rows) != len(episodes):
            raise ValueError(f"{label}: {len(rows)} status rows for {len(episodes)} episodes")
        if any((row.get("eight", row)).get("status") != "complete" for row in rows):
            raise ValueError(f"{label}: batch contains unfinished rows")

    lines = [
        "# HO-Cap PnP: дневник полного прогона (2026-10-13)",
        "",
        f"Источник: `data/datasets/hocap_eightview/`; {len(episodes)} эпизодов. "
        "Статусы обеих очередей: complete для каждого эпизода, без пропусков.",
        "",
        "## Протокол и ограничения",
        "",
        "- 8 RealSense RGB-D ракурсов; MediaPipe CPU, калиброванная триангуляция; "
        "точное сопоставление по номеру кадра в `hocap_tag_1`. "
        "Детали и ревизия датасета — в сырых статусах и "
        "`docs/2026-10-13-hocap-batch-protocol.md`.",
        "- Референс кисти и объектов — опубликованные полуавтоматические аннотации "
        "HO-Cap, не независимый mocap. MPJPE, p95 и PCK считаются только по "
        "сопоставленным суставам; coverage — их доля среди референсных.",
        "- Объектный трекер получает опубликованные instance-маски (oracle), "
        "а не предсказывает сегментацию. Сравниваются относительные позы после "
        "одного правого SE(3)-совмещения локальной системы объекта на первом "
        "уверенном кадре. Это drift, не абсолютная локализация.",
        "- Ошибки по неподвижным периодам могут занижать сложность движения. "
        "Выброс `subject_7-20231023_162803 / G18_2` оставлен как limitation. "
        "Эти результаты не валидируют калибровку локальных Kinect и не "
        "доказывают безопасность или физический успех ретаргета.",
        "",
        "## Кисть по эпизодам (Prepare: smoothed_all)",
        "",
        "| Эпизод | Кадры | Coverage | MPJPE, мм | p95, мм | PCK@20 | PCK@50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    object_rows = []
    matched_total = gt_total = 0
    weighted_error = 0.0
    for episode in episodes:
        hand = _read(episode / EPISODE_FILES[0])
        score = hand["prepare"]["smoothed_all"]
        gt_total += score["gt_joints"]
        matched_total += score["matched_joints"]
        weighted_error += score["mpjpe_mm"] * score["matched_joints"]
        lines.append(
            f"| `{episode.name.split('_105322')[0]}` | {hand['frame_count']} | "
            f"{_metric(score['coverage'], 4)} | {_metric(score['mpjpe_mm'])} | "
            f"{_metric(score['p95_mm'])} | {_metric(score['pck_20mm'], 4)} | "
            f"{_metric(score['pck_50mm'], 4)} |"
        )
        objects = _read(episode / EPISODE_FILES[1])["objects"]
        for object_id, metrics in sorted(objects.items()):
            object_rows.append((episode.name.split('_105322')[0], object_id, metrics))
    lines.extend([
        "",
        f"Суммарно: {matched_total}/{gt_total} сопоставленных суставов "
        f"(coverage {_metric(matched_total / gt_total, 4)}); "
        f"взвешенный по сопоставленным суставам MPJPE "
        f"{_metric(weighted_error / matched_total)} мм. "
        "Сводный p95 из отдельных p95 не вычислялся: без распределения "
        "покадровых ошибок он был бы некорректен.",
        "",
        "## Объекты по эпизодам",
        "",
        "| Эпизод | Объект | Coverage | Смещение mean/p95, мм | Поворот mean/p95, ° | Якорь, кадр |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name, object_id, score in object_rows:
        lines.append(
            f"| `{name}` | `{object_id}` | {_metric(score['coverage'], 4)} | "
            f"{_metric(score['translation_mean_mm'])} / {_metric(score['translation_p95_mm'])} | "
            f"{_metric(score['rotation_mean_deg'])} / {_metric(score['rotation_p95_deg'])} | "
            f"{score['anchor_frame']} |"
        )
    lines.extend([
        "",
        f"Объектных треков: {len(object_rows)}. Числа в таблицах округлены "
        "только для чтения; далее приведены исходные файлы без изменения "
        "содержимого, с SHA-256 каждого файла.",
        "",
        "## Полные исходные JSON",
        "",
    ])
    for path in [*(dataset / name for name in STATUS_FILES),
                 *(episode / name for episode in episodes for name in EPISODE_FILES)]:
        raw = path.read_bytes()
        json.loads(raw)
        lines.extend([
            f"### `{path.relative_to(dataset)}`",
            "",
            f"SHA-256: `{hashlib.sha256(raw).hexdigest()}`",
            "",
            "```json",
            raw.decode("utf-8").rstrip("\n"),
            "```",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build_report(args.dataset), encoding="utf-8")


if __name__ == "__main__":
    main()
