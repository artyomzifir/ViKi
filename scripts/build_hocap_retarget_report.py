"""Render the resumable HO-Cap retarget status into a lossless Markdown diary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_report(status_path: Path) -> str:
    raw = status_path.read_bytes()
    status = json.loads(raw)
    episodes = status["episodes"]
    rows = sorted(episodes.items())
    complete = sum(row["status"] == "complete" for _, row in rows)
    lines = [
        "# HO-Cap retarget: численный дневник (2026-10-14)",
        "",
        f"Эпизодов: {len(rows)}, завершено: {complete}, "
        f"не завершено: {len(rows) - complete}. "
        "Источник: `data/datasets/hocap_eightview/retarget_batch_status.json`.",
        "",
        "UR10 + Robotiq 2F-85; hand-only IK, без cube-object grasp. "
        "База `[0, -1.0, 0]` и бесконечная плоскость стола `z=0` "
        "заданы в `hocap_tag_1`. Искусственный подход из robot-home "
        "отключён (`approach_sec=0`): оценивается только записанная "
        "демонстрация, безопасный переход из home этим тестом не доказан. "
        "Жёсткая столешница ограничивает "
        "движущиеся collision bodies и contact points захвата, но её реальная "
        "высота не измерена независимо. Стол не является конечной mesh-геометрией; "
        "человек и предметы не участвуют в collision avoidance. Это не "
        "гарантия физической безопасности или успешного захвата.",
        "",
        "Сначала 676-кадровый smoke. Батч допускается при RMSE позиции "
        "≤30 мм, p95 ≤50 мм, RMSE ориентации ≤45°, минимальном зазоре "
        "плоскости ≥−0.1 мм, self-collision margin ≥−0.1 мм и конечных "
        "суставах. Это операционный gate, не научный порог качества.",
        "",
        "| Эпизод | Статус | Кадры | RMSE позиции, мм | p95, мм | RMSE ориентации, ° | Поправка скорости, мрад | Мин. зазор до плоскости, мм | Мин. self-collision margin, мм |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for episode, row in rows:
        score = row.get("metrics", {})

        def fmt(key: str) -> str:
            value = score.get(key)
            return "—" if value is None else f"{value:.2f}"

        collision = score.get("min_collision_margin")
        collision_mm = "—" if collision is None else f"{collision * 1000:.2f}"
        repair = score.get("velocity_repair_max_rad")
        repair_mrad = "—" if repair is None else f"{repair * 1000:.3f}"
        lines.append(
            f"| `{episode.split('_105322')[0]}` | {row['status']} | "
            f"{row.get('frames', '—')} | {fmt('position_rmse_mm')} | "
            f"{fmt('position_p95_mm')} | {fmt('orientation_rmse_deg')} | "
            f"{repair_mrad} | {fmt('min_floor_margin_mm')} | {collision_mm} |"
        )
    lines.extend([
        "",
        "Ниже исходный JSON статуса целиком, без округления и без удаления "
        "полей. Полный `plan.h5` каждого успешного эпизода остаётся рядом "
        "с входным `cln.npz`; его `metrics_json` и `config_json` скопированы "
        "в статус без изменения значений.",
        "",
        f"SHA-256 статуса: `{hashlib.sha256(raw).hexdigest()}`",
        "",
        "```json",
        raw.decode("utf-8").rstrip("\n"),
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build_report(args.status), encoding="utf-8")


if __name__ == "__main__":
    main()
