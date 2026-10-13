"""Summarize durable artifacts and object-track quality for the cube batch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


MODEL = Path("intermediates/segmentation/sam2.1_hiera_small/object_models.npz")


def _track_stats(archive: np.lib.npyio.NpzFile, index: int) -> dict:
    position = np.asarray(archive["translation_world"][:, index], dtype=float)
    confidence = np.asarray(archive["track_confidence"][:, index], dtype=float)
    raw_confidence = np.asarray(archive["track_confidence_raw"][:, index], dtype=float)
    contact = np.asarray(archive["contact"][:, index], dtype=bool)
    offsets = np.asarray(archive["model_offsets"], dtype=int)
    surface = np.asarray(
        archive["model_xyz_object"][offsets[index]:offsets[index + 1]], dtype=float)
    model_kind = np.asarray(archive["model_kind"][offsets[index]:offsets[index + 1]])
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    return {
        "model_points": len(surface),
        "core_points": int(np.count_nonzero(model_kind == 1)),
        "extent_p02_p98_m": (
            np.percentile(surface, 98, axis=0)
            - np.percentile(surface, 2, axis=0)
        ).tolist(),
        "first_xyz_m": position[0].tolist(),
        "last_xyz_m": position[-1].tolist(),
        "travel_m": float(np.linalg.norm(position[-1] - position[0])),
        "path_length_m": float(step.sum()),
        "largest_frame_step_m": float(step.max(initial=0)),
        "confidence_median": float(np.median(confidence)),
        "confidence_p10": float(np.percentile(confidence, 10)),
        "confidence_zero_fraction": float(np.mean(confidence < 0.05)),
        "raw_confidence_median": float(np.median(raw_confidence)),
        "contact_fraction": float(np.mean(contact)),
        "contact_first_frame": int(np.flatnonzero(contact)[0]) if contact.any() else None,
        "contact_last_frame": int(np.flatnonzero(contact)[-1]) if contact.any() else None,
        "residual_median_m": float(np.median(
            np.asarray(archive["residual_median_m"][:, index], dtype=float)[
                np.isfinite(archive["residual_median_m"][:, index])
            ])) if np.isfinite(archive["residual_median_m"][:, index]).any() else None,
        "coverage_median": float(np.median(archive["coverage"][:, index])),
        "retained_fraction_median": float(np.median(
            archive["retained_fraction"][:, index])),
    }


def _model_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        ids = [int(value) for value in archive["object_ids"]]
        labels = [str(value) for value in archive["object_labels"]]
        objects = {
            str(object_id): _track_stats(archive, index)
            for index, object_id in enumerate(ids)
        }
        blue_index = next((index for index, label in enumerate(labels)
                           if label == "manipulated_object"), None)
        if blue_index is None or len(ids) != 3:
            reason = "wrong_object_labels_or_count"
        else:
            blue = objects[str(ids[blue_index])]
            if not np.isfinite(archive["translation_world"][:, blue_index]).all():
                reason = "nonfinite_track"
            elif not 0.2 <= blue["travel_m"] <= 0.6:
                reason = "blue_travel"
            elif blue["confidence_median"] < 0.65:
                reason = "blue_confidence_median"
            elif blue["confidence_p10"] < 0.5:
                reason = "blue_confidence_p10"
            else:
                reason = "accepted"
        return {
            "schema": str(archive["schema"]),
            "frame_count": int(archive["frame_count"]),
            "object_ids": ids,
            "object_labels": labels,
            "objects": objects,
            "quality_reason": reason,
        }


def _plan_stats(path: Path) -> dict:
    with h5py.File(path) as archive:
        metrics = json.loads(archive["metrics_json"][()].decode())
        grasp = metrics.get("object_grasp") or {}
        return {
            "frames": len(archive["q"]),
            "solver_status": archive["solver_status"][()].decode(),
            "position_rmse_mm": metrics.get("position_rmse_mm"),
            "object_grasp_mode": grasp.get("mode"),
            "phase_bounds": grasp.get("phase_bounds"),
        }


def audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for relative_path in manifest["episodes"]:
        episode = Path(relative_path)
        row = {"episode": episode.name, "path": str(episode)}
        status_path = episode / "status.json"
        if status_path.is_file():
            stages = json.loads(status_path.read_text()).get("stages", {})
            row["completed_stages"] = sorted(
                stage for stage, record in stages.items() if record.get("done"))
        model_path = episode / MODEL
        if model_path.is_file():
            try:
                row["model"] = _model_stats(model_path)
            except (OSError, ValueError, KeyError, IndexError) as error:
                row["model_error"] = str(error)
        plan_path = episode / "plan.h5"
        if plan_path.is_file():
            try:
                row["plan"] = _plan_stats(plan_path)
            except (OSError, ValueError, KeyError) as error:
                row["plan_error"] = str(error)
        rows.append(row)
    reasons = Counter(
        row.get("model", {}).get("quality_reason", "no_model") for row in rows)
    modes = Counter(row.get("plan", {}).get("object_grasp_mode", "no_plan")
                    for row in rows)
    return {
        "manifest": str(manifest_path),
        "episode_count": len(rows),
        "model_quality_reasons": dict(sorted(reasons.items())),
        "plan_modes": dict(sorted(modes.items())),
        "episodes": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/batch_extract_2026-10-09.json"))
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.manifest), indent=2, sort_keys=True))
