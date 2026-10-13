"""Build thesis metrics and figures from the recorded cube-transfer artifacts.

Run from the repository root in the ViKi Docker image, mounting ``paper/`` at
``/app/paper``. The script reads source artifacts but never changes them.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/datasets/40mm-pnp"
EXPERIMENTS = ROOT / "data/experiments"
OUTPUT = ROOT / "paper/thesis/figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def wilson(successes: int, count: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    fraction = successes / count
    denominator = 1 + z * z / count
    centre = (fraction + z * z / (2 * count)) / denominator
    radius = z * sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return centre - radius, centre + radius


def blue_extent_mm(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as model:
        labels = model["object_labels"].tolist()
        index = labels.index("manipulated_object")
        offsets = model["model_offsets"]
        xyz = model["model_xyz_object"][offsets[index] : offsets[index + 1]]
    return np.percentile(xyz, 98, axis=0) * 1000 - np.percentile(xyz, 2, axis=0) * 1000


def contact_count(result: dict, support: str) -> int:
    return sum(result["support_robot_contact_counts"][support].values())


def gather() -> tuple[dict, dict]:
    audit = read_json(EXPERIMENTS / "2026-10-09-full-audit.json")
    episodes = audit["episodes"]
    episode_names = [episode["episode"] for episode in episodes]
    assert len(episode_names) == len(set(episode_names)) == audit["episode_count"]

    gaps_us = []
    joint_solved = joint_total = 0
    joint_episode_coverage = []
    prepare_pass = 0
    supported_frames = total_prepared_frames = 0
    support_episode_coverage = []
    unsupported_runs = []
    plan_rmse_mm = []
    for name in episode_names:
        episode = DATASET / name
        times = read_json(episode / "raw/timestamps.json")
        for frame in times:
            device_times = frame["device_timestamps_us"]
            gaps_us.append(abs(device_times["kinect_0"] - device_times["kinect_1"]))

        hand = read_json(episode / "raw/joints3d_summary.json")
        joint_solved += hand["joints_solved"]
        joint_total += hand["joints_total"]
        joint_episode_coverage.append(hand["joints_solved"] / hand["joints_total"])

        stages = read_json(episode / "status.json")["stages"]
        if stages.get("prepare", {}).get("done"):
            prepare_pass += 1
            with np.load(episode / "cln.npz", allow_pickle=False) as clean:
                support = clean["geometry_support_mask"].astype(bool)
            supported_frames += int(support.sum())
            total_prepared_frames += len(support)
            support_episode_coverage.append(float(support.mean()))
            run = longest_run(~support)
            unsupported_runs.append(run)

        if (episode / "plan.h5").exists():
            plan_rmse_mm.append(stages["retarget"]["position_rmse_mm"])

    comparison = read_json(EXPERIMENTS / "track-recovery-comparison.json")
    assert comparison["count"] == prepare_pass
    accepted = [episode for episode in comparison["episodes"] if episode["after"] == "accepted"]
    blue_extents = []
    for episode in accepted:
        name = episode["episode"]
        path = (
            DATASET / name / "intermediates/segmentation/sam2.1_hiera_small/object_models.npz"
            if episode["before"] == "accepted"
            else EXPERIMENTS / "track-recovery-full" / f"{name}.npz"
        )
        blue_extents.append(blue_extent_mm(path))
    blue_extents = np.asarray(blue_extents)

    recovery = read_json(EXPERIMENTS / "recovered-retarget/summary.json")
    final_status = {}
    for line in (EXPERIMENTS / "recovered-retarget/status.jsonl").read_text().splitlines():
        status = json.loads(line)
        final_status[status["episode"]] = status["status"]
    accepted_plan_names = {name for name, status in final_status.items() if status == "accepted"}
    physics = read_json(EXPERIMENTS / "recovered-retarget-physics.json")["episodes"]

    plain_bc = read_json(EXPERIMENTS / "bc-v1/evaluation.json")
    sweeps = {}
    for extent, directory in ((20, "augmented"), (50, "augmented-50mm")):
        manifest = read_json(EXPERIMENTS / "bc-v1" / directory / "manifest.json")
        successes = sum(bool(candidate["released_stack_success"]) for candidate in manifest["candidates"])
        sweeps[str(extent)] = {"successes": successes, "candidates": manifest["candidate_count"]}
        assert successes == manifest["success_count"]

    layout = read_json(EXPERIMENTS / "bc-v1/layout_bc_v1/evaluation.json")
    independent = read_json(EXPERIMENTS / "bc-v1/layout_bc_v1/independent_50mm.json")
    layout_splits = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in layout["layouts"] if row["split"] == split]
        layout_splits[split] = {
            "count": len(rows),
            "learned": sum(bool(row["learned"]["released_stack_success"]) for row in rows),
            "template": sum(bool(row["fixed_template"]["released_stack_success"]) for row in rows),
        }
    independent_rows = independent["layouts"]
    independent_counts = {
        "count": len(independent_rows),
        "teacher_accepted": sum(bool(row["teacher_accepted"]) for row in independent_rows),
        "learned": sum(bool(row["learned"]["released_stack_success"]) for row in independent_rows),
        "template": sum(bool(row["fixed_template"]["released_stack_success"]) for row in independent_rows),
        "red_contacts": sum(contact_count(row["learned"], "red") > 0 for row in independent_rows),
    }

    quality = read_json(EXPERIMENTS / "bc-v1/quality-batch-001/manifest.json")
    baseline_errors = []
    selected_errors = []
    baseline_rotation = []
    selected_rotation = []
    baseline_impulse = []
    selected_impulse = []
    for episode in quality["episodes"]:
        variants = {item["variant"]: item for item in episode["variants"]}
        baseline, selected = variants["baseline"], variants[episode["best_variant"]]
        assert baseline["released_stack_success"] and selected["released_stack_success"]
        baseline_errors.append(baseline["center_error_mm"])
        selected_errors.append(selected["center_error_mm"])
        baseline_rotation.append(baseline["orientation_error_deg"])
        selected_rotation.append(selected["orientation_error_deg"])
        baseline_impulse.append(baseline["red_contact_impulse_Ns"])
        selected_impulse.append(selected["red_contact_impulse_Ns"])

    gaps = np.asarray(gaps_us)
    rmse = np.asarray(plan_rmse_mm)
    summary = {
        "capture": {
            "episodes": len(episode_names), "paired_frames": len(gaps_us),
            "device_pair_gap_us": {"median": float(np.median(gaps)), "p95": float(np.percentile(gaps, 95)),
                                   "maximum": int(gaps.max()), "over_1000": int((gaps > 1000).sum())},
        },
        "hand": {"solved": joint_solved, "total": joint_total, "coverage": joint_solved / joint_total,
                 "episode_coverage_p10_median_p90": np.percentile(joint_episode_coverage, [10, 50, 90]).tolist()},
        "prepare": {"passed": prepare_pass, "support_frames": supported_frames,
                    "total_frames": total_prepared_frames, "support_fraction": supported_frames / total_prepared_frames,
                    "episode_support_p10_median_p90": np.percentile(support_episode_coverage, [10, 50, 90]).tolist(),
                    "longest_unsupported_median_p95_max": [float(np.median(unsupported_runs)),
                                                          float(np.percentile(unsupported_runs, 95)), max(unsupported_runs)]},
        "object": {"original_accepted": comparison["before"]["accepted"], "recovered_accepted": len(accepted),
                   "blue_extent_median_mm": np.median(blue_extents, axis=0).tolist(),
                   "any_extent_over_50mm": int((blue_extents > 50).any(axis=1).sum())},
        "retarget": {"canonical_plans": len(plan_rmse_mm), "canonical_rmse_median_mm": float(np.median(rmse)),
                     "canonical_rmse_p90_mm": float(np.percentile(rmse, 90)),
                     "canonical_rmse_max_mm": float(rmse.max()),
                     "scratch_selected": recovery["selected"], "scratch_counts": recovery["counts"]},
        "physics": {"scratch_plans": len(physics),
                    "all_success": sum(bool(row["released_stack_success"]) for row in physics),
                    "geometry_accepted_success": sum(bool(row["released_stack_success"]) and row["episode"] in accepted_plan_names for row in physics),
                    "red_contacts": sum(contact_count(row, "red") > 0 for row in physics),
                    "green_contacts": sum(contact_count(row, "green") > 0 for row in physics)},
        "plain_bc": {"episodes": sum(plain_bc["episode_count"].values()),
                     "successes": sum(plain_bc["success_count"].values())},
        "sweeps": sweeps,
        "layout_policy": {"source_episode": layout["source_episode"], "splits": layout_splits,
                          "independent_50mm": independent_counts,
                          "test_wilson_95": wilson(layout_splits["test"]["learned"], layout_splits["test"]["count"]),
                          "independent_wilson_95": wilson(independent_counts["learned"], independent_counts["count"])},
        "quality_selection": {"count": len(baseline_errors),
                              "mean_center_error_baseline_selected_mm": [float(np.mean(baseline_errors)), float(np.mean(selected_errors))],
                              "mean_rotation_baseline_selected_deg": [float(np.mean(baseline_rotation)), float(np.mean(selected_rotation))],
                              "mean_red_impulse_baseline_selected_Ns": [float(np.mean(baseline_impulse)), float(np.mean(selected_impulse))]},
    }
    assert summary["physics"]["geometry_accepted_success"] <= recovery["counts"]["accepted"]
    plot_data = {"gaps": gaps, "quality_baseline": baseline_errors, "quality_selected": selected_errors}
    return summary, plot_data


def longest_run(mask: np.ndarray) -> int:
    current = maximum = 0
    for value in mask:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(summary: dict, plot_data: dict) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1, 1.2]})
    counts = [summary["capture"]["episodes"], summary["prepare"]["passed"],
              summary["object"]["original_accepted"], summary["object"]["recovered_accepted"]]
    labels = ["Recorded", "Hand gate", "Object original", "Object recovered"]
    axes[0].barh(labels[::-1], counts[::-1], color=["#4a6fa5", "#27867a", "#a85c4b", "#4a6fa5"])
    axes[0].set_xlim(0, 88)
    axes[0].set_xlabel("Episodes (common cohort of 78)")
    for index, value in enumerate(counts[::-1]):
        axes[0].text(value + 1, index, str(value), va="center")
    axes[1].hist(plot_data["gaps"], bins=np.arange(100, 211, 5), color="#4a6fa5")
    axes[1].axvline(summary["capture"]["device_pair_gap_us"]["median"], color="#a85c4b", linestyle="--",
                    label="median")
    axes[1].set_xlabel("Kinect device-timestamp pair gap (µs)")
    axes[1].set_ylabel("Paired frame groups")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "capture_and_gates")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    groups = [summary["layout_policy"]["splits"]["test"], summary["layout_policy"]["independent_50mm"]]
    positions = np.arange(2)
    for offset, key, color, label in ((-0.19, "learned", "#27867a", "Learned correction"),
                                      (0.19, "template", "#a85c4b", "Fixed source plan")):
        fractions = [group[key] / group["count"] for group in groups]
        bars = ax.bar(positions + offset, fractions, width=0.36, color=color, label=label)
        for bar, group in zip(bars, groups):
            low, high = wilson(group[key], group["count"])
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=[[bar.get_height() - low], [high - bar.get_height()]],
                        color="#222222", capsize=3, fmt="none")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.035,
                    f"{group[key]}/{group['count']}", ha="center", color="white" if fractions[0] > 0.15 else "black")
    ax.set_xticks(positions, ["Held-out selected\nlayouts (n=14)", "Independent seed\nall layouts (n=30)"])
    ax.set_ylabel("Released stable-stack success fraction")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    save_figure(fig, "layout_policy_success")

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    baseline = np.asarray(plot_data["quality_baseline"])
    selected = np.asarray(plot_data["quality_selected"])
    limit = max(baseline.max(), selected.max()) + 1
    ax.plot([0, limit], [0, limit], color="#777777", linestyle="--", linewidth=1)
    ax.scatter(baseline, selected, color="#27867a", s=35)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Baseline centre error (mm)",
           ylabel="Selected centre error (mm)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(fig, "placement_quality_selection")


def main() -> None:
    summary, plot_data = gather()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(summary, plot_data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
