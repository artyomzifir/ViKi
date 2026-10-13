"""Learn a phase- and layout-conditioned correction to one cube-transfer plan."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def successful_layouts(augmented_root: Path) -> list[dict]:
    layouts = []
    for folder, width in (("augmented", 20), ("augmented-50mm", 50)):
        manifest_path = augmented_root / folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["candidates"]:
            if row["released_stack_success"]:
                layouts.append({
                    "layout_id": f"{width}mm_{row['candidate']:03d}",
                    "range_mm": width,
                    "plan": manifest_path.parent / row["plan"],
                    "red_offset_xy_m": row["red_offset_xy_m"],
                    "green_offset_xy_m": row["green_offset_xy_m"],
                })
    if len(layouts) != len({row["layout_id"] for row in layouts}):
        raise ValueError("duplicate layout identifiers")
    return layouts


def split_layouts(layouts: list[dict], seed: int = 42) -> dict[str, list[str]]:
    random = np.random.default_rng(seed)
    result = {name: [] for name in ("train", "validation", "test")}
    for width in sorted({row["range_mm"] for row in layouts}):
        names = sorted(row["layout_id"] for row in layouts if row["range_mm"] == width)
        if len(names) < 10:
            raise ValueError("each layout range needs at least ten successful samples")
        shuffled = random.permutation(names).tolist()
        test_count = max(2, round(0.2 * len(names)))
        validation_count = max(2, round(0.1 * len(names)))
        result["test"].extend(shuffled[:test_count])
        result["validation"].extend(shuffled[test_count:test_count + validation_count])
        result["train"].extend(shuffled[test_count + validation_count:])
    return {name: sorted(names) for name, names in result.items()}


def phase_features(times: np.ndarray, duration_s: float) -> np.ndarray:
    phase = np.clip(np.asarray(times, dtype=float).reshape(-1) / duration_s, 0.0, 1.0)
    harmonics = 2 * np.pi * phase[:, None] * np.arange(1, 9)
    return np.column_stack((phase, np.sin(harmonics), np.cos(harmonics))).astype(np.float32)


def command_schedule(plan_path: Path, time_scale: float = 3.0,
                     pressure_rad: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    from experiments.gripper_control import pressure_commands

    with h5py.File(plan_path) as plan:
        approach = np.asarray(plan["q_approach"], float)
        demonstration = np.asarray(plan["q"], float)
        drive = np.asarray(plan["gripper_joint_position"], float)
        timestamps = np.asarray(plan["timestamps"], float)
        frame_dt = float(plan["dt"][()])
        grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
    if (approach.ndim != 2 or approach.shape[1] != 6
            or demonstration.ndim != 2 or demonstration.shape[1] != 6
            or len(drive) != len(demonstration) or len(timestamps) != len(demonstration)
            or not np.all(np.diff(timestamps) > 0)):
        raise ValueError(f"invalid plan timing or dimensions: {plan_path}")
    drive = pressure_commands(drive, grasp, pressure_rad)
    warmup_s = 0.3
    approach_times = warmup_s + np.arange(len(approach)) * frame_dt
    demonstration_times = warmup_s + len(approach) * frame_dt + (timestamps - timestamps[0]) / 1e6
    times = np.concatenate(([0.0, warmup_s], approach_times[1:],
                            demonstration_times)) * time_scale
    joints = np.concatenate((approach[:1], approach, demonstration))
    fingers = np.concatenate((np.zeros(len(approach) + 1), drive))
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"non-increasing command schedule: {plan_path}")
    return times, np.column_stack((joints, fingers))


def sample_commands(schedule: tuple[np.ndarray, np.ndarray],
                    times: np.ndarray) -> np.ndarray:
    command_times, commands = schedule
    return np.column_stack([np.interp(times, command_times, commands[:, axis])
                            for axis in range(7)]).astype(np.float32)


def network():
    from torch import nn

    return nn.Sequential(nn.Linear(17, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 24))


def predicted_residual(model, phase, offsets):
    return (model(phase).reshape(-1, 6, 4) * offsets[:, None, :]).sum(dim=2)


def train(augmented_root: Path, source: Path, model_root: Path,
          output: Path, seed: int = 42, epochs: int = 180) -> dict:
    import torch

    from experiments.scene import load_scene

    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    layouts = successful_layouts(augmented_root)
    splits = split_layouts(layouts, seed)
    source_plan = source / "plan.h5"
    source_schedule = command_schedule(source_plan)
    duration = float(source_schedule[0][-1] + 0.7)
    sample_times = np.arange(0.0, duration, 0.02)
    source_actions = sample_commands(source_schedule, sample_times)
    phase = phase_features(sample_times, duration)
    source_object_model = (source / "intermediates/segmentation/"
                           "sam2.1_hiera_small/object_models.npz")
    scene = load_scene(source_plan, source_object_model, model_root)
    red_xy = scene.body_pos[scene.body("red_cube").id, :2].copy()
    green_xy = scene.body_pos[scene.body("green_cube").id, :2].copy()
    by_id = {row["layout_id"]: row for row in layouts}

    def dataset(names: list[str]):
        features = []
        offsets = []
        targets = []
        for name in names:
            row = by_id[name]
            candidate_schedule = command_schedule(row["plan"])
            if not np.allclose(candidate_schedule[0], source_schedule[0]):
                raise ValueError(f"candidate timing differs from source: {name}")
            candidate_actions = sample_commands(candidate_schedule, sample_times)
            if not np.allclose(candidate_actions[:, 6], source_actions[:, 6]):
                raise ValueError(f"candidate gripper command differs from source: {name}")
            offset = np.r_[row["red_offset_xy_m"], row["green_offset_xy_m"]]
            features.append(phase)
            offsets.append(np.tile((offset / 0.05).astype(np.float32), (len(phase), 1)))
            targets.append(candidate_actions[:, :6] - source_actions[:, :6])
        return (torch.from_numpy(np.concatenate(features)),
                torch.from_numpy(np.concatenate(offsets)),
                torch.from_numpy(np.concatenate(targets)))

    train_phase, train_offsets, train_targets = dataset(splits["train"])
    validation_phase, validation_offsets, validation_targets = dataset(splits["validation"])
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_phase)).split(2048):
            error = predicted_residual(model, train_phase[indices],
                                       train_offsets[indices]) - train_targets[indices]
            loss = error.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float((predicted_residual(
                model, validation_phase, validation_offsets)
                - validation_targets).square().mean())
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch}: validation MSE {validation_loss:.8f}", flush=True)
        if stale_epochs >= 25:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_loss = float((predicted_residual(model, train_phase, train_offsets)
                            - train_targets).square().mean())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "layout_bc_v1.pt"
    torch.save({"state_dict": best_state, "splits": splits,
                "source_red_xy_m": torch.from_numpy(red_xy.astype(np.float32)),
                "source_green_xy_m": torch.from_numpy(green_xy.astype(np.float32)),
                "duration_s": duration, "seed": seed}, checkpoint)
    report = {"checkpoint": str(checkpoint), "seed": seed,
              "source_episode": source.name,
              "layout_count": {name: len(names) for name, names in splits.items()},
              "sample_count": {"train": len(train_phase),
                               "validation": len(validation_phase)},
              "best_epoch": best_epoch, "train_joint_residual_mse_rad2": train_loss,
              "validation_joint_residual_mse_rad2": best_loss,
              "splits": splits}
    (output / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


class LayoutBCPolicy:
    def __init__(self, checkpoint: Path, source_plan: Path):
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.torch = torch
        self.model = network()
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.source_schedule = command_schedule(source_plan)
        self.duration = float(saved["duration_s"])
        self.source_red_xy = np.asarray(saved["source_red_xy_m"], float)
        self.source_green_xy = np.asarray(saved["source_green_xy_m"], float)
        self.offsets = None
        self.splits = saved["splits"]

    def __call__(self, state: np.ndarray, time_s: float) -> np.ndarray:
        if self.offsets is None:
            self.offsets = np.r_[state[14:16] - self.source_red_xy,
                                 state[27:29] - self.source_green_xy]
        phase = self.torch.from_numpy(phase_features(np.asarray([time_s]), self.duration))
        offsets = self.torch.from_numpy((self.offsets / 0.05).astype(np.float32)[None])
        with self.torch.no_grad():
            correction = predicted_residual(self.model, phase, offsets).numpy()[0]
        command = sample_commands(self.source_schedule, np.asarray([time_s]))[0]
        command[:6] += np.clip(correction, -0.3, 0.3)
        return command


def evaluate(checkpoint: Path, augmented_root: Path, source: Path,
             model_root: Path, output: Path) -> dict:
    from experiments.replay import replay

    layouts = {row["layout_id"]: row for row in successful_layouts(augmented_root)}
    source_plan = source / "plan.h5"
    policy_info = LayoutBCPolicy(checkpoint, source_plan)
    duration = policy_info.duration
    rows = []
    for split in ("train", "validation", "test"):
        for name in policy_info.splits[split]:
            row = layouts[name]
            red = np.asarray(row["red_offset_xy_m"])
            green = np.asarray(row["green_offset_xy_m"])
            policy = LayoutBCPolicy(checkpoint, source_plan)
            learned = replay(source, model_root, 3.0, 0.10,
                             red_offset_xy=red, green_offset_xy=green,
                             policy=policy, policy_uses_time=True,
                             policy_horizon_s=duration, control_hz=50.0)
            baseline = replay(source, model_root, 3.0, 0.10,
                              red_offset_xy=red, green_offset_xy=green,
                              command_plan_path=source_plan, hold_commands=True)
            rows.append({"layout_id": name, "split": split,
                         "red_offset_xy_m": red.tolist(),
                         "green_offset_xy_m": green.tolist(),
                         "learned": learned, "fixed_template": baseline})
            print(json.dumps({"layout_id": name, "split": split,
                              "learned": learned["released_stack_success"],
                              "fixed_template": baseline["released_stack_success"]}), flush=True)
    report = {"checkpoint": str(checkpoint), "policy_type": "time_and_initial_layout_conditioned_template_residual",
              "source_episode": source.name, "control_hz": 50,
              "success_count": {split: {
                  mode: sum(item[mode]["released_stack_success"] is True
                            for item in rows if item["split"] == split)
                  for mode in ("learned", "fixed_template")}
                  for split in policy_info.splits},
              "layout_count": {split: len(names) for split, names in policy_info.splits.items()},
              "layouts": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def summarize_independent(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["teacher_accepted"]]
    return {
        "all_layouts": {mode: sum(row[mode]["released_stack_success"] is True
                                  for row in rows)
                        for mode in ("learned", "fixed_template")},
        "teacher_accepted_layouts": {
            mode: sum(row[mode]["released_stack_success"] is True
                      for row in accepted)
            for mode in ("learned", "fixed_template")},
        "all_count": len(rows),
        "teacher_accepted_count": len(accepted),
    }


def evaluate_independent(checkpoint: Path, manifest_path: Path, source: Path,
                         model_root: Path, output: Path) -> dict:
    from experiments.replay import replay

    manifest = json.loads(manifest_path.read_text())
    source_plan = source / "plan.h5"
    duration = LayoutBCPolicy(checkpoint, source_plan).duration
    rows = []
    for candidate in manifest["candidates"]:
        red = np.asarray(candidate["red_offset_xy_m"])
        green = np.asarray(candidate["green_offset_xy_m"])
        learned = replay(source, model_root, 3.0, 0.10,
                         red_offset_xy=red, green_offset_xy=green,
                         policy=LayoutBCPolicy(checkpoint, source_plan),
                         policy_uses_time=True, policy_horizon_s=duration,
                         control_hz=50.0)
        fixed = replay(source, model_root, 3.0, 0.10,
                       red_offset_xy=red, green_offset_xy=green,
                       command_plan_path=source_plan, hold_commands=True)
        row = {"candidate": candidate["candidate"],
               "teacher_accepted": candidate["released_stack_success"],
               "red_offset_xy_m": red.tolist(),
               "green_offset_xy_m": green.tolist(),
               "learned": learned, "fixed_template": fixed}
        rows.append(row)
        print(json.dumps({"candidate": row["candidate"],
                          "teacher_accepted": row["teacher_accepted"],
                          "learned": learned["released_stack_success"],
                          "fixed_template": fixed["released_stack_success"]}), flush=True)
    report = {"checkpoint": str(checkpoint), "manifest": str(manifest_path),
              "seed": manifest["seed"], "offset_mm_per_axis": manifest["offset_mm_per_axis"],
              **summarize_independent(rows), "layouts": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate", "evaluate-seed"))
    parser.add_argument("--augmented-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/experiments/recovered-retarget/2026-10-06_11-01-57"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("/app/output/layout_bc_v1"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/app/output/augmented-test-50mm/manifest.json"))
    parser.add_argument("--epochs", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.mode == "train":
        result = train(arguments.augmented_root, arguments.source, arguments.models,
                       arguments.output, epochs=arguments.epochs)
    elif arguments.mode == "evaluate":
        result = evaluate(arguments.output / "layout_bc_v1.pt",
                          arguments.augmented_root, arguments.source, arguments.models,
                          arguments.output / "evaluation.json")
    else:
        result = evaluate_independent(arguments.output / "layout_bc_v1.pt",
                                      arguments.manifest, arguments.source,
                                      arguments.models,
                                      arguments.output / "independent_50mm.json")
    print(json.dumps({key: value for key, value in result.items() if key != "layouts"},
                     indent=2))
