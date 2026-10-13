"""Small, state-based BC baseline on physically successful cube rollouts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.observations import ACTION_SCHEMA, OBSERVATION_SCHEMA


def split_episodes(names: list[str], seed: int = 42) -> dict[str, list[str]]:
    if len(names) < 10 or len(set(names)) != len(names):
        raise ValueError("BC baseline requires at least ten distinct source episodes")
    shuffled = [str(name) for name in np.random.default_rng(seed).permutation(sorted(names))]
    test_count = max(3, round(0.2 * len(shuffled)))
    validation_count = max(2, round(0.1 * len(shuffled)))
    return {
        "train": sorted(shuffled[test_count + validation_count:]),
        "validation": sorted(shuffled[test_count:test_count + validation_count]),
        "test": sorted(shuffled[:test_count]),
    }


def action_residual(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if observations.ndim != 2 or observations.shape[1] != 53 or actions.shape != (len(observations), 7):
        raise ValueError("expected aligned 53-D observations and 7-D actions")
    current = np.column_stack((observations[:, :6], observations[:, 12]))
    return actions - current


def load_rollouts(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    expected = {row["episode"] for row in manifest["episodes"]
                if row["released_stack_success"]}
    found = {path.stem for path in directory.glob("*.npz")}
    if expected != found or len(expected) != manifest["successful_count"]:
        raise ValueError("rollout files disagree with the physics-success manifest")
    result = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive["observation_schema"]) != OBSERVATION_SCHEMA
                    or str(archive["action_schema"]) != ACTION_SCHEMA
                    or str(archive["source_episode"]) != path.stem):
                raise ValueError(f"incompatible rollout schema or source: {path}")
            observations = np.asarray(archive["observation"], np.float32)
            actions = np.asarray(archive["action"], np.float32)
            times = np.asarray(archive["time_s"], np.float64)
        action_residual(observations, actions)
        if (len(times) != len(observations) or len(times) < 2
                or not np.all(np.diff(times) > 0)
                or not np.isfinite(observations).all()
                or not np.isfinite(actions).all()):
            raise ValueError(f"invalid rollout samples: {path}")
        result[path.stem] = observations, actions
    return result


def network():
    from torch import nn

    return nn.Sequential(nn.Linear(53, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 7))


def train(rollouts_dir: Path, output_dir: Path, seed: int = 42,
          epochs: int = 80) -> dict:
    import torch

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rollouts = load_rollouts(rollouts_dir)
    splits = split_episodes(list(rollouts), seed)

    def concatenate(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        obs = np.concatenate([rollouts[name][0] for name in names])
        actions = np.concatenate([rollouts[name][1] for name in names])
        return obs, action_residual(obs, actions)

    train_obs, train_actions = concatenate(splits["train"])
    validation_obs, validation_actions = concatenate(splits["validation"])
    obs_mean = train_obs.mean(axis=0)
    obs_scale = np.maximum(train_obs.std(axis=0), 0.01)
    action_mean = train_actions.mean(axis=0)
    action_scale = np.maximum(train_actions.std(axis=0), 0.01)

    def tensor_pair(obs: np.ndarray, actions: np.ndarray):
        return (torch.from_numpy((obs - obs_mean) / obs_scale),
                torch.from_numpy((actions - action_mean) / action_scale))

    train_x, train_y = tensor_pair(train_obs, train_actions)
    validation_x, validation_y = tensor_pair(validation_obs, validation_actions)
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_x)).split(512):
            prediction = model(train_x[indices])
            loss = torch.nn.functional.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.nn.functional.mse_loss(
                model(validation_x), validation_y))
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "bc_state_v1.pt"
    torch.save({
        "state_dict": best_state,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "obs_mean": torch.from_numpy(obs_mean),
        "obs_scale": torch.from_numpy(obs_scale),
        "action_mean": torch.from_numpy(action_mean),
        "action_scale": torch.from_numpy(action_scale),
        "action_min": torch.from_numpy(np.min(
            train_actions + np.column_stack((train_obs[:, :6], train_obs[:, 12])), axis=0)),
        "action_max": torch.from_numpy(np.max(
            train_actions + np.column_stack((train_obs[:, :6], train_obs[:, 12])), axis=0)),
        "splits": splits,
        "seed": seed,
    }, checkpoint)
    with torch.no_grad():
        train_loss = float(torch.nn.functional.mse_loss(model(train_x), train_y))
    report = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "best_epoch": best_epoch,
        "train_normalized_action_mse": train_loss,
        "validation_normalized_action_mse": best_validation,
        "sample_count": {name: int(sum(len(rollouts[episode][0]) for episode in episodes))
                         for name, episodes in splits.items()},
        "splits": splits,
    }
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


class BCPolicy:
    def __init__(self, checkpoint: Path):
        import torch

        self.torch = torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["observation_schema"] != OBSERVATION_SCHEMA
                or saved["action_schema"] != ACTION_SCHEMA):
            raise ValueError("checkpoint uses an incompatible observation/action schema")
        self.model = network()
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.obs_mean = np.asarray(saved["obs_mean"], np.float32)
        self.obs_scale = np.asarray(saved["obs_scale"], np.float32)
        self.action_mean = np.asarray(saved["action_mean"], np.float32)
        self.action_scale = np.asarray(saved["action_scale"], np.float32)
        self.action_min = np.asarray(saved["action_min"], np.float32)
        self.action_max = np.asarray(saved["action_max"], np.float32)
        self.splits = saved["splits"]

    def __call__(self, state: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            normalized = self.torch.from_numpy(
                (state - self.obs_mean) / self.obs_scale)
            residual = self.model(normalized).numpy() * self.action_scale + self.action_mean
        current = np.r_[state[:6], state[12]]
        bounded = np.clip(residual, [-0.1] * 6 + [-0.08], [0.1] * 6 + [0.08])
        target = np.clip(current + bounded, self.action_min - 0.1,
                         self.action_max + 0.1)
        target[6] = np.clip(target[6], 0.0, 0.8)
        return target


def evaluate(checkpoint: Path, episodes_root: Path, output: Path,
             model_root: Path) -> dict:
    from experiments.replay import replay

    policy = BCPolicy(checkpoint)
    rows = []
    for split in ("train", "validation", "test"):
        for episode_name in policy.splits[split]:
            print(f"Evaluating {split}: {episode_name}", flush=True)
            result = replay(episodes_root / episode_name, model_root,
                            time_scale=3.0, policy=policy, control_hz=50.0)
            rows.append({"split": split, **result})
            print(json.dumps({"episode": episode_name, "split": split,
                              "success": result["released_stack_success"],
                              "reason": result["invalid_reason"]}), flush=True)
    report = {
        "checkpoint": str(checkpoint),
        "horizon_s": 40.0,
        "control_hz": 50.0,
        "success_count": {split: sum(row["released_stack_success"] is True
                                     for row in rows if row["split"] == split)
                          for split in policy.splits},
        "episode_count": {split: len(policy.splits[split]) for split in policy.splits},
        "episodes": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--episodes", type=Path,
                        default=Path("data/experiments/recovered-retarget"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=80)
    arguments = parser.parse_args()
    if arguments.mode == "train":
        result = train(arguments.output / "rollouts", arguments.output,
                       epochs=arguments.epochs)
    else:
        result = evaluate(arguments.output / "bc_state_v1.pt", arguments.episodes,
                          arguments.output / "evaluation.json", arguments.models)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "episodes"}, indent=2))
