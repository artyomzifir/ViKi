"""Train and inspect the first state-based HO-Cap grasp SAC baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_checker import check_env

from experiments.hocap_grasp_env import HocapGraspEnv


def seed_contact_experience(model: SAC, environment: HocapGraspEnv,
                            episodes: int, seed: int) -> dict:
    if episodes < 0:
        raise ValueError("expert episode count must not be negative")
    if episodes == 0:
        return {"episodes": 0, "successes": 0, "transitions": 0}
    model.replay_buffer = ReplayBuffer(
        model.buffer_size, environment.observation_space,
        environment.action_space, device=model.device, n_envs=1,
    )
    successes = 0
    transitions = 0
    for episode_index in range(episodes):
        state, _ = environment.reset(seed=seed + 100 + episode_index)
        outcome = {}
        for step_index in range(environment.max_steps):
            action = np.zeros(5, dtype=np.float32)
            if step_index < 2:
                action[0] = -1.0
            elif step_index < 4:
                action[1] = 1.0
            elif step_index < 34:
                action[4] = float(environment.data.ctrl[6] < 0.45)
            else:
                action[2] = 1.0
            next_state, reward, terminated, truncated, outcome = environment.step(action)
            model.replay_buffer.add(
                state[None], next_state[None], action[None],
                np.array([reward], dtype=np.float32),
                np.array([terminated or truncated]), [outcome],
            )
            transitions += 1
            state = next_state
            if terminated or truncated:
                break
        successes += int(outcome["success"])
    return {"episodes": episodes, "successes": successes, "transitions": transitions}


def train(arguments: argparse.Namespace) -> dict:
    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("schema") != "viki_hocap_rl_object_disjoint_v1":
        raise ValueError("unexpected HO-Cap RL split schema")
    if arguments.scene_id not in manifest["train_scenes"]:
        raise ValueError("training scene is not in the train partition")
    if arguments.object_name not in manifest["scene_objects"][arguments.scene_id]:
        raise ValueError("target object is not present in this scene")
    if arguments.timesteps < 1 or arguments.eval_episodes < 1:
        raise ValueError("training and evaluation counts must be positive")
    if arguments.reward_mode == "object_track" and arguments.expert_episodes:
        raise ValueError("pickup-only scripted seed cannot train a full object track")
    if arguments.expert_episodes and not (
        arguments.scene_id.startswith("subject_1_20231025_170105_")
        and arguments.object_name == "G10_4"
        and arguments.object_id == 4
        and arguments.gt_anchor_frame == 420
        and arguments.reset_frame == 44
        and arguments.plan.parent.name == "hocap_g10_4_orient_weight_003"
    ):
        raise ValueError("scripted contact seed is validated only for the G10_4 frame-44 pilot")
    environment = HocapGraspEnv(
        arguments.plan, arguments.track, arguments.models, arguments.mesh,
        arguments.gt_poses, arguments.object_name, arguments.object_id,
        arguments.gt_anchor_frame, arguments.reset_frame,
        reward_mode=arguments.reward_mode, max_steps=arguments.max_steps,
    )
    check_env(environment, warn=True)
    model = SAC(
        "MlpPolicy", environment, seed=arguments.seed, device="cpu", verbose=1,
        learning_starts=min(200, max(10, arguments.timesteps // 5)),
        batch_size=64, buffer_size=50000, train_freq=1, gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
    )
    expert = seed_contact_experience(
        model, environment, arguments.expert_episodes, arguments.seed,
    )
    model.learn(total_timesteps=arguments.timesteps)
    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint = arguments.output / "sac_pilot.zip"
    model.save(checkpoint)
    evaluations = []
    for episode_index in range(arguments.eval_episodes):
        state, _ = environment.reset(seed=arguments.seed + 1000 + episode_index)
        last_info = {}
        peak_lift = 0.0
        bilateral_steps = 0
        nearest_pad_distance_sum = float("inf")
        peak_pad_force_n = {"left": 0.0, "right": 0.0}
        total_reward = 0.0
        outside_track_sum = 0.0
        peak_outside_track = 0.0
        non_pad_contacts = 0
        for step_index in range(environment.max_steps):
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, last_info = environment.step(action)
            total_reward += reward
            outside_track_sum += last_info.get("outside_track_m", 0.0)
            peak_outside_track = max(peak_outside_track, last_info.get("outside_track_m", 0.0))
            non_pad_contacts += int(last_info.get("non_pad_contact", False))
            peak_lift = max(peak_lift, last_info["lift_m"])
            bilateral_steps += int(last_info["bilateral_contact"])
            nearest_pad_distance_sum = min(
                nearest_pad_distance_sum, sum(last_info["pad_distances_m"]),
            )
            for side in peak_pad_force_n:
                peak_pad_force_n[side] = max(
                    peak_pad_force_n[side], last_info["pad_forces_n"][side],
                )
            if terminated or truncated:
                break
        evaluations.append({
            "seed": arguments.seed + 1000 + episode_index,
            "steps": step_index + 1,
            "success": last_info["success"],
            "peak_lift_m": peak_lift,
            "bilateral_steps": bilateral_steps,
            "nearest_pad_distance_sum_m": nearest_pad_distance_sum,
            "peak_pad_force_n": peak_pad_force_n,
            "final_object_shift_xy_m": last_info["object_shift_xy_m"],
            "episode_return": total_reward,
            "mean_outside_track_m": outside_track_sum / (step_index + 1),
            "peak_outside_track_m": peak_outside_track,
            "final_goal_distance_m": last_info.get("goal_distance_m"),
            "has_lifted": last_info.get("has_lifted"),
            "non_pad_contact_steps": non_pad_contacts,
        })
    environment.close()
    inputs = {
        "plan": arguments.plan,
        "track": arguments.track,
        "mesh": arguments.mesh,
        "gt_poses": arguments.gt_poses,
    }
    report = {
        "scene_id": arguments.scene_id,
        "input_files": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        },
        "model_root": str(arguments.models),
        "object_id": arguments.object_id,
        "gt_anchor_frame": arguments.gt_anchor_frame,
        "reset_frame": arguments.reset_frame,
        "object_name": arguments.object_name,
        "reward_mode": arguments.reward_mode,
        "max_steps": environment.max_steps,
        "reference_duration_s": (environment.object_reference.duration_s
                                 if arguments.reward_mode == "object_track" else None),
        "reference_time_scale": (environment.object_reference.time_scale
                                 if arguments.reward_mode == "object_track" else None),
        "reference_lag_s": (environment.object_reference.lag_s
                            if arguments.reward_mode == "object_track" else None),
        "reference_radius_m": (environment.object_reference.radius_m
                               if arguments.reward_mode == "object_track" else None),
        "timesteps": arguments.timesteps,
        "seed": arguments.seed,
        "checkpoint": str(checkpoint),
        "evaluation": evaluations,
        "success_count": sum(result["success"] for result in evaluations),
        "evaluation_scope": "same training scene, randomized mass/friction/XY; NOT unseen-object test",
        "mass_provenance": manifest["mass_provenance"],
        "scripted_contact_seed": expert,
    }
    (arguments.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/hocap_rl_split.json"))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--gt-poses", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--gt-anchor-frame", type=int, required=True)
    parser.add_argument("--reset-frame", type=int, required=True)
    parser.add_argument("--reward-mode", choices=("pickup", "object_track"),
                        default="pickup")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--expert-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/app/rl-output/pilot"))
    print(json.dumps(train(parser.parse_args()), indent=2))
