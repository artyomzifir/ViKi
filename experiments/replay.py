"""Physics replay of one ViKi plan with free cubes and no object teleportation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import h5py
import mujoco
import numpy as np

from experiments.gripper_control import pressure_commands
from experiments.observations import ACTION_SCHEMA, OBSERVATION_SCHEMA, observation
from experiments.quality import cube_pose_errors
from experiments.scene import ARM_JOINTS, load_scene
from experiments.success import released_stack_success
from experiments.verify_gripper import MIMIC_MULTIPLIERS, PAD_OFFSET_M


def replay(episode: Path, model_root: Path, time_scale: float = 1.0,
           grasp_pressure_offset_rad: float = 0.0,
           *, capture_path: Path | None = None,
           control_hz: float = 50.0,
           policy: Callable[..., np.ndarray] | None = None,
           command_plan_path: Path | None = None,
           red_offset_xy: np.ndarray | None = None,
           green_offset_xy: np.ndarray | None = None,
           hold_commands: bool = False,
           policy_uses_time: bool = False,
           policy_horizon_s: float = 40.0) -> dict:
    if time_scale <= 0 or policy_horizon_s <= 0:
        raise ValueError("time scale and policy horizon must be positive")
    if control_hz <= 0 or (capture_path is not None and policy is not None):
        raise ValueError("invalid control rate or simultaneous policy and teacher capture")
    plan_path = episode / "plan.h5"
    object_model_path = (episode / "intermediates/segmentation/"
                         "sam2.1_hiera_small/object_models.npz")
    model = load_scene(plan_path, object_model_path, model_root,
                       red_offset_xy, green_offset_xy)
    data = mujoco.MjData(model)
    with h5py.File(command_plan_path or plan_path) as plan:
        approach = np.asarray(plan["q_approach"], dtype=float)
        joints = np.asarray(plan["q"], dtype=float)
        drive = np.asarray(plan["gripper_joint_position"], dtype=float)
        object_grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
        drive = pressure_commands(drive, object_grasp, grasp_pressure_offset_rad)
        timestamps = np.asarray(plan["timestamps"], dtype=float)
        frame_dt = float(plan["dt"][()])
    times = (timestamps - timestamps[0]) / 1e6
    if not np.all(np.diff(times) > 0):
        raise ValueError("plan timestamps must be strictly increasing")
    arm_qpos = [int(model.jnt_qposadr[model.joint(name).id]) for name in ARM_JOINTS]
    drive_qpos = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_qpos] = approach[0]
    for name, multiplier in MIMIC_MULTIPLIERS.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.0 * multiplier
    data.ctrl[:6] = approach[0]
    data.ctrl[6] = 0.0
    mujoco.mj_forward(model, data)
    blue_geom = model.geom("blue_geom").id
    green_geom = model.geom("green_geom").id
    red_geom = model.geom("red_geom").id
    blue_body = model.body("blue_cube").id
    green_body = model.body("green_cube").id
    left_pad_body = model.body("left_distal_phalanx").id
    right_pad_body = model.body("right_distal_phalanx").id
    pad_geoms = {
        model.geom("left_distal_phalanx_rubber_pad").id,
        model.geom("right_distal_phalanx_rubber_pad").id,
    }
    wrist_body = model.body("wrist_3_link").id
    blue_dof = int(model.jnt_dofadr[model.joint("blue_free").id])
    logs = {key: [] for key in ("time_s", "moved_xyz_m", "target_xyz_m",
                                 "moved_speed_m_s", "support_contact",
                                 "gripper_contact", "gripper_clear")}
    peak_height = float(data.xpos[blue_body, 2])
    initial_height = peak_height
    any_robot_contact = False
    max_robot_penetration = 0.0
    grasp_relative_origin = None
    max_airborne_slip = 0.0
    airborne_pad_gaps = []
    airborne_steps = 0
    bilateral_pad_steps = 0
    airborne_contact_gap = 0.0
    max_airborne_contact_gap = 0.0
    max_joint_error = 0.0
    max_drive_error = 0.0
    max_free_drive_error = 0.0
    invalid_reason = None
    warmup_s = 0.3
    approach_times = warmup_s + np.arange(len(approach)) * frame_dt
    demonstration_times = warmup_s + len(approach) * frame_dt + times
    command_times = np.concatenate(([0.0, warmup_s], approach_times[1:],
                                    demonstration_times)) * time_scale
    command_joints = np.concatenate((approach[:1], approach, joints))
    command_drive = np.concatenate((np.zeros(len(approach) + 1), drive))
    phase_bounds = object_grasp.get("phase_bounds") if object_grasp else None
    carry_time = (float(demonstration_times[phase_bounds[1]]) * time_scale
                  if phase_bounds else None)
    release_time = (float(demonstration_times[phase_bounds[2]]) * time_scale
                    if phase_bounds else None)
    retreat_time = (float(demonstration_times[phase_bounds[3]]) * time_scale
                    if phase_bounds else None)
    duration = float(command_times[-1] + 0.7) if policy is None else policy_horizon_s
    sample_steps = round(1.0 / (control_hz * model.opt.timestep))
    if sample_steps < 1 or not np.isclose(sample_steps * model.opt.timestep,
                                           1.0 / control_hz, atol=1e-9):
        raise ValueError("control rate must be an integer subdivision of physics rate")
    captured_observations = []
    captured_actions = []
    captured_times = []
    support_bodies = {"red": model.body("red_cube").id,
                      "green": model.body("green_cube").id}
    support_geoms = {"red": red_geom, "green": green_geom}
    robot_root = model.body("robot_base").id
    robot_geoms = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id and body_id != robot_root:
            body_id = int(model.body_parentid[body_id])
        if body_id == robot_root:
            robot_geoms.add(geom_id)
    support_contact_counts = {name: {"before_release": 0, "release": 0,
                                     "retreat": 0} for name in support_bodies}
    support_contact_steps = {name: 0 for name in support_bodies}
    support_contact_impulse = {name: 0.0 for name in support_bodies}
    support_contact_force_peak = {name: 0.0 for name in support_bodies}
    support_contact_penetration_peak = {name: 0.0 for name in support_bodies}
    support_contact_geoms = {name: {} for name in support_bodies}
    contact_force = np.zeros(6)
    support_contact_bodies = {name: set() for name in support_bodies}
    support_contact_span = {name: [None, None] for name in support_bodies}
    support_at_release = {}
    support_max_post_release_displacement = {name: 0.0 for name in support_bodies}
    steps = int(np.ceil(duration / model.opt.timestep))
    for step in range(steps):
        now = float(data.time)
        if policy is None and (not hold_commands and capture_path is None
                               or step % sample_steps == 0):
            for joint_index in range(len(ARM_JOINTS)):
                data.ctrl[joint_index] = np.interp(
                    now, command_times, command_joints[:, joint_index])
            data.ctrl[6] = np.interp(now, command_times, command_drive)
        elif step % sample_steps == 0:
            state = observation(model, data)
            command = np.asarray(policy(state, now) if policy_uses_time
                                 else policy(state), dtype=float)
            if command.shape != (7,) or not np.isfinite(command).all():
                raise ValueError("policy must return seven finite position targets")
            data.ctrl[:] = command
        if capture_path is not None and step % sample_steps == 0:
            captured_observations.append(observation(model, data))
            captured_actions.append(data.ctrl.copy())
            captured_times.append(now)
        mujoco.mj_step(model, data)
        max_joint_error = max(max_joint_error, float(np.max(
            np.abs(data.qpos[arm_qpos] - data.ctrl[:6]))))
        drive_error = float(abs(data.qpos[drive_qpos] - data.ctrl[6]))
        max_drive_error = max(max_drive_error, drive_error)
        if not np.isfinite(data.qpos).all() or max_joint_error > 0.5 or max_drive_error > 0.5:
            invalid_reason = "actuator tracking diverged; physics result is not evaluable"
            break
        blue_xyz = data.xpos[blue_body].copy()
        green_xyz = data.xpos[green_body].copy()
        if release_time is not None and now >= release_time:
            for name, body_id in support_bodies.items():
                if name not in support_at_release:
                    support_at_release[name] = data.xpos[body_id].copy()
                support_max_post_release_displacement[name] = max(
                    support_max_post_release_displacement[name],
                    float(np.linalg.norm(data.xpos[body_id]
                                         - support_at_release[name])))
        peak_height = max(peak_height, float(blue_xyz[2]))
        blue_contacts = []
        robot_contact_bodies = set()
        touching_pad_geoms = set()
        contacted_supports = set()
        for contact_index, contact in enumerate(data.contact[:data.ncon]):
            for name, geom_id in support_geoms.items():
                if geom_id not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                if other not in robot_geoms:
                    continue
                contacted_supports.add(name)
                geom_name = model.geom(other).name or f"unnamed_geom_{other}"
                support_contact_geoms[name][geom_name] = (
                    support_contact_geoms[name].get(geom_name, 0) + 1)
                support_contact_penetration_peak[name] = max(
                    support_contact_penetration_peak[name], max(0.0, -float(contact.dist)))
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                normal_force = max(0.0, float(contact_force[0]))
                support_contact_impulse[name] += normal_force * model.opt.timestep
                support_contact_force_peak[name] = max(
                    support_contact_force_peak[name], normal_force)
                phase = ("before_release" if release_time is None or now < release_time
                         else "release" if retreat_time is None or now < retreat_time
                         else "retreat")
                support_contact_counts[name][phase] += 1
                support_contact_bodies[name].add(
                    model.body(int(model.geom_bodyid[other])).name)
                if support_contact_span[name][0] is None:
                    support_contact_span[name][0] = now
                support_contact_span[name][1] = now
            if blue_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == blue_geom else contact.geom1
            blue_contacts.append(other)
            if other in pad_geoms:
                touching_pad_geoms.add(other)
            if other not in (green_geom, red_geom) and model.geom(other).name != "table":
                robot_contact_bodies.add(int(model.geom_bodyid[other]))
                max_robot_penetration = max(max_robot_penetration,
                                            max(0.0, -float(contact.dist)))
        for name in contacted_supports:
            support_contact_steps[name] += 1
        support = green_geom in blue_contacts
        if not touching_pad_geoms:
            max_free_drive_error = max(max_free_drive_error, drive_error)
        robot_contact = bool(robot_contact_bodies)
        any_robot_contact |= robot_contact
        pad_points = []
        for name in ("left_distal_phalanx", "right_distal_phalanx"):
            body_id = model.body(name).id
            pad_points.append(data.xpos[body_id]
                              + data.xmat[body_id].reshape(3, 3) @ PAD_OFFSET_M)
        clear = all(np.linalg.norm(point - blue_xyz) > 0.06
                    for point in pad_points)
        if blue_xyz[2] > initial_height + 0.025:
            airborne_steps += 1
            bilateral_pad_steps += pad_geoms.issubset(touching_pad_geoms)
            midpoint = (pad_points[0] + pad_points[1]) / 2
            wrist_rotation = data.xmat[wrist_body].reshape(3, 3)
            relative = wrist_rotation.T @ (blue_xyz - midpoint)
            if {left_pad_body, right_pad_body}.issubset(robot_contact_bodies):
                airborne_pad_gaps.append(float(np.linalg.norm(
                    pad_points[1] - pad_points[0])))
            if grasp_relative_origin is None and pad_geoms.issubset(touching_pad_geoms):
                grasp_relative_origin = relative
            if grasp_relative_origin is not None:
                max_airborne_slip = max(max_airborne_slip, float(np.linalg.norm(
                    relative - grasp_relative_origin)))
            airborne_contact_gap = 0.0 if robot_contact else (
                airborne_contact_gap + model.opt.timestep)
            max_airborne_contact_gap = max(max_airborne_contact_gap,
                                           airborne_contact_gap)
        else:
            airborne_contact_gap = 0.0
        logs["time_s"].append(float(data.time))
        logs["moved_xyz_m"].append(blue_xyz)
        logs["target_xyz_m"].append(green_xyz)
        logs["moved_speed_m_s"].append(float(np.linalg.norm(
            data.qvel[blue_dof:blue_dof + 3])))
        logs["support_contact"].append(support)
        logs["gripper_contact"].append(robot_contact)
        logs["gripper_clear"].append(clear)
    result = {
        "episode": episode.name,
        "control_mode": ("learned_policy" if policy is not None else
                         "source_template" if command_plan_path is not None else
                         "recorded_plan"),
        "command_plan": str(command_plan_path or plan_path),
        "red_offset_xy_m": (np.zeros(2) if red_offset_xy is None else
                            np.asarray(red_offset_xy, float)).tolist(),
        "green_offset_xy_m": (np.zeros(2) if green_offset_xy is None else
                              np.asarray(green_offset_xy, float)).tolist(),
        "control_hz": control_hz if policy is not None or capture_path is not None or hold_commands else None,
        "policy_uses_time": policy_uses_time if policy is not None else None,
        "command_time_scale": time_scale,
        "grasp_pressure_offset_rad": grasp_pressure_offset_rad,
        "physics_time_s": float(data.time),
        "blue_peak_z_m": peak_height,
        "blue_final_xyz_m": logs["moved_xyz_m"][-1].tolist(),
        "green_final_xyz_m": logs["target_xyz_m"][-1].tolist(),
        **cube_pose_errors(data.xpos[blue_body], data.xpos[green_body],
                           data.xmat[blue_body].reshape(3, 3),
                           data.xmat[green_body].reshape(3, 3)),
        "support_robot_contact_counts": support_contact_counts,
        "support_robot_contact_duration_s": {
            name: steps * model.opt.timestep for name, steps in support_contact_steps.items()},
        "support_robot_normal_impulse_Ns": support_contact_impulse,
        "support_robot_peak_normal_force_N": support_contact_force_peak,
        "support_robot_max_penetration_m": support_contact_penetration_peak,
        "support_robot_contact_geom_counts": support_contact_geoms,
        "support_robot_contact_bodies": {
            name: sorted(names) for name, names in support_contact_bodies.items()},
        "support_robot_contact_span_s": support_contact_span,
        "object_phase_times_s": {"carry": carry_time, "release": release_time,
                                 "retreat": retreat_time},
        "support_max_post_release_displacement_m":
            support_max_post_release_displacement,
        "any_robot_blue_contact": any_robot_contact,
        "final_blue_green_contact": bool(logs["support_contact"][-1]),
        "max_joint_tracking_error_rad": max_joint_error,
        "max_gripper_tracking_error_rad": max_drive_error,
        "max_free_gripper_tracking_error_rad": max_free_drive_error,
        "max_robot_blue_penetration_m": max_robot_penetration,
        "max_airborne_grasp_slip_m": max_airborne_slip,
        "median_airborne_pad_center_gap_m": (
            float(np.median(airborne_pad_gaps)) if airborne_pad_gaps else None),
        "bilateral_rubber_pad_contact_fraction": (
            bilateral_pad_steps / airborne_steps if airborne_steps else 0.0),
        "max_airborne_contact_gap_s": max_airborne_contact_gap,
        "invalid_reason": invalid_reason,
    }
    result["valid_tracking"] = (
        invalid_reason is None and max_joint_error <= 0.1
        and max_free_drive_error <= 0.1 and max_drive_error <= 0.25)
    result["valid_contact_physics"] = max_robot_penetration <= 0.004
    result["valid_grasp"] = (
        grasp_relative_origin is not None
        and max_airborne_slip <= 0.01
        and max_airborne_contact_gap <= 0.05
        and result["bilateral_rubber_pad_contact_fraction"] >= 0.95
        and result["valid_contact_physics"]
    )
    result["terminal_stack_success"] = (
        released_stack_success(**logs) if result["valid_tracking"] else None)
    result["released_stack_success"] = (
        result["terminal_stack_success"] and result["valid_grasp"]
        if result["valid_tracking"] and result["valid_contact_physics"] else None)
    if capture_path is not None and result["released_stack_success"]:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            capture_path,
            observation=np.asarray(captured_observations, dtype=np.float32),
            action=np.asarray(captured_actions, dtype=np.float32),
            time_s=np.asarray(captured_times, dtype=np.float64),
            observation_schema=OBSERVATION_SCHEMA,
            action_schema=ACTION_SCHEMA,
            source_episode=episode.name,
            control_hz=control_hz,
            command_time_scale=time_scale,
            grasp_pressure_offset_rad=grasp_pressure_offset_rad,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--grasp-pressure-offset-rad", type=float, default=0.0)
    arguments = parser.parse_args()
    print(json.dumps(replay(arguments.episode, arguments.models,
                            arguments.time_scale,
                            arguments.grasp_pressure_offset_rad), indent=2))
