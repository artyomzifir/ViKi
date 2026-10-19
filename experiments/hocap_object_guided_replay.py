"""Guided HO-Cap object transfer in MuJoCo, with optional contact attachment."""

from __future__ import annotations

import argparse
import hashlib

import h5py
import json
import tarfile
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.hocap_grasp_env import HocapGraspEnv
from experiments.object_trajectory_reward import ObjectTrajectoryReference


def _position_action(error_m: np.ndarray) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    action[:3] = np.clip(np.asarray(error_m, dtype=float) / 0.005, -1.0, 1.0)
    return action


def _step(environment: HocapGraspEnv, action: np.ndarray, metrics: dict) -> bool:
    _, _, terminated, truncated, info = environment.step(action)
    metrics["steps"] += 1
    metrics["peak_lift_m"] = max(metrics["peak_lift_m"], info["lift_m"])
    metrics["bilateral_steps"] += int(info["bilateral_contact"])
    metrics["non_pad_contact_steps"] += int(info["non_pad_contact"])
    if info["non_pad_contact"]:
        object_geom = environment.object_geom
        contacts = sorted({
            environment.model.body(environment.model.geom_bodyid[
                contact.geom2 if contact.geom1 == object_geom else contact.geom1
            ]).name
            for contact in environment.data.contact[:environment.data.ncon]
            if object_geom in (contact.geom1, contact.geom2)
            and (contact.geom2 if contact.geom1 == object_geom else contact.geom1)
            not in (*environment.pad_geoms, environment.table_geom)
        })
        metrics["non_pad_contact_geoms"].extend(
            name for name in contacts if name not in metrics["non_pad_contact_geoms"]
        )
    metrics["attached_steps"] += int(info["contact_attachment_active"])
    metrics["other_object_contact_ids"].update(info["other_object_contact_ids"])
    metrics["robot_table_contact_bodies"].update(info["robot_table_contact_bodies"])
    metrics["max_other_object_shift_m"] = max(
        metrics["max_other_object_shift_m"],
        max(info["other_object_shift_m"].values(), default=0.0),
    )
    metrics["peak_pad_force_n"] = {
        side: max(metrics["peak_pad_force_n"][side], force)
        for side, force in info["pad_forces_n"].items()
    }
    metrics["last_info"] = info
    return not (truncated or (terminated and not info["success"]))


def _force_command(force_n: dict[str, float], joint_target_rad: float,
                   target_force_n: float) -> float:
    if min(force_n.values()) < target_force_n - 0.2 and joint_target_rad < 0.78:
        return 1.0
    if min(force_n.values()) > target_force_n + 1.0 and joint_target_rad > 0.1:
        return -1.0
    return 0.0


def run_trial(
    environment: HocapGraspEnv,
    seed: int,
    offset_xy_m: tuple[float, float],
    closing_joint_rad: float,
    carry_time_scale: float,
    target_pad_force_n: float = 2.0,
    lift_action_z: float = 0.5,
    offset_z_m: float = 0.0,
    force_during_lift: bool = False,
    wrist_roll_deg: float = 0.0,
    start_from_current: bool = False,
    place_on_table: bool = False,
) -> dict:
    """Run one reset-pose ablation, not a collision-planned approach."""
    if (np.asarray(offset_xy_m).shape != (2,)
            or not np.isfinite(offset_xy_m).all()
            or not np.isfinite(wrist_roll_deg) or abs(wrist_roll_deg) > 90.0
            or not 0.1 <= closing_joint_rad <= 0.8 or carry_time_scale < 2.0
            or not 0.5 <= target_pad_force_n <= 4.0
            or not 0.1 <= lift_action_z <= 1.0
            or not -0.04 <= offset_z_m <= 0.02):
        raise ValueError("invalid grasp command, force target or time scale")
    reset_info = (
        {"mass_kg": float(environment.model.body_mass[environment.object_body]),
         "friction": float(environment.model.geom_friction[environment.object_geom, 0])}
        if start_from_current else environment.reset(seed=seed)[1]
    )
    if wrist_roll_deg:
        wrist_joint = environment.arm_joints[-1]
        wrist_qpos = environment.arm_qpos[-1]
        wrist_target = (environment.data.qpos[wrist_qpos]
                        + np.deg2rad(wrist_roll_deg))
        if (environment.model.jnt_limited[wrist_joint]
                and not environment.model.jnt_range[wrist_joint, 0] <= wrist_target
                <= environment.model.jnt_range[wrist_joint, 1]):
            raise ValueError("candidate wrist roll exceeds robot joint limits")
        environment.data.qpos[wrist_qpos] = wrist_target
        environment.data.ctrl[5] = wrist_target
        mujoco.mj_forward(environment.model, environment.data)
    start_position = environment.data.xpos[environment.object_body].copy()
    start_reference = environment.object_reference.positions_m[0]
    target_midpoint = start_position.copy()
    target_midpoint[:2] += offset_xy_m
    target_midpoint[2] = np.mean(environment._pad_positions(), axis=0)[2] + offset_z_m
    metrics = {
        "seed": seed, "offset_xy_m": list(offset_xy_m), "offset_z_m": offset_z_m,
        "mass_kg": reset_info["mass_kg"], "object_friction": reset_info["friction"],
        "closing_joint_rad": closing_joint_rad,
        "carry_time_scale": carry_time_scale,
        "target_pad_force_n": target_pad_force_n,
        "lift_action_z": lift_action_z,
        "force_during_lift": force_during_lift,
        "wrist_roll_deg": wrist_roll_deg,
        "steps": 0, "peak_lift_m": 0.0, "bilateral_steps": 0,
        "non_pad_contact_steps": 0, "attached_steps": 0,
        "non_pad_contact_geoms": [],
        "other_object_contact_ids": set(), "max_other_object_shift_m": 0.0,
        "robot_table_contact_bodies": set(),
        "peak_pad_force_n": {"left": 0.0, "right": 0.0},
        "failure_phase": None, "last_info": {},
    }

    for _ in range(6 if offset_z_m == 0.0 else 10):
        midpoint = np.mean(environment._pad_positions(), axis=0)
        action = _position_action(target_midpoint - midpoint)
        if not _step(environment, action, metrics):
            metrics["failure_phase"] = "align"
            break

    if metrics["failure_phase"] is None:
        for _ in range(25):
            action = np.zeros(5, dtype=np.float32)
            action[4] = float(environment.data.ctrl[6] < closing_joint_rad - 0.025)
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "close"
                break
            if environment.data.ctrl[6] >= closing_joint_rad - 0.025:
                break

    if metrics["failure_phase"] is None:
        for _ in range(10):
            action = np.zeros(5, dtype=np.float32)
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "settle"
                break

    if metrics["failure_phase"] is None:
        force = metrics["last_info"]["pad_forces_n"]
        if min(force.values()) < 0.2 and not metrics["last_info"]["contact_attachment_active"]:
            metrics["failure_phase"] = "no_bilateral_preload"

    if metrics["failure_phase"] is None:
        for _ in range(60):
            action = np.zeros(5, dtype=np.float32)
            action[2] = lift_action_z
            if (force_during_lift and not metrics["last_info"]["contact_attachment_active"]
                    and metrics["last_info"]["lift_m"] >= 0.02
                    and metrics["last_info"]["bilateral_contact"]):
                action[4] = _force_command(metrics["last_info"]["pad_forces_n"],
                                           float(environment.data.ctrl[6]), target_pad_force_n)
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "lift"
                break
            if (metrics["last_info"]["stable_lift_steps"] >= 13
                    and metrics["last_info"]["lift_m"] >= 0.06):
                break
        else:
            metrics["failure_phase"] = "no_stable_lift"

    if metrics["failure_phase"] is None:
        reference = environment.object_reference
        lifted_position = environment.data.xpos[environment.object_body].copy()
        reference_index = int(np.argmin(np.abs(
            reference.positions_m[:30, 2] - lifted_position[2],
        )))
        reference_start = reference.positions_m[reference_index]
        reference_start_time_s = reference.times_s[reference_index]
        duration_s = (reference.duration_s - reference_start_time_s) * carry_time_scale / 2.0
        carry_steps = int(np.ceil(duration_s / 0.04))
        grip_offset = (
            np.mean(environment._pad_positions(), axis=0)
            - lifted_position
        )
        metrics["reference_join_frame_from_reset"] = reference_index
        metrics["reference_anchor_shift_m"] = (
            lifted_position - reference_start
        ).tolist()
        wrist_rotation = environment.data.xmat[environment.wrist_body].reshape(3, 3)
        relative_start = wrist_rotation.T @ (lifted_position - np.mean(
            environment._pad_positions(), axis=0,
        ))
        metrics["max_carry_relative_slip_m"] = 0.0
        lost_contact_steps = 0
        for carry_step in range(carry_steps):
            reference_time_s = min(
                reference_start_time_s + (carry_step + 1) * 0.04 * 2.0 / carry_time_scale,
                reference.duration_s,
            )
            desired_object = lifted_position + (
                reference.target(reference_time_s) - reference_start
            )
            pad_midpoint = np.mean(environment._pad_positions(), axis=0)
            action = _position_action(desired_object + grip_offset - pad_midpoint)
            if not metrics["last_info"]["contact_attachment_active"]:
                action[4] = _force_command(metrics["last_info"]["pad_forces_n"],
                                           float(environment.data.ctrl[6]), target_pad_force_n)
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "carry"
                break
            wrist_rotation = environment.data.xmat[environment.wrist_body].reshape(3, 3)
            relative_now = wrist_rotation.T @ (
                environment.data.xpos[environment.object_body]
                - np.mean(environment._pad_positions(), axis=0)
            )
            metrics["max_carry_relative_slip_m"] = max(
                metrics["max_carry_relative_slip_m"],
                float(np.linalg.norm(relative_now - relative_start)),
            )
            lost_contact_steps = (0 if (metrics["last_info"]["bilateral_contact"]
                                         or metrics["last_info"]["contact_attachment_active"])
                                  else lost_contact_steps + 1)
            if lost_contact_steps >= 8:
                metrics["failure_phase"] = "lost_grasp"
                break

    if metrics["failure_phase"] is None and place_on_table:
        metrics["placement_initial_gap_m"] = float(mujoco.mj_geomDistance(
            environment.model, environment.data, environment.object_geom,
            environment.table_geom, 0.2, None,
        ))
        metrics["placement_descent_steps"] = 0
        for _ in range(50):
            gap = float(mujoco.mj_geomDistance(
                environment.model, environment.data, environment.object_geom,
                environment.table_geom, 0.2, None,
            ))
            if gap <= 0.001:
                break
            action = np.zeros(5, dtype=np.float32)
            action[2] = -min(1.0, max(0.1, (gap - 0.001) / 0.005))
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "placement_descent"
                break
            metrics["placement_descent_steps"] += 1
            if (environment.attachment is not None
                    and not environment.attachment.attached):
                metrics["failure_phase"] = "lost_attachment_during_descent"
                break
        else:
            metrics["failure_phase"] = "placement_descent_timeout"
        metrics["placement_final_gap_m"] = float(mujoco.mj_geomDistance(
            environment.model, environment.data, environment.object_geom,
            environment.table_geom, 0.2, None,
        ))

    if metrics["failure_phase"] is None:
        metrics["object_position_before_release_m"] = (
            environment.data.xpos[environment.object_body].tolist()
        )
        metrics["object_rotation_before_release"] = (
            environment.data.xmat[environment.object_body].reshape(3, 3).tolist()
        )
        metrics["pad_midpoint_before_release_m"] = (
            np.mean(environment._pad_positions(), axis=0).tolist()
        )
        metrics["table_support_before_release"] = environment._contacts()[1]
        metrics["release_trace"] = []
        for _ in range(18):
            action = np.zeros(5, dtype=np.float32)
            action[4] = -1.0
            continues = _step(environment, action, metrics)
            metrics["release_trace"].append({
                "time_s": float(environment.data.time),
                "finger_command_rad": float(environment.data.ctrl[6]),
                "finger_position_rad": float(environment.data.qpos[environment.finger_qpos]),
                "object_position_m": environment.data.xpos[environment.object_body].tolist(),
                "object_up_z": float(environment.data.xmat[environment.object_body, 8]),
                "attached": metrics["last_info"]["contact_attachment_active"],
                "table_support": metrics["last_info"]["table_support"],
                "pad_forces_n": metrics["last_info"]["pad_forces_n"],
            })
            if not continues:
                metrics["failure_phase"] = "release"
                break

    if metrics["failure_phase"] is None:
        for _ in range(15):
            action = np.zeros(5, dtype=np.float32)
            action[2] = 1.0
            if not _step(environment, action, metrics):
                metrics["failure_phase"] = "withdraw"
                break

    final_position = environment.data.xpos[environment.object_body].copy()
    metrics["final_object_position_m"] = final_position.tolist()
    metrics["final_other_object_position_m"] = {
        other_id: environment.data.xpos[
            environment.model.body(f"tracked_object_{other_id}").id
        ].tolist()
        for other_id in environment.other_object_joints
    }
    target_position = start_position + (
        environment.object_reference.positions_m[-1] - start_reference
    )
    final_rotation = environment.data.xmat[environment.object_body].reshape(3, 3)
    metrics["final_object_rotation_matrix"] = final_rotation.tolist()
    target_rotation = Rotation.from_matrix(np.asarray(
        environment.geometry["reference_final_rotation"], dtype=float,
    ))
    metrics["final_translation_error_m"] = float(np.linalg.norm(final_position - target_position))
    metrics["final_orientation_error_deg"] = float(np.rad2deg((
        target_rotation.inv() * Rotation.from_matrix(final_rotation)
    ).magnitude()))
    final_forces, table_support, _ = environment._contacts()
    metrics["final_table_support"] = bool(table_support)
    metrics["final_object_speed_m_s"] = float(np.linalg.norm(
        environment.data.qvel[environment.object_dof:environment.object_dof + 3],
    ))
    metrics["released"] = all(force < 0.05 for force in final_forces.values())
    metrics["attachment"] = (
        {"activated": environment.attachment.attach_time_s is not None,
         "attached_at_s": environment.attachment.attach_time_s,
         "released_at_s": environment.attachment.release_time_s,
         "forces_at_attach_n": environment.attachment.attach_forces_n,
         "pose_jump_at_attach_m": environment.attachment.attach_pose_jump_m,
         "still_attached": environment.attachment.attached}
        if environment.attachment is not None else None
    )
    metrics["other_object_contact_ids"] = sorted(metrics["other_object_contact_ids"])
    metrics["robot_table_contact_bodies"] = sorted(metrics["robot_table_contact_bodies"])
    metrics["physical_transfer_proxy"] = bool(
        environment.attachment is None
        and metrics["failure_phase"] is None
        and metrics["peak_lift_m"] >= 0.025
        and metrics["bilateral_steps"] >= 13
        and metrics.get("max_carry_relative_slip_m", float("inf")) < 0.01
        and metrics["final_translation_error_m"] <= 0.05
        and metrics["final_orientation_error_deg"] <= 30.0
        and metrics["final_table_support"]
        and metrics["final_object_speed_m_s"] <= 0.03
        and metrics["released"]
        and metrics["non_pad_contact_steps"] == 0
    )
    metrics["assisted_transfer_proxy"] = bool(
        environment.attachment is not None
        and metrics["attachment"]["activated"]
        and metrics["attachment"]["released_at_s"] is not None
        and not metrics["attachment"]["still_attached"]
        and metrics["failure_phase"] is None
        and metrics["peak_lift_m"] >= 0.025
        and metrics["final_translation_error_m"] <= 0.05
        and metrics["final_orientation_error_deg"] <= 30.0
        and metrics["released"]
        and metrics["final_table_support"]
        and metrics["final_object_speed_m_s"] <= 0.03
        and metrics["non_pad_contact_steps"] == 0
        and not metrics["other_object_contact_ids"]
        and not metrics["robot_table_contact_bodies"]
        and metrics["max_other_object_shift_m"] <= 0.005
    )
    return metrics


def prepare_full_scene(environment: HocapGraspEnv, plan_path: Path,
                       track_path: Path, object_id: int, grasp_frame: int,
                       seed: int, render_frame=None) -> dict:
    environment.reset(seed=seed)
    with h5py.File(plan_path) as plan:
        joints = np.asarray(plan["q"][:grasp_frame + 1], dtype=float)
        timestamps = np.asarray(plan["timestamps"], dtype=float)
    if len(joints) != grasp_frame + 1 or grasp_frame < 1:
        raise ValueError("invalid full-scene approach window")
    with np.load(track_path, allow_pickle=False) as track:
        object_ids = list(track["object_ids"])
        object_index = object_ids.index(object_id)
        positions = np.asarray(track["translation_world"][:, object_index], dtype=float)
        rotations = np.asarray(track["rotation_world_object"], dtype=float)
        all_positions = np.asarray(track["translation_world"], dtype=float)
    pad_geoms = list(environment.pad_geoms)
    start_frame = None
    start_clearances = None
    for frame in range(grasp_frame):
        environment.data.qpos[environment.arm_qpos] = joints[frame]
        environment.data.qvel[:] = 0.0
        mujoco.mj_forward(environment.model, environment.data)
        clearances = [float(mujoco.mj_geomDistance(
            environment.model, environment.data, pad_geom,
            environment.table_geom, 0.1, None,
        )) for pad_geom in pad_geoms]
        if min(clearances) >= 0.005:
            start_frame = frame
            start_clearances = clearances
            break
    if start_frame is None:
        raise RuntimeError("no collision-free recorded approach start")
    environment.data.qpos[environment.object_qpos:environment.object_qpos + 3] = (
        all_positions[start_frame, object_index]
    )
    environment.data.qpos[environment.object_qpos + 3:environment.object_qpos + 7] = (
        Rotation.from_matrix(rotations[start_frame, object_index])
        .as_quat(scalar_first=True)
    )
    for other_id, address in environment.other_object_joints.items():
        index = object_ids.index(other_id)
        environment.data.qpos[address:address + 3] = all_positions[start_frame, index]
        environment.data.qpos[address + 3:address + 7] = (
            Rotation.from_matrix(rotations[start_frame, index])
            .as_quat(scalar_first=True)
        )
    environment.data.ctrl[:6] = joints[start_frame]
    mujoco.mj_forward(environment.model, environment.data)
    table_projection_m = {}
    for other_geom, other_id in environment.other_object_geoms.items():
        gap = float(mujoco.mj_geomDistance(
            environment.model, environment.data, other_geom,
            environment.table_geom, 0.2, None,
        ))
        if abs(gap) > 0.05:
            raise ValueError(f"other object {other_id} is over 5 cm from the table")
        environment.data.qpos[environment.other_object_joints[other_id] + 2] -= gap
        table_projection_m[other_id] = -gap
    mujoco.mj_forward(environment.model, environment.data)
    before_settle = {
        other_id: environment.data.xpos[
            environment.model.body(f"tracked_object_{other_id}").id
        ].copy()
        for other_id in environment.other_object_joints
    }
    warmup_other_contacts = set()
    warmup_robot_table_contacts = set()
    warmup_target_contacts = 0
    for _ in range(round(2.0 / environment.model.opt.timestep)):
        mujoco.mj_step(environment.model, environment.data)
        warmup_other_contacts.update(environment._other_object_contacts())
        warmup_robot_table_contacts.update(environment._robot_table_contacts())
        forces, _, non_pad_contact = environment._contacts()
        warmup_target_contacts += int(non_pad_contact or max(forces.values()) >= 0.2)
    settling = {
        other_id: float(np.linalg.norm(
            environment.data.xpos[environment.model.body(f"tracked_object_{other_id}").id]
            - position,
        ))
        for other_id, position in before_settle.items()
    }
    environment.initial_other_positions = {
        other_id: environment.data.xpos[
            environment.model.body(f"tracked_object_{other_id}").id
        ].copy()
        for other_id in environment.other_object_joints
    }
    peak_joint_error = 0.0
    other_contacts = set()
    target_contacts = 0
    robot_table_contacts = 0
    for frame in range(start_frame + 1, grasp_frame + 1):
        environment.data.ctrl[:6] = joints[frame]
        for _ in range(round(0.08 / environment.model.opt.timestep)):
            mujoco.mj_step(environment.model, environment.data)
            other_contacts.update(environment._other_object_contacts())
            forces, _, non_pad_contact = environment._contacts()
            target_contacts += int(non_pad_contact or max(forces.values()) >= 0.2)
            robot_table_contacts += int(bool(environment._robot_table_contacts()))
        if render_frame is not None:
            render_frame()
        peak_joint_error = max(peak_joint_error, float(np.max(np.abs(
            environment.data.qpos[environment.arm_qpos] - joints[frame],
        ))))
    environment.reference_joint = joints[-1].copy()
    environment.object_reference = ObjectTrajectoryReference(
        timestamps, positions, grasp_frame,
    )
    environment.initial_position = environment.data.xpos[environment.object_body].copy()
    environment.initial_height = float(environment.initial_position[2])
    environment.step_count = 0
    approach_shifts = {
        other_id: float(np.linalg.norm(
            environment.data.xpos[environment.model.body(f"tracked_object_{other_id}").id]
            - position,
        ))
        for other_id, position in environment.initial_other_positions.items()
    }
    return {
        "input_frames": len(timestamps),
        "start_frame": start_frame,
        "grasp_frame": grasp_frame,
        "skipped_intersecting_preamble_frames": start_frame,
        "initial_pad_table_clearance_m": start_clearances,
        "other_object_table_projection_z_m": table_projection_m,
        "settling_other_object_shift_m": settling,
        "warmup_other_contact_ids": sorted(warmup_other_contacts),
        "warmup_robot_table_contact_bodies": sorted(warmup_robot_table_contacts),
        "warmup_target_contact_steps": warmup_target_contacts,
        "approach_other_object_shift_m": approach_shifts,
        "approach_other_contact_ids": sorted(other_contacts),
        "approach_target_contact_steps": target_contacts,
        "approach_robot_table_contact_steps": robot_table_contacts,
        "approach_peak_joint_error_rad": peak_joint_error,
        "geometrically_clear": (
            not warmup_other_contacts and not warmup_robot_table_contacts
            and warmup_target_contacts == 0
            and not other_contacts and target_contacts == 0
            and robot_table_contacts == 0
            and max(approach_shifts.values(), default=0.0) <= 0.005
        ),
    }


def run_search(report_path: Path, output_path: Path, seed: int,
               carry_time_scale: float = 4.0, target_pad_force_n: float = 0.5,
               lift_action_z: float = 1.0, offset_z_m: float = 0.0,
               force_during_lift: bool = False,
               video_path: Path | None = None,
               wrist_roll_deg: float = 0.0,
               contact_attach: bool = False,
               video_candidate: tuple[float, float, float] = (0.005, 0.0, 0.5),
               full_scene: bool = False,
               release_opening_rad: float = 0.12,
                          full_scene_grid: bool = False,
               full_scene_meshes: bool = False,
               mesh_archive_path: Path = Path("data/_hocap_assets/models.tar"),
               descriptive: bool = False) -> dict:
    source = json.loads(report_path.read_text())
    inputs = {name: Path(details["path"]) for name, details in source["input_files"].items()}
    for name, path in inputs.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["input_files"][name]["sha256"]:
            raise ValueError(f"source input changed: {name}")
    if full_scene_meshes and not full_scene:
        raise ValueError("official meshes for all objects require --full-scene")
    mesh_sources = {}
    if full_scene_meshes:
        with tempfile.TemporaryDirectory() as directory:
            with np.load(inputs["track"], allow_pickle=False) as track:
                other_ids = [int(value) for value in track["object_ids"]
                             if int(value) != source["object_id"]]
            other_mesh_paths = {}
            group_name = source["object_name"].rsplit("_", 1)[0]
            with tarfile.open(mesh_archive_path) as archive:
                for tracked_id in other_ids:
                    member_name = f"{group_name}_{tracked_id}.cleaned_mesh_2000.obj"
                    member = archive.extractfile(member_name)
                    if member is None:
                        raise ValueError(f"missing official mesh: {member_name}")
                    content = member.read()
                    path = Path(directory) / member_name
                    path.write_bytes(content)
                    other_mesh_paths[tracked_id] = path
                    mesh_sources[tracked_id] = {
                        "source": f"{mesh_archive_path}!{member_name}",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
            environment = HocapGraspEnv(
                inputs["plan"], inputs["track"], Path(source["model_root"]),
                inputs["mesh"], inputs["gt_poses"], source["object_name"],
                source["object_id"], source["gt_anchor_frame"], 0,
                reward_mode="object_track", max_steps=1000,
                contact_attach=contact_attach, all_objects=True,
                release_opening_rad=release_opening_rad,
                other_mesh_paths=other_mesh_paths,
            )
    else:
        environment = HocapGraspEnv(
            inputs["plan"], inputs["track"], Path(source["model_root"]),
            inputs["mesh"], inputs["gt_poses"], source["object_name"],
            source["object_id"], source["gt_anchor_frame"],
            0 if full_scene else source["reset_frame"],
            reward_mode="object_track", max_steps=1000 if full_scene else 650,
            contact_attach=contact_attach, all_objects=full_scene,
            release_opening_rad=release_opening_rad,
        )
    renderer = writer = render_frame = None
    try:
        if video_path is not None:
            import cv2

            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(environment.model, height=480, width=640)
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 480),
            )
            if not writer.isOpened():
                raise RuntimeError("could not create diagnostic video")
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = 1.15 if full_scene else 0.7
            camera.azimuth = 130.0 if full_scene else 95.0
            camera.elevation = -70.0 if full_scene else -25.0
            side_camera = mujoco.MjvCamera()
            side_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            side_camera.distance = 0.9 if full_scene else 0.7
            side_camera.azimuth = 0.0
            side_camera.elevation = -20.0
            original_step = environment.step

            def render_frame():
                object_position = environment.data.xpos[environment.object_body]
                camera.lookat[:] = object_position
                side_camera.lookat[:] = object_position
                renderer.update_scene(environment.data, camera=camera)
                first_view = renderer.render().copy()
                renderer.update_scene(environment.data, camera=side_camera)
                second_view = renderer.render()
                pair = np.concatenate((first_view, second_view), axis=1)
                writer.write(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

            def step_with_recording(action):
                transition = original_step(action)
                render_frame()
                return transition

            environment.step = step_with_recording
        trials = []
        candidates = ([video_candidate] if video_path is not None else
                      [(0.005, 0.0, 0.5)] if full_scene and not full_scene_grid else [
            (offset_x, offset_y, closing_joint)
            for offset_x in (-0.005, 0.0, 0.005)
            for offset_y in (-0.005, 0.0, 0.005)
            for closing_joint in (0.5, 0.55, 0.6)
        ])
        prelude = None
        for offset_x, offset_y, closing_joint in candidates:
            if full_scene:
                prelude = prepare_full_scene(
                    environment, inputs["plan"], inputs["track"],
                    source["object_id"], source["reset_frame"], seed, render_frame,
                )
                if not prelude["geometrically_clear"]:
                    break
            trial = run_trial(
                environment, seed, (offset_x, offset_y), closing_joint,
                carry_time_scale, target_pad_force_n, lift_action_z,
                offset_z_m, force_during_lift, wrist_roll_deg,
                start_from_current=full_scene,
                place_on_table=full_scene_meshes,
            )
            if prelude is not None:
                trial["prelude"] = prelude
            if descriptive:
                trial["physical_transfer_proxy"] = None
                trial["assisted_transfer_proxy"] = None
            trials.append(trial)
        result = {
            "protocol": ("full-scene HO-Cap object-guided grasp replay" if full_scene
                         else "one-scene oracle-object guided physical grasp search"),
            "input_report": str(report_path), "seed": seed,
            "input_files": source["input_files"],
            "carry_time_scale": carry_time_scale,
            "target_pad_force_n": target_pad_force_n,
            "lift_action_z": lift_action_z,
            "offset_z_m": offset_z_m,
            "force_during_lift": force_during_lift,
            "wrist_roll_deg": wrist_roll_deg,
            "contact_attach": contact_attach,
            "release_opening_rad": release_opening_rad,
            "full_scene": full_scene,
            "full_scene_grid": full_scene_grid,
            "full_scene_meshes": full_scene_meshes,
            "evaluation_mode": "descriptive_no_acceptance_limits" if descriptive else "gated",
            "place_on_table_before_release": full_scene_meshes,
            "table_z_m": environment.geometry["table_z_m"],
            "collision_geometry": {
                "target": environment.geometry["collision_geometry"],
                "others": environment.geometry["other_object_geometry"],
            },
            "all_object_geoms_are_meshes": all(
                environment.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
                for geom_id in [environment.object_geom, *environment.other_object_geoms]
            ) if full_scene else False,
            "mesh_sources": mesh_sources,
            "mesh_alignment": {
                tracked_id: {
                    "median_mm": details["cloud_to_mesh_vertex_median_mm"],
                    "p95_mm": details["cloud_to_mesh_vertex_p95_mm"],
                }
                for tracked_id, details in environment.geometry["other_mesh_details"].items()
            },
            "prelude": prelude,
            "video_candidate": list(video_candidate) if video_path is not None else None,
            "video_path": str(video_path) if video_path is not None else None,
            "trials": trials,
            "successes": (None if descriptive else sum(
                trial["physical_transfer_proxy"] for trial in trials
            )),
            "assisted_successes": (None if descriptive else sum(
                trial["assisted_transfer_proxy"] for trial in trials
            )),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--carry-time-scale", type=float, default=4.0)
    parser.add_argument("--target-pad-force-n", type=float, default=0.5)
    parser.add_argument("--lift-action-z", type=float, default=1.0)
    parser.add_argument("--offset-z-m", type=float, default=0.0)
    parser.add_argument("--force-during-lift", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--wrist-roll-deg", type=float, default=0.0)
    parser.add_argument("--contact-attach", action="store_true")
    parser.add_argument("--full-scene", action="store_true")
    parser.add_argument("--full-scene-grid", action="store_true")
    parser.add_argument("--full-scene-meshes", action="store_true")
    parser.add_argument("--mesh-archive", type=Path,
                        default=Path("data/_hocap_assets/models.tar"))
    parser.add_argument("--descriptive", action="store_true")
    parser.add_argument("--release-opening-rad", type=float, default=0.12)
    parser.add_argument("--video-offset-x", type=float, default=0.005)
    parser.add_argument("--video-offset-y", type=float, default=0.0)
    parser.add_argument("--video-close-rad", type=float, default=0.5)
    arguments = parser.parse_args()
    report = run_search(
        arguments.source_report, arguments.output, arguments.seed,
        arguments.carry_time_scale, arguments.target_pad_force_n,
        arguments.lift_action_z, arguments.offset_z_m, arguments.force_during_lift,
        arguments.video, arguments.wrist_roll_deg, arguments.contact_attach,
        (arguments.video_offset_x, arguments.video_offset_y, arguments.video_close_rad),
        arguments.full_scene, arguments.release_opening_rad,
        arguments.full_scene_grid, arguments.full_scene_meshes,
        arguments.mesh_archive, arguments.descriptive,
    )
    print(json.dumps({"output": str(arguments.output), "successes": report["successes"],
                      "assisted_successes": report["assisted_successes"],
                      "trials": len(report["trials"])}, indent=2))
