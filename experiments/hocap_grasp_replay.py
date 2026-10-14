"""One-scene HO-Cap grasp A/B in MuJoCo using a fitted box proxy.

The proxy and tabletop are inferred from a partial tracked point cloud; this
experiment is not an HO-Cap mesh or a calibrated force/contact benchmark.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.assemble_urdf import assembled_xml
from experiments.gripper_control import pressure_commands
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS, PAD_OFFSET_M


def _scene(plan_path: Path, track_path: Path, model_root: Path,
           object_id: int, mass_kg: float):
    if not 0.01 <= mass_kg <= 1.0:
        raise ValueError("proxy mass must be in [0.01, 1.0] kg")
    with h5py.File(plan_path) as plan:
        base_position = np.asarray(plan["base_position_calibration"], dtype=float)
        base_rpy = np.deg2rad(np.asarray(plan["base_rpy_deg_calibration"], dtype=float))
    with np.load(track_path, allow_pickle=False) as track:
        indices = np.flatnonzero(track["object_ids"] == object_id)
        if len(indices) != 1:
            raise ValueError("object ID is absent or ambiguous")
        object_index = int(indices[0])
        offsets = track["model_offsets"]
        surface = np.asarray(track["model_xyz_object"][offsets[object_index]:offsets[object_index + 1]], dtype=float)
        origin = np.asarray(track["translation_world"][0, object_index], dtype=float)
        orientation = np.asarray(track["rotation_world_object"][0, object_index], dtype=float)
        reference_final = np.asarray(track["translation_world"][-1, object_index], dtype=float)
        reference_final_rotation = np.asarray(track["rotation_world_object"][-1, object_index], dtype=float)
    low, high = np.quantile(surface, [0.05, 0.95], axis=0)
    half_size = (high - low) / 2
    local_center = (low + high) / 2
    if np.any(half_size < 0.005) or np.any(half_size > 0.15):
        raise ValueError("fitted object proxy has implausible dimensions")
    box_corners = np.array(np.meshgrid(*[[-value, value] for value in half_size])).reshape(3, -1).T
    world_corners = (box_corners + local_center) @ orientation.T + origin
    table_z = float(np.min(world_corners[:, 2]))
    with tempfile.TemporaryDirectory() as directory:
        urdf_path = Path(directory) / "ur10_robotiq.urdf"
        urdf_path.write_text(assembled_xml(model_root))
        root = ET.fromstring(mujoco.MjSpec.from_file(str(urdf_path)).to_xml())
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.0005")
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    robot = ET.Element("body", name="robot_base",
                       pos=" ".join(map(str, base_position)),
                       euler=" ".join(map(str, base_rpy)))
    for child in list(world):
        world.remove(child)
        robot.append(child)
    for body in robot.iter("body"):
        body.set("gravcomp", "1")
    world.append(robot)
    ET.SubElement(world, "geom", name="table", type="box",
                  pos=f"{origin[0]} {origin[1]} {table_z - 0.025}",
                  size="0.6 0.6 0.025", friction="1.2 0.005 0.0001")
    item = ET.SubElement(world, "body", name="tracked_object",
                         pos=" ".join(map(str, origin)),
                         quat=" ".join(map(str, Rotation.from_matrix(orientation).as_quat(scalar_first=True))))
    ET.SubElement(item, "freejoint", name="tracked_object_free")
    ET.SubElement(item, "geom", name="tracked_object_geom", type="box",
                  pos=" ".join(map(str, local_center)),
                  size=" ".join(map(str, half_size)),
                  mass=str(mass_kg), friction="1.2 0.005 0.0001", rgba="0.3 0.65 0.8 1")
    contacts = ET.SubElement(root, "contact")
    gripper_bodies = [body.get("name") for body in robot.iter("body")
                      if "phalanx" in body.get("name", "") or "knuckle" in body.get("name", "")]
    for first, second in combinations(["wrist_3_link", *gripper_bodies], 2):
        ET.SubElement(contacts, "exclude", body1=first, body2=second)
    ET.SubElement(contacts, "exclude", body1="forearm_link", body2="wrist_2_link")
    actuators = ET.SubElement(root, "actuator")
    for name in ARM_JOINTS:
        ET.SubElement(actuators, "position", name=f"cmd_{name}", joint=name,
                      kp="1000", kv="100")
    ET.SubElement(actuators, "position", name="cmd_finger", joint="finger_joint",
                  kp="50", kv="5")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), {
        "proxy_half_size_m": half_size.tolist(), "table_z_m": table_z,
        "proxy_mass_kg": mass_kg,
        "excluded_nominal_robot_pair": ["forearm_link", "wrist_2_link"],
        "initial_object_position_m": origin.tolist(),
        "reference_final_position_m": reference_final.tolist(),
        "reference_final_rotation": reference_final_rotation.tolist(),
    }


def replay(plan_path: Path, track_path: Path, model_root: Path, object_id: int,
           frame_start: int = 0, frame_end: int | None = None,
           pressure_rad: float = 0.0, mass_kg: float = 0.2,
           time_scale: float = 2.0, pad_friction: float = 1.0,
           phase_plan_path: Path | None = None,
           grasp_dwell_s: float = 0.0) -> dict:
    if time_scale <= 0:
        raise ValueError("command time scale must be positive")
    if not 0 <= grasp_dwell_s <= 2.0:
        raise ValueError("grasp dwell must be in [0, 2] seconds")
    if not 0.5 <= pad_friction <= 3.0:
        raise ValueError("pad friction sensitivity must be in [0.5, 3.0]")
    model, geometry = _scene(plan_path, track_path, model_root, object_id, mass_kg)
    for name in ("left_distal_phalanx", "right_distal_phalanx"):
        geom_id = int(model.body_geomadr[model.body(name).id])
        model.geom_friction[geom_id, 0] = pad_friction
    with h5py.File(plan_path) as plan:
        joints = np.asarray(plan["q"][frame_start:frame_end], dtype=float)
        grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
        all_timestamps = np.asarray(plan["timestamps"], dtype=float)
        drive = pressure_commands(
            np.asarray(plan["gripper_joint_position"], dtype=float), grasp, pressure_rad,
        )[frame_start:frame_end]
        timestamps = all_timestamps[frame_start:frame_end]
    scoring_grasp = grasp
    if phase_plan_path is not None:
        with h5py.File(phase_plan_path) as phase_plan:
            phase_timestamps = np.asarray(phase_plan["timestamps"], dtype=float)
            phase_grasp = json.loads(phase_plan["metrics_json"][()].decode()).get("object_grasp")
        if not np.array_equal(all_timestamps, phase_timestamps) or phase_grasp is None:
            raise ValueError("scoring phase plan must describe the same frames and an object grasp")
        scoring_grasp = phase_grasp
    if len(joints) < 2 or len(drive) != len(joints):
        raise ValueError("plan slice is empty or unaligned")
    command_times = time_scale * (timestamps - timestamps[0]) / 1e6 + 0.3
    if np.any(np.diff(command_times) <= 0):
        raise ValueError("plan timestamps are not increasing")
    command_times = np.concatenate(([0.0, 0.3], command_times[1:]))
    command_joints = np.concatenate((joints[:1], joints))
    command_drive = np.concatenate((drive[:1], drive))
    if grasp_dwell_s:
        if grasp is None:
            raise ValueError("grasp dwell requires an object-aware plan")
        close_frame = int(grasp["close_complete"])
        if not frame_start <= close_frame < frame_start + len(joints) - 1:
            raise ValueError("grasp close frame is outside the replay window")
        command_index = 1 + close_frame - frame_start
        command_times = np.insert(
            command_times, command_index + 1,
            command_times[command_index] + grasp_dwell_s,
        )
        command_times[command_index + 2:] += grasp_dwell_s
        command_joints = np.insert(
            command_joints, command_index + 1, command_joints[command_index], axis=0,
        )
        command_drive = np.insert(
            command_drive, command_index + 1, command_drive[command_index],
        )
    carry_time = release_time = None
    if scoring_grasp is not None:
        carry_frame, release_frame = scoring_grasp["phase_bounds"][1:3]
        carry_time = time_scale * (all_timestamps[carry_frame] - timestamps[0]) / 1e6 + 0.3
        release_time = time_scale * (all_timestamps[release_frame] - timestamps[0]) / 1e6 + 0.3
        if grasp_dwell_s:
            carry_time += grasp_dwell_s
            release_time += grasp_dwell_s
    data = mujoco.MjData(model)
    arm_indices = [int(model.jnt_qposadr[model.joint(name).id]) for name in ARM_JOINTS]
    drive_index = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_indices] = joints[0]
    for name, multiplier in MIMIC_MULTIPLIERS.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = drive[0] * multiplier
    data.ctrl[:6] = joints[0]
    data.ctrl[6] = drive[0]
    mujoco.mj_forward(model, data)
    object_body = model.body("tracked_object").id
    object_geom = model.geom("tracked_object_geom").id
    table_geom = model.geom("table").id
    wrist_body = model.body("wrist_3_link").id
    pad_geoms_by_side = {
        name: int(model.body_geomadr[model.body(name).id])
        for name in ("left_distal_phalanx", "right_distal_phalanx")
    }
    pad_geoms = set(pad_geoms_by_side.values())
    initial_z = float(data.xpos[object_body, 2])
    peak_z = initial_z
    carry_lifted_steps = 0
    bilateral_carry_lifted_steps = 0
    robot_object_contacts = 0
    pad_contact_steps = {name: 0 for name in pad_geoms_by_side}
    bilateral_contact_steps = 0
    object_contact_bodies = {}
    peak_penetration = 0.0
    robot_table_contact_steps = 0
    peak_robot_table_penetration = 0.0
    grasp_relative_origin = None
    maximum_airborne_slip = 0.0
    maximum_airborne_contact_gap_s = 0.0
    airborne_contact_gap_s = 0.0
    maximum_joint_error = 0.0
    maximum_drive_error = 0.0
    invalid_reason = None
    invalid_contacts = []
    sampled_trace = []
    for _ in range(int(np.ceil((command_times[-1] + 0.5) / model.opt.timestep))):
        now = float(data.time)
        for joint_index in range(len(ARM_JOINTS)):
            data.ctrl[joint_index] = np.interp(now, command_times, command_joints[:, joint_index])
        data.ctrl[6] = np.interp(now, command_times, command_drive)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("physics diverged")
        maximum_joint_error = max(maximum_joint_error, float(np.max(np.abs(data.qpos[arm_indices] - data.ctrl[:6]))))
        maximum_drive_error = max(maximum_drive_error, float(abs(data.qpos[drive_index] - data.ctrl[6])))
        if maximum_joint_error > 0.5 or maximum_drive_error > 0.5:
            invalid_reason = "actuator tracking diverged"
            invalid_contacts = [
                [model.body(model.geom_bodyid[contact.geom1]).name,
                 model.body(model.geom_bodyid[contact.geom2]).name,
                 float(contact.dist)]
                for contact in data.contact[:data.ncon]
            ]
            break
        touching_pads = set()
        for contact in data.contact[:data.ncon]:
            if table_geom in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == table_geom else contact.geom1
                if other != object_geom:
                    robot_table_contact_steps += 1
                    peak_robot_table_penetration = max(
                        peak_robot_table_penetration, max(0.0, -float(contact.dist)),
                    )
            if object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
            if other in pad_geoms:
                touching_pads.add(other)
            if other != model.geom("table").id:
                robot_object_contacts += 1
                body_name = model.body(model.geom_bodyid[other]).name
                object_contact_bodies[body_name] = object_contact_bodies.get(body_name, 0) + 1
                peak_penetration = max(peak_penetration, max(0.0, -float(contact.dist)))
        for name, geom_id in pad_geoms_by_side.items():
            pad_contact_steps[name] += geom_id in touching_pads
        bilateral_contact_steps += pad_geoms.issubset(touching_pads)
        current_z = float(data.xpos[object_body, 2])
        peak_z = max(peak_z, current_z)
        in_carry = (carry_time is not None and carry_time <= data.time < release_time)
        if in_carry and current_z > initial_z + 0.025:
            carry_lifted_steps += 1
            bilateral_carry_lifted_steps += pad_geoms.issubset(touching_pads)
            pad_points = [
                data.xpos[model.body(name).id]
                + data.xmat[model.body(name).id].reshape(3, 3) @ PAD_OFFSET_M
                for name in ("left_distal_phalanx", "right_distal_phalanx")
            ]
            midpoint = (pad_points[0] + pad_points[1]) / 2
            wrist_rotation = data.xmat[wrist_body].reshape(3, 3)
            relative = wrist_rotation.T @ (data.xpos[object_body] - midpoint)
            if pad_geoms.issubset(touching_pads):
                if grasp_relative_origin is None:
                    grasp_relative_origin = relative.copy()
                airborne_contact_gap_s = 0.0
            else:
                airborne_contact_gap_s += model.opt.timestep
                maximum_airborne_contact_gap_s = max(
                    maximum_airborne_contact_gap_s, airborne_contact_gap_s,
                )
            if grasp_relative_origin is not None:
                maximum_airborne_slip = max(
                    maximum_airborne_slip,
                    float(np.linalg.norm(relative - grasp_relative_origin)),
                )
        else:
            airborne_contact_gap_s = 0.0
        if len(sampled_trace) == 0 or data.time >= sampled_trace[-1]["time_s"] + 0.1:
            pad_points = [
                data.xpos[model.body(name).id]
                + data.xmat[model.body(name).id].reshape(3, 3) @ PAD_OFFSET_M
                for name in ("left_distal_phalanx", "right_distal_phalanx")
            ]
            sampled_trace.append({
                "time_s": float(data.time),
                "object_position_m": data.xpos[object_body].tolist(),
                "object_rotation_quat_wxyz": Rotation.from_matrix(
                    data.xmat[object_body].reshape(3, 3)
                ).as_quat(scalar_first=True).tolist(),
                "pad_midpoint_m": (0.5 * (pad_points[0] + pad_points[1])).tolist(),
                "pad_gap_m": float(np.linalg.norm(pad_points[1] - pad_points[0])),
                "pad_contacts": len(touching_pads),
            })
    final_position = data.xpos[object_body].copy()
    final_rotation = data.xmat[object_body].reshape(3, 3).copy()
    bilateral_fraction = (
        bilateral_carry_lifted_steps / carry_lifted_steps
        if carry_lifted_steps else 0.0
    )
    result = {
        **geometry,
        "plan": str(plan_path), "frames": len(joints),
        "phase_plan": str(phase_plan_path) if phase_plan_path is not None else None,
        "physics_time_s": float(data.time),
        "peak_lift_m": peak_z - initial_z,
        "final_object_position_m": final_position.tolist(),
        "final_translation_error_m": float(np.linalg.norm(
            final_position - np.asarray(geometry["reference_final_position_m"]))),
        "final_orientation_error_deg": float(np.rad2deg((
            Rotation.from_matrix(final_rotation).inv()
            * Rotation.from_matrix(geometry["reference_final_rotation"])
        ).magnitude())),
        "bilateral_pad_contact_fraction_while_lifted_carry": bilateral_fraction,
        "max_carry_grasp_slip_m": maximum_airborne_slip,
        "max_carry_contact_gap_s": maximum_airborne_contact_gap_s,
        "carry_time_s": carry_time,
        "release_time_s": release_time,
        "robot_object_contact_steps": robot_object_contacts,
        "pad_contact_steps": pad_contact_steps,
        "bilateral_pad_contact_steps": bilateral_contact_steps,
        "object_contact_bodies": object_contact_bodies,
        "peak_robot_object_penetration_m": peak_penetration,
        "robot_table_contact_steps": robot_table_contact_steps,
        "peak_robot_table_penetration_m": peak_robot_table_penetration,
        "maximum_joint_tracking_error_rad": maximum_joint_error,
        "maximum_gripper_tracking_error_rad": maximum_drive_error,
        "invalid_reason": invalid_reason,
        "invalid_contacts": invalid_contacts,
        "sampled_trace": sampled_trace,
        "command_time_scale": time_scale,
        "grasp_pressure_offset_rad": pressure_rad,
        "grasp_dwell_s": grasp_dwell_s,
        "physics_timestep_s": model.opt.timestep,
        "pad_friction_coefficient": pad_friction,
    }
    result["grasp_success_proxy"] = bool(
        result["peak_lift_m"] > 0.025
        and bilateral_fraction > 0.8
        and grasp_relative_origin is not None
        and maximum_airborne_slip < 0.01
        and maximum_airborne_contact_gap_s < 0.05
        and peak_penetration < 0.004
        and peak_robot_table_penetration < 0.004
        and maximum_joint_error < 0.1
        and maximum_drive_error < 0.25
        and invalid_reason is None
    )
    result["transfer_success_proxy"] = bool(
        result["grasp_success_proxy"]
        and result["final_translation_error_m"] < 0.05
        and result["final_orientation_error_deg"] < 30.0
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--pressure-rad", type=float, default=0.0)
    parser.add_argument("--mass-kg", type=float, default=0.2)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--pad-friction", type=float, default=1.0)
    parser.add_argument("--phase-plan", type=Path, default=None)
    parser.add_argument("--grasp-dwell-s", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(replay(args.plan, args.track, args.models,
                            args.object_id, args.frame_start, args.frame_end,
                            args.pressure_rad, args.mass_kg, args.time_scale,
                            args.pad_friction, args.phase_plan,
                            args.grasp_dwell_s), indent=2))
