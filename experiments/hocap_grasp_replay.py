"""One-scene HO-Cap grasp replay with optional official object meshes.

The tabletop is inferred from tracked geometry, and MuJoCo collides against
convex mesh hulls; this is not a calibrated force/contact benchmark.
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
from experiments.grasp_gate import insert_grasp_dwell
from experiments.hocap_mesh import aligned_mesh
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS, PAD_OFFSET_M
from viki.retarget.robots import normalize_robot


def _scene(plan_path: Path, track_path: Path, model_root: Path,
           object_id: int, mass_kg: float, rubber_pads: bool = False,
           all_objects: bool = False, mesh_path: Path | None = None,
           gt_pose_path: Path | None = None, gt_frame: int | None = None,
           object_name: str | None = None,
           other_mesh_paths: dict[int, Path] | None = None,
           weld_all_objects: bool = False):
    if not 0.01 <= mass_kg <= 1.0:
        raise ValueError("proxy mass must be in [0.01, 1.0] kg")
    with h5py.File(plan_path) as plan:
        base_position = np.asarray(plan["base_position_calibration"], dtype=float)
        base_rpy = np.deg2rad(np.asarray(plan["base_rpy_deg_calibration"], dtype=float))
        robot_key = plan["robot_key"][()].decode()
    robot_config = normalize_robot(robot_key)
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
        additional_objects = []
        other_surfaces = {}
        if all_objects:
            for index, tracked_id in enumerate(track["object_ids"]):
                if index == object_index:
                    continue
                other_surface = np.asarray(
                    track["model_xyz_object"][offsets[index]:offsets[index + 1]], dtype=float,
                )
                other_surfaces[int(tracked_id)] = other_surface
                other_low, other_high = np.quantile(other_surface, [0.05, 0.95], axis=0)
                other_half_size = (other_high - other_low) / 2
                if np.any(other_half_size < 0.005) or np.any(other_half_size > 0.15):
                    raise ValueError(f"object {tracked_id} has implausible proxy dimensions")
                additional_objects.append((
                    int(tracked_id), np.asarray(track["translation_world"][0, index], dtype=float),
                    np.asarray(track["rotation_world_object"][0, index], dtype=float),
                    (other_low + other_high) / 2, other_half_size,
                ))
    low, high = np.quantile(surface, [0.05, 0.95], axis=0)
    half_size = (high - low) / 2
    local_center = (low + high) / 2
    if np.any(half_size < 0.005) or np.any(half_size > 0.15):
        raise ValueError("fitted object proxy has implausible dimensions")
    mesh_options = (mesh_path, gt_pose_path, gt_frame, object_name)
    if any(value is not None for value in mesh_options) and not all(
        value is not None for value in mesh_options
    ):
        raise ValueError("official mesh requires path, pose, frame and object name")
    if other_mesh_paths and (not all_objects or gt_pose_path is None or gt_frame is None):
        raise ValueError("other official meshes require all objects and GT poses")
    if other_mesh_paths and set(other_mesh_paths) != {entry[0] for entry in additional_objects}:
        raise ValueError("one official mesh is required for each other object")
    mesh_text = None
    mesh_details = {}
    if mesh_path is not None:
        mesh_text, aligned_vertices, mesh_details = aligned_mesh(
            mesh_path, gt_pose_path, object_name, gt_frame,
            orientation, origin, surface,
        )
        world_points = aligned_vertices @ orientation.T + origin
    else:
        box_corners = np.array(np.meshgrid(*[[-value, value] for value in half_size])).reshape(3, -1).T
        world_points = (box_corners + local_center) @ orientation.T + origin
    table_z = float(np.min(world_points[:, 2]))
    other_mesh_texts = {}
    other_mesh_details = {}
    for tracked_id, other_origin, other_orientation, other_center, other_half_size in additional_objects:
        if other_mesh_paths:
            other_text, other_vertices, details = aligned_mesh(
                other_mesh_paths[tracked_id], gt_pose_path,
                f"{object_name.rsplit('_', 1)[0]}_{tracked_id}", gt_frame,
                other_orientation, other_origin, other_surfaces[tracked_id],
            )
            other_mesh_texts[tracked_id] = other_text
            other_mesh_details[tracked_id] = details
            world_other = other_vertices @ other_orientation.T + other_origin
        else:
            other_corners = np.array(np.meshgrid(
                *[[-value, value] for value in other_half_size],
            )).reshape(3, -1).T
            world_other = (other_corners + other_center) @ other_orientation.T + other_origin
        table_z = min(table_z, float(np.min(world_other[:, 2])))
    with tempfile.TemporaryDirectory() as directory:
        urdf_path = Path(directory) / f"{robot_key}_robotiq.urdf"
        urdf_path.write_text(assembled_xml(model_root, robot_key))
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
        if rubber_pads and body.get("name") in ("left_distal_phalanx", "right_distal_phalanx"):
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            ET.SubElement(
                body, "geom", name=f"{body.get('name').split('_')[0]}_rubber_pad",
                type="box", pos=" ".join(map(str, PAD_OFFSET_M)),
                size="0.011 0.003175 0.01875", mass="0",
                friction="1.0 0.005 0.0001", rgba="0.8 0.8 0.8 1",
            )
    world.append(robot)
    ET.SubElement(world, "geom", name="table", type="box",
                  pos=f"{origin[0]} {origin[1]} {table_z - 0.025}",
                  size="0.6 0.6 0.025", friction="1.2 0.005 0.0001")
    item = ET.SubElement(world, "body", name="tracked_object",
                         pos=" ".join(map(str, origin)),
                         quat=" ".join(map(str, Rotation.from_matrix(orientation).as_quat(scalar_first=True))))
    ET.SubElement(item, "freejoint", name="tracked_object_free")
    object_geom_attributes = {
        "name": "tracked_object_geom", "mass": str(mass_kg),
        "friction": "1.2 0.005 0.0001", "rgba": "0.3 0.65 0.8 1",
    }
    if mesh_text is None:
        object_geom_attributes.update(
            type="box", pos=" ".join(map(str, local_center)),
            size=" ".join(map(str, half_size)),
        )
    else:
        object_geom_attributes.update(type="mesh", mesh="tracked_object_mesh")
    ET.SubElement(item, "geom", **object_geom_attributes)
    colors = ("0.9 0.5 0.1 1", "0.1 0.8 0.35 1", "0.8 0.15 0.2 1")
    for index, (tracked_id, other_origin, other_orientation, other_center,
                other_half_size) in enumerate(additional_objects):
        other = ET.SubElement(
            world, "body", name=f"tracked_object_{tracked_id}",
            pos=" ".join(map(str, other_origin)),
            quat=" ".join(map(str, Rotation.from_matrix(other_orientation).as_quat(scalar_first=True))),
        )
        ET.SubElement(other, "freejoint", name=f"tracked_object_{tracked_id}_free")
        other_attributes = {
            "name": f"tracked_object_{tracked_id}_geom",
            "mass": str(mass_kg), "friction": "1.2 0.005 0.0001",
            "rgba": colors[index % len(colors)],
        }
        if tracked_id in other_mesh_texts:
            other_attributes.update(type="mesh", mesh=f"tracked_object_{tracked_id}_mesh")
        else:
            other_attributes.update(type="box", pos=" ".join(map(str, other_center)),
                                    size=" ".join(map(str, other_half_size)))
        ET.SubElement(other, "geom", **other_attributes)
    contacts = ET.SubElement(root, "contact")
    gripper_bodies = [body.get("name") for body in robot.iter("body")
                      if "phalanx" in body.get("name", "") or "knuckle" in body.get("name", "")]
    for first, second in combinations([robot_config.ee_frame, *gripper_bodies], 2):
        ET.SubElement(contacts, "exclude", body1=first, body2=second)
    if robot_key == "ur10":
        ET.SubElement(contacts, "exclude", body1="forearm_link", body2="wrist_2_link")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality, "weld", name="tracked_object_grasp", body1=robot_config.ee_frame,
        body2="tracked_object", active="false",
    )
    if weld_all_objects:
        for tracked_id, *_ in additional_objects:
            ET.SubElement(
                equality, "weld", name=f"tracked_object_{tracked_id}_grasp",
                body1=robot_config.ee_frame, body2=f"tracked_object_{tracked_id}",
                active="false",
            )
    actuators = ET.SubElement(root, "actuator")
    for name in robot_config.joint_names:
        ET.SubElement(actuators, "position", name=f"cmd_{name}", joint=name,
                      kp="1000", kv="100")
    ET.SubElement(actuators, "position", name="cmd_finger", joint="finger_joint",
                  kp="50", kv="5")
    with tempfile.TemporaryDirectory() as directory:
        if mesh_text is not None:
            aligned_path = Path(directory) / "tracked_object_aligned.obj"
            aligned_path.write_text(mesh_text)
            assets = root.find("asset")
            if assets is None:
                assets = ET.SubElement(root, "asset")
            ET.SubElement(
                assets, "mesh", name="tracked_object_mesh", file=str(aligned_path),
            )
        for tracked_id, other_text in other_mesh_texts.items():
            aligned_path = Path(directory) / f"tracked_object_{tracked_id}_aligned.obj"
            aligned_path.write_text(other_text)
            assets = root.find("asset")
            if assets is None:
                assets = ET.SubElement(root, "asset")
            ET.SubElement(assets, "mesh", name=f"tracked_object_{tracked_id}_mesh",
                          file=str(aligned_path))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, {
        "proxy_half_size_m": half_size.tolist(), "table_z_m": table_z,
        "proxy_mass_kg": mass_kg,
        "collision_geometry": "official_mesh_convex_hull" if mesh_text is not None else "partial_cloud_box",
        **mesh_details,
        "rubber_pad_collision_proxy": rubber_pads,
        "all_object_proxies": all_objects,
        "other_object_geometry": "official_mesh_convex_hull" if other_mesh_paths else "partial_cloud_box",
        "other_mesh_details": other_mesh_details,
        "other_object_ids": [object_id for object_id, *_ in additional_objects],
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
           grasp_dwell_s: float = 0.0, rubber_pads: bool = False,
           mesh_path: Path | None = None, gt_pose_path: Path | None = None,
           gt_frame: int | None = None, object_name: str | None = None,
           fully_close: bool = False) -> dict:
    if time_scale <= 0:
        raise ValueError("command time scale must be positive")
    if not 0 <= grasp_dwell_s <= 2.0:
        raise ValueError("grasp dwell must be in [0, 2] seconds")
    if not 0.5 <= pad_friction <= 3.0:
        raise ValueError("pad friction sensitivity must be in [0.5, 3.0]")
    model, geometry = _scene(
        plan_path, track_path, model_root, object_id, mass_kg, rubber_pads,
        mesh_path=mesh_path, gt_pose_path=gt_pose_path, gt_frame=gt_frame,
        object_name=object_name,
    )
    contact_names = (("left_rubber_pad", "right_rubber_pad") if rubber_pads
                     else ("left_distal_phalanx", "right_distal_phalanx"))
    pad_geoms_by_side = {}
    for name in contact_names:
        geom_id = (model.geom(name).id if rubber_pads
                   else int(model.body_geomadr[model.body(name).id]))
        pad_geoms_by_side[name] = geom_id
        model.geom_friction[geom_id, 0] = pad_friction
    with h5py.File(plan_path) as plan:
        joints = np.asarray(plan["q"][frame_start:frame_end], dtype=float)
        grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
        all_timestamps = np.asarray(plan["timestamps"], dtype=float)
        drive = pressure_commands(
            np.asarray(plan["gripper_joint_position"], dtype=float), grasp, pressure_rad,
            fully_close=fully_close,
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
        command_times, command_joints, command_drive, _, _ = insert_grasp_dwell(
            command_times, command_joints, command_drive,
            int(grasp["close_complete"]), frame_start, grasp_dwell_s,
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
    pad_geoms = set(pad_geoms_by_side.values())
    initial_z = float(data.xpos[object_body, 2])
    peak_z = initial_z
    carry_lifted_steps = 0
    bilateral_carry_lifted_steps = 0
    robot_object_contacts = 0
    pad_contact_steps = {name: 0 for name in pad_geoms_by_side}
    bilateral_contact_steps = 0
    object_contact_bodies = {}
    object_contact_geoms = {}
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
                geom_name = model.geom(other).name or body_name
                object_contact_geoms[geom_name] = object_contact_geoms.get(geom_name, 0) + 1
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
        "object_contact_geoms": object_contact_geoms,
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
        "fully_close": fully_close,
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
    parser.add_argument("--fully-close", action="store_true")
    parser.add_argument("--mass-kg", type=float, default=0.2)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--pad-friction", type=float, default=1.0)
    parser.add_argument("--phase-plan", type=Path, default=None)
    parser.add_argument("--grasp-dwell-s", type=float, default=0.0)
    parser.add_argument("--rubber-pads", action="store_true")
    parser.add_argument("--official-mesh", type=Path)
    parser.add_argument("--gt-object-poses", type=Path)
    parser.add_argument("--gt-anchor-frame", type=int)
    parser.add_argument("--object-name")
    args = parser.parse_args()
    print(json.dumps(replay(args.plan, args.track, args.models,
                            args.object_id, args.frame_start, args.frame_end,
                            args.pressure_rad, args.mass_kg, args.time_scale,
                            args.pad_friction, args.phase_plan,
                            args.grasp_dwell_s, args.rubber_pads,
                            args.official_mesh, args.gt_object_poses,
                            args.gt_anchor_frame, args.object_name,
                            args.fully_close), indent=2))
