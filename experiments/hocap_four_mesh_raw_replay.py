"""Descriptive full-episode replay with four free official HO-Cap meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.hocap_grasp_replay import _scene
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS
from viki.benchmark.hocap_provenance import require_hand_fit_clean, require_hand_fit_plan


def replay(episode: Path, archive_path: Path, output_path: Path,
           video_path: Path | None = None, time_scale: float = 2.0,
           static_robot: bool = False, allow_legacy_plan: bool = False) -> dict:
    if time_scale <= 0:
        raise ValueError("time scale must be positive")
    plan_path = episode / "plan.h5"
    if not allow_legacy_plan:
        require_hand_fit_clean(episode)
        require_hand_fit_plan(plan_path)
    track_path = episode / "intermediates/segmentation/hocap-oracle-v1/object_models.npz"
    gt_path = episode / "gt/object_poses_world.npz"
    with h5py.File(plan_path) as plan:
        effective_pose = plan["source_pose"][()] if "source_pose" in plan else "unrecorded"
        if isinstance(effective_pose, bytes):
            effective_pose = effective_pose.decode()
        joints = np.asarray(plan["q"], dtype=float)
        drive = np.asarray(plan["gripper_joint_position"], dtype=float)
        timestamps = np.asarray(plan["timestamps"], dtype=float)
    with np.load(track_path, allow_pickle=False) as track:
        identifiers = [int(value) for value in track["object_ids"]]
        track_frames = len(track["translation_world"])
        final_positions = np.asarray(track["translation_world"][-1], dtype=float)
        final_rotations = np.asarray(track["rotation_world_object"][-1], dtype=float)
    with np.load(gt_path, allow_pickle=False) as poses:
        names = [str(value) for value in poses["object_ids"]]
    if (len(joints) != len(drive) or len(joints) != len(timestamps)
            or len(joints) < 2 or len(joints) != track_frames
            or identifiers != [1, 2, 3, 4]
            or names != [f"G10_{identifier}" for identifier in identifiers]
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("expected aligned four-object 831-frame HO-Cap episode")
    with tempfile.TemporaryDirectory() as directory:
        mesh_paths = {}
        mesh_hashes = {}
        with tarfile.open(archive_path) as archive:
            for identifier, name in zip(identifiers, names):
                member_name = f"{name}.cleaned_mesh_2000.obj"
                member = archive.extractfile(member_name)
                if member is None:
                    raise ValueError(f"missing official mesh: {member_name}")
                content = member.read()
                mesh_hashes[name] = hashlib.sha256(content).hexdigest()
                mesh_paths[identifier] = Path(directory) / member_name
                mesh_paths[identifier].write_bytes(content)
        model, geometry = _scene(
            plan_path, track_path, Path("models"), 4, 0.2,
            rubber_pads=True, all_objects=True, mesh_path=mesh_paths[4],
            gt_pose_path=gt_path, gt_frame=0, object_name=names[3],
            other_mesh_paths={key: value for key, value in mesh_paths.items() if key != 4},
        )

    data = mujoco.MjData(model)
    arm_indices = [int(model.jnt_qposadr[model.joint(name).id]) for name in ARM_JOINTS]
    finger_index = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_indices] = joints[0]
    for name, multiplier in MIMIC_MULTIPLIERS.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = drive[0] * multiplier
    data.ctrl[:6] = joints[0]
    data.ctrl[6] = drive[0]
    mujoco.mj_forward(model, data)

    body_ids = {identifier: model.body(
        "tracked_object" if identifier == 4 else f"tracked_object_{identifier}",
    ).id for identifier in identifiers}
    geom_ids = {identifier: model.geom(
        "tracked_object_geom" if identifier == 4 else f"tracked_object_{identifier}_geom",
    ).id for identifier in identifiers}
    geom_to_object = {geom_id: identifier for identifier, geom_id in geom_ids.items()}
    pad_geoms = {model.geom("left_rubber_pad").id, model.geom("right_rubber_pad").id}
    initial_positions = {identifier: data.xpos[body_id].copy()
                         for identifier, body_id in body_ids.items()}
    peak_excursion = {identifier: 0.0 for identifier in identifiers}
    peak_height = {identifier: 0.0 for identifier in identifiers}
    bilateral_steps = {identifier: 0 for identifier in identifiers}
    robot_object_steps = {identifier: 0 for identifier in identifiers}
    peak_robot_object_force = {identifier: 0.0 for identifier in identifiers}
    first_robot_object_contact_time = {identifier: None for identifier in identifiers}
    robot_table_steps = 0
    object_object_steps = 0
    peak_joint_error = 0.0
    trace = []
    command_times = time_scale * (timestamps - timestamps[0]) / 1e6 + 0.3
    command_times = np.concatenate(([0.0, 0.3], command_times[1:]))
    command_joints = np.concatenate((joints[:1], joints))
    command_drive = np.concatenate((drive[:1], drive))
    if static_robot:
        command_joints[:] = joints[0]
        command_drive[:] = drive[0]
    duration = float(command_times[-1] + 0.5)
    frame_cursor = 0
    renderer = writer = None
    next_video_time = 0.0
    try:
        if video_path is not None:
            import cv2
            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=480, width=640)
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (1280, 480))
            if not writer.isOpened():
                raise RuntimeError("could not create video")
            center = np.mean(data.xpos[list(body_ids.values())], axis=0)
            cameras = []
            for azimuth in (130.0, 0.0):
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = center
                camera.distance = 1.25
                camera.azimuth = azimuth
                camera.elevation = -35.0
                cameras.append(camera)
        while data.time < duration:
            now = float(data.time)
            for index in range(6):
                data.ctrl[index] = np.interp(now, command_times, command_joints[:, index])
            data.ctrl[6] = np.interp(now, command_times, command_drive)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"physics diverged at {data.time:.3f} s")
            peak_joint_error = max(peak_joint_error, float(np.max(
                np.abs(data.qpos[arm_indices] - data.ctrl[:6]),
            )))
            contact_pads = {identifier: set() for identifier in identifiers}
            for contact_index, contact in enumerate(data.contact[:data.ncon]):
                first_object = geom_to_object.get(contact.geom1)
                second_object = geom_to_object.get(contact.geom2)
                if first_object is not None and second_object is not None:
                    object_object_steps += 1
                for identifier, opposite in ((first_object, contact.geom2),
                                             (second_object, contact.geom1)):
                    if identifier is None:
                        continue
                    if opposite in pad_geoms:
                        contact_pads[identifier].add(opposite)
                    elif opposite not in (*geom_ids.values(), model.geom("table").id):
                        robot_object_steps[identifier] += 1
                        if first_robot_object_contact_time[identifier] is None:
                            first_robot_object_contact_time[identifier] = float(data.time)
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        peak_robot_object_force[identifier] = max(
                            peak_robot_object_force[identifier], abs(float(force[0])),
                        )
                if model.geom("table").id in (contact.geom1, contact.geom2):
                    opposite = contact.geom2 if contact.geom1 == model.geom("table").id else contact.geom1
                    robot_table_steps += opposite not in geom_to_object
            for identifier, body_id in body_ids.items():
                position = data.xpos[body_id]
                peak_excursion[identifier] = max(
                    peak_excursion[identifier], float(np.linalg.norm(
                        position - initial_positions[identifier],
                    )),
                )
                peak_height[identifier] = max(
                    peak_height[identifier],
                    float(position[2] - initial_positions[identifier][2]),
                )
                bilateral_steps[identifier] += len(contact_pads[identifier]) == 2
            if frame_cursor < len(joints) and data.time >= command_times[frame_cursor]:
                trace.append({
                    "frame": frame_cursor,
                    "time_s": float(data.time),
                    "positions_m": {str(identifier): data.xpos[body_id].tolist()
                                    for identifier, body_id in body_ids.items()},
                })
                frame_cursor += 1
            if writer is not None and data.time >= next_video_time:
                frames = []
                for camera in cameras:
                    renderer.update_scene(data, camera=camera)
                    frames.append(renderer.render().copy())
                writer.write(cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR))
                next_video_time += 0.1
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()

    objects = {}
    for index, identifier in enumerate(identifiers):
        body_id = body_ids[identifier]
        final_position = data.xpos[body_id].copy()
        final_rotation = data.xmat[body_id].reshape(3, 3)
        objects[str(identifier)] = {
            "name": names[index],
            "initial_position_m": initial_positions[identifier].tolist(),
            "final_position_m": final_position.tolist(),
            "tracked_final_position_m": final_positions[index].tolist(),
            "final_position_error_m": float(np.linalg.norm(final_position - final_positions[index])),
            "final_orientation_error_deg": float(np.rad2deg((
                Rotation.from_matrix(final_rotation).inv()
                * Rotation.from_matrix(final_rotations[index])
            ).magnitude())),
            "peak_excursion_m": peak_excursion[identifier],
            "peak_lift_m": peak_height[identifier],
            "bilateral_pad_contact_steps": bilateral_steps[identifier],
            "robot_object_contact_samples": robot_object_steps[identifier],
            "peak_robot_object_force_n": peak_robot_object_force[identifier],
            "first_robot_object_contact_time_s": first_robot_object_contact_time[identifier],
        }
    result = {
        "protocol": "raw full-episode four-official-mesh physical replay, no acceptance gates",
        "static_robot_control": static_robot,
        "effective_pose_source": effective_pose,
        "legacy_plan_override": allow_legacy_plan,
        "episode": str(episode),
        "mesh_archive": str(archive_path),
        "mesh_sha256": mesh_hashes,
        "all_object_geoms_are_meshes": all(
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            for geom_id in geom_ids.values()
        ),
        "collision_geometry": "official mesh convex hull",
        "frame_count": len(joints),
        "frames_reached": frame_cursor,
        "time_scale": time_scale,
        "physics_time_s": float(data.time),
        "table_z_m": geometry["table_z_m"],
        "peak_joint_tracking_error_rad": peak_joint_error,
        "robot_table_contact_samples": robot_table_steps,
        "object_object_contact_samples": object_object_steps,
        "objects": objects,
        "trace": trace,
        "video": str(video_path) if video_path is not None else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--mesh-archive", type=Path,
                        default=Path("data/_hocap_assets/models.tar"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--static-robot", action="store_true")
    parser.add_argument("--allow-legacy-plan", action="store_true",
                        help="explicitly replay archived landmark-only control")
    arguments = parser.parse_args()
    report = replay(arguments.episode, arguments.mesh_archive, arguments.output,
                    arguments.video, arguments.time_scale, arguments.static_robot,
                    arguments.allow_legacy_plan)
    print(json.dumps({"output": str(arguments.output),
                      "frames_reached": report["frames_reached"],
                      "objects": report["objects"]}, indent=2))
