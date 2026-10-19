"""One continuous four-object HO-Cap replay of object-conditioned IK segments."""

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

from experiments.contact_attachment import ContactAttachment
from experiments.hocap_grasp_replay import _scene
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS, PAD_OFFSET_M
from viki.benchmark.hocap_provenance import require_hand_fit_clean, require_hand_fit_plan
from viki.retarget.robots import normalize_robot


WINDOWS = ((2, 0, 190), (3, 185, 420), (4, 420, 600), (1, 635, 831))
DEFAULT_PLANS = {
    2: Path("data/_retarget_diagnostics/hocap_g10_full_object_2/plan.h5"),
    3: Path("data/_retarget_diagnostics/hocap_g10_full_object_3/plan.h5"),
    4: Path("data/_retarget_diagnostics/hocap_g10_4_object_pilot/plan.h5"),
    1: Path("data/_retarget_diagnostics/hocap_g10_full_object_1/plan.h5"),
}


def assemble_object_trajectory(
    timestamps: np.ndarray, plan_paths: dict[int, Path],
    windows: tuple[tuple[int, int, int], ...] = WINDOWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    frame_count = len(timestamps)
    if frame_count < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("expected a complete, ordered HO-Cap episode")
    with h5py.File(plan_paths[windows[0][0]]) as first_plan:
        joint_count = first_plan["q"].shape[1]
    joints = np.full((frame_count, joint_count), np.nan)
    drive = np.full(frame_count, np.nan)
    active_object = np.zeros(frame_count, dtype=np.int16)
    segments = []
    for identifier, first, last in windows:
        path = plan_paths[identifier]
        require_hand_fit_plan(path)
        with h5py.File(path) as plan:
            local_joints = np.asarray(plan["q"], dtype=float)
            local_drive = np.asarray(plan["gripper_joint_position"], dtype=float)
            local_timestamps = np.asarray(plan["timestamps"])
            grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
        if (grasp is None or grasp.get("object_id") != identifier
                or local_joints.shape != (last - first, joint_count)
                or local_drive.shape != (last - first,)
                or not np.array_equal(local_timestamps, timestamps[first:last])
                or not np.isfinite(local_joints).all()
                or not np.isfinite(local_drive).all()):
            raise ValueError(f"object {identifier} lacks an aligned object-aware plan")
        start = first + int(grasp["prealign_start"])
        stop = first + min(last - first, int(grasp["phase_bounds"][3]) + 16)
        if start >= stop or np.isfinite(joints[start:stop]).any():
            raise ValueError("object-aware phase windows overlap or are empty")
        joints[start:stop] = local_joints[start - first:stop - first]
        drive[start:stop] = local_drive[start - first:stop - first]
        close = first + int(grasp["phase_bounds"][0])
        opened = first + int(grasp["phase_bounds"][3])
        active_object[close:opened + 1] = identifier
        segments.append({
            "object_id": identifier, "input_plan": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_window": [first, last], "active_window": [start, stop],
            "grasp_window": [close, opened + 1],
            "object_grasp": grasp,
        })
    valid = np.flatnonzero(np.isfinite(joints[:, 0]))
    if len(valid) < 2:
        raise ValueError("no object-aware trajectory segments")
    for index in range(joint_count):
        joints[:, index] = np.interp(np.arange(frame_count), valid, joints[valid, index])
    drive[:valid[0]] = drive[valid[0]]
    drive[valid[-1] + 1:] = drive[valid[-1]]
    for left, right in zip(valid[:-1], valid[1:]):
        if right - left > 1:
            drive[left + 1:right] = np.linspace(drive[left], drive[right], right - left + 1)[1:-1]
    if not np.isfinite(drive).all():
        raise RuntimeError("incomplete gripper transition")
    return joints, drive, active_object, segments


def replay(
    episode: Path, output_path: Path, trajectory_path: Path,
    video_path: Path | None = None, archive_path: Path = Path("data/_hocap_assets/models.tar"),
    time_scale: float = 2.0, release_opening_rad: float = 0.03,
    plan_paths: dict[int, Path] | None = None,
    attach_on_any_pad: bool = False,
    windows: tuple[tuple[int, int, int], ...] = WINDOWS,
) -> dict:
    if time_scale <= 0:
        raise ValueError("time scale must be positive")
    require_hand_fit_clean(episode)
    plan_paths = dict(DEFAULT_PLANS if plan_paths is None else plan_paths)
    track_path = episode / "intermediates/segmentation/hocap-oracle-v1/object_models.npz"
    gt_path = episode / "gt/object_poses_world.npz"
    with np.load(track_path, allow_pickle=False) as track:
        timestamps = np.asarray(np.load(episode / "cln.npz")["timestamps"])
        identifiers = [int(value) for value in track["object_ids"]]
        final_positions = np.asarray(track["translation_world"][-1], dtype=float)
        final_rotations = np.asarray(track["rotation_world_object"][-1], dtype=float)
        if (len(track["translation_world"]) != len(timestamps)
                or set(identifiers) != set(plan_paths)):
            raise ValueError("expected aligned plans for every tracked object")
    with np.load(gt_path, allow_pickle=False) as ground_truth:
        names = [str(name) for name in ground_truth["object_ids"]]
    object_names = {identifier: next(
        (name for name in names if name.endswith(f"_{identifier}")), None,
    ) for identifier in identifiers}
    if any(name is None for name in object_names.values()):
        raise ValueError("official HO-Cap object names are incomplete")
    joints, drive, active_object, segments = assemble_object_trajectory(
        timestamps, plan_paths, windows,
    )
    with h5py.File(plan_paths[windows[0][0]]) as first_plan:
        robot_key = first_plan["robot_key"][()].decode()
    robot_config = normalize_robot(robot_key)
    joint_count = len(robot_config.joint_names)
    if joints.shape[1] != joint_count:
        raise ValueError("plan joints do not match the selected robot")
    release_frames = {
        segment["object_id"]: segment["source_window"][0]
        + int(segment["object_grasp"]["phase_bounds"][2])
        for segment in segments
    }
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_path, q=joints, gripper_joint_position=drive,
                        timestamps=timestamps, active_object_id=active_object,
                        segments_json=json.dumps(segments))
    with tempfile.TemporaryDirectory() as directory:
        mesh_paths = {}
        mesh_hashes = {}
        with tarfile.open(archive_path) as archive:
            for identifier in identifiers:
                name = f"{object_names[identifier]}.cleaned_mesh_2000.obj"
                member = archive.extractfile(name)
                if member is None:
                    raise ValueError(f"missing official mesh {name}")
                content = member.read()
                mesh_hashes[name] = hashlib.sha256(content).hexdigest()
                mesh_paths[identifier] = Path(directory) / name
                mesh_paths[identifier].write_bytes(content)
        selected_id = identifiers[0]
        model, geometry = _scene(
            plan_paths[selected_id], track_path, Path("models"), selected_id, 0.2,
            rubber_pads=True, all_objects=True, mesh_path=mesh_paths[selected_id],
            gt_pose_path=gt_path, gt_frame=0, object_name=object_names[selected_id],
            other_mesh_paths={key: value for key, value in mesh_paths.items() if key != selected_id},
            weld_all_objects=True,
        )
    data = mujoco.MjData(model)
    arm_indices = [int(model.jnt_qposadr[model.joint(name).id])
                   for name in robot_config.joint_names]
    finger_index = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_indices] = joints[0]
    for name, multiplier in MIMIC_MULTIPLIERS.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = drive[0] * multiplier
    data.ctrl[:joint_count] = joints[0]
    data.ctrl[joint_count] = drive[0]
    mujoco.mj_forward(model, data)
    body_ids = {identifier: model.body(
        "tracked_object" if identifier == selected_id else f"tracked_object_{identifier}",
    ).id for identifier in identifiers}
    geom_ids = {identifier: model.geom(
        "tracked_object_geom" if identifier == selected_id else f"tracked_object_{identifier}_geom",
    ).id for identifier in identifiers}
    if any(model.geom_type[value] != mujoco.mjtGeom.mjGEOM_MESH for value in geom_ids.values()):
        raise RuntimeError("all four collision geoms must use official meshes")
    geom_to_object = {value: key for key, value in geom_ids.items()}
    pad_ids = {side: model.geom(f"{side}_rubber_pad").id for side in ("left", "right")}
    pad_bodies = [model.body(f"{side}_distal_phalanx").id for side in ("left", "right")]
    table_id = model.geom("table").id
    attachments = {
        identifier: ContactAttachment(
            model, data,
            "tracked_object_grasp" if identifier == selected_id else f"tracked_object_{identifier}_grasp",
            robot_config.ee_frame,
            "tracked_object" if identifier == selected_id else f"tracked_object_{identifier}",
            hold_s=model.opt.timestep if attach_on_any_pad else 0.08,
            release_opening_rad=release_opening_rad,
            attach_on_any_pad=attach_on_any_pad,
        ) for identifier in identifiers
    }
    initial_positions = {identifier: data.xpos[body].copy() for identifier, body in body_ids.items()}
    peak_lift = {identifier: 0.0 for identifier in identifiers}
    bilateral_steps = {identifier: 0 for identifier in identifiers}
    unintended_peak_force = {identifier: 0.0 for identifier in identifiers}
    object_object_contacts = {}
    robot_table_contacts = 0
    max_joint_error = 0.0
    frame_cursor = 0
    trace = []
    command_times = time_scale * (timestamps - timestamps[0]) / 1e6 + 0.3
    command_times = np.concatenate(([0.0, 0.3], command_times[1:]))
    command_joints = np.concatenate((joints[:1], joints))
    command_drive = np.concatenate((drive[:1], drive))
    duration = float(command_times[-1] + 0.5)
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
                raise RuntimeError("could not create diagnostic video")
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
            for index in range(joint_count):
                data.ctrl[index] = np.interp(now, command_times, command_joints[:, index])
            data.ctrl[joint_count] = np.interp(now, command_times, command_drive)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"physics diverged at {data.time:.3f} s")
            max_joint_error = max(max_joint_error, float(np.max(
                np.abs(data.qpos[arm_indices] - data.ctrl[:joint_count]),
            )))
            frame_index = min(max(np.searchsorted(command_times, data.time, side="right") - 1, 0),
                              len(active_object) - 1)
            active_id = int(active_object[frame_index])
            forces = {identifier: {"left": 0.0, "right": 0.0} for identifier in identifiers}
            non_pad = {identifier: False for identifier in identifiers}
            for contact_index, contact in enumerate(data.contact[:data.ncon]):
                first_id = geom_to_object.get(contact.geom1)
                second_id = geom_to_object.get(contact.geom2)
                if first_id is not None and second_id is not None:
                    pair = "-".join(map(str, sorted((first_id, second_id))))
                    contact_force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, contact_force)
                    event = object_object_contacts.setdefault(
                        pair, {"first_time_s": float(data.time),
                               "peak_normal_force_n": 0.0, "samples": 0},
                    )
                    event["samples"] += 1
                    event["peak_normal_force_n"] = max(
                        event["peak_normal_force_n"], max(0.0, float(contact_force[0])),
                    )
                for identifier, opposite in ((first_id, contact.geom2),
                                             (second_id, contact.geom1)):
                    if identifier is None:
                        continue
                    side = next((name for name, geom_id in pad_ids.items()
                                 if opposite == geom_id), None)
                    if side is None and opposite == table_id:
                        continue
                    contact_force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, contact_force)
                    normal = max(0.0, float(contact_force[0]))
                    if side is not None:
                        forces[identifier][side] += normal
                    elif opposite not in geom_to_object and normal >= 0.2:
                        non_pad[identifier] = True
                    if active_id != identifier and opposite not in (*geom_ids.values(), table_id):
                        unintended_peak_force[identifier] = max(
                            unintended_peak_force[identifier], normal,
                        )
                if table_id in (contact.geom1, contact.geom2):
                    opposite = contact.geom2 if contact.geom1 == table_id else contact.geom1
                    robot_table_contacts += opposite not in geom_to_object
            pad_positions = [
                data.xpos[body] + data.xmat[body].reshape(3, 3) @ PAD_OFFSET_M
                for body in pad_bodies
            ]
            for identifier, attachment in attachments.items():
                bilateral_steps[identifier] += min(forces[identifier].values()) >= 0.2
                peak_lift[identifier] = max(
                    peak_lift[identifier],
                    float(data.xpos[body_ids[identifier], 2] - initial_positions[identifier][2]),
                )
                if attach_on_any_pad and frame_index >= release_frames[identifier]:
                    attachment.release()
                    continue
                if identifier == active_id or attachment.attached:
                    attachment.observe(
                        forces[identifier], pad_positions, non_pad[identifier],
                        float(data.ctrl[joint_count]), float(data.qpos[finger_index]),
                        model.opt.timestep,
                    )
            if frame_cursor < len(joints) and data.time >= command_times[frame_cursor]:
                trace.append({
                    "frame": frame_cursor, "time_s": float(data.time),
                    "active_object_id": active_id,
                    "pad_positions_m": [position.tolist() for position in pad_positions],
                    "finger_joint_rad": float(data.qpos[finger_index]),
                    "positions_m": {str(identifier): data.xpos[body].tolist()
                                    for identifier, body in body_ids.items()},
                    "attached": {str(identifier): attachment.attached
                                 for identifier, attachment in attachments.items()},
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
        body = body_ids[identifier]
        rotation = data.xmat[body].reshape(3, 3)
        attachment = attachments[identifier]
        objects[str(identifier)] = {
            "name": object_names[identifier],
            "final_position_m": data.xpos[body].tolist(),
            "tracked_final_position_m": final_positions[index].tolist(),
            "final_position_error_m": float(np.linalg.norm(data.xpos[body] - final_positions[index])),
            "final_orientation_error_deg": float(np.rad2deg((
                Rotation.from_matrix(rotation).inv()
                * Rotation.from_matrix(final_rotations[index])
            ).magnitude())),
            "peak_lift_m": peak_lift[identifier],
            "bilateral_pad_contact_steps": bilateral_steps[identifier],
            "unintended_peak_robot_force_n": unintended_peak_force[identifier],
            "attachment_activated_at_s": attachment.attach_time_s,
            "attachment_released_at_s": attachment.release_time_s,
            "forces_at_attachment_n": attachment.attach_forces_n,
            "still_attached": attachment.attached,
        }
    result = {
        "protocol": "one continuous object-conditioned contact-assisted official-mesh replay",
        "robot": robot_key,
        "episode": str(episode), "frame_count": len(joints), "frames_reached": frame_cursor,
        "time_scale": time_scale, "physics_time_s": float(data.time),
        "trajectory": str(trajectory_path), "segments": segments,
        "mesh_archive": str(archive_path), "mesh_sha256": mesh_hashes,
        "collision_geometry": "official mesh convex hulls",
        "attach_on_any_pad": attach_on_any_pad,
        "table_z_m": geometry["table_z_m"],
        "peak_joint_tracking_error_rad": max_joint_error,
        "robot_table_contact_samples": robot_table_contacts,
        "object_object_contacts": object_object_contacts,
        "objects": objects, "trace": trace,
        "video": str(video_path) if video_path is not None else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--release-opening-rad", type=float, default=0.03)
    parser.add_argument("--attach-on-any-pad", action="store_true")
    parser.add_argument("--mesh-archive", type=Path,
                        default=Path("data/_hocap_assets/models.tar"))
    arguments = parser.parse_args()
    report = replay(arguments.episode, arguments.output, arguments.trajectory,
                    arguments.video, arguments.mesh_archive, arguments.time_scale,
                    arguments.release_opening_rad, attach_on_any_pad=arguments.attach_on_any_pad)
    print(json.dumps({"frames_reached": report["frames_reached"],
                      "objects": report["objects"]}, indent=2))
