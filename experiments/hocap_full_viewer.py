"""Play a complete HO-Cap plan with all tracked objects as free MuJoCo proxies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from experiments.gripper_control import pressure_commands
from experiments.grasp_gate import BilateralForceGate, insert_grasp_dwell
from experiments.hocap_grasp_replay import _scene
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS


def watch(plan_path: Path, track_path: Path, model_root: Path,
          object_id: int, mass_kg: float, time_scale: float,
          speed: float, frame_start: int = 0, frame_end: int | None = None,
          all_objects: bool = True, rubber_pads: bool = False,
          mesh_path: Path | None = None, gt_pose_path: Path | None = None,
          gt_frame: int | None = None, object_name: str | None = None,
          pressure_rad: float = 0.0, grasp_dwell_s: float = 0.0,
          minimum_pad_force_n: float | None = None,
          force_hold_s: float = 0.1, fully_close: bool = False) -> dict:
    if speed <= 0 or time_scale <= 0:
        raise ValueError("playback speed and command time scale must be positive")
    if minimum_pad_force_n is not None and (not rubber_pads or grasp_dwell_s < force_hold_s):
        raise ValueError("force gate requires rubber pads and dwell at least as long as force hold")
    model, geometry = _scene(
        plan_path, track_path, model_root, object_id, mass_kg,
        rubber_pads=rubber_pads, all_objects=all_objects, mesh_path=mesh_path,
        gt_pose_path=gt_pose_path, gt_frame=gt_frame, object_name=object_name,
    )
    with h5py.File(plan_path) as plan:
        joints = np.asarray(plan["q"][frame_start:frame_end], dtype=float)
        timestamps = np.asarray(plan["timestamps"][frame_start:frame_end], dtype=float)
        grasp = json.loads(plan["metrics_json"][()].decode()).get("object_grasp")
        drive = pressure_commands(
            np.asarray(plan["gripper_joint_position"], dtype=float), grasp, pressure_rad,
            fully_close=fully_close,
        )[frame_start:frame_end]
    if (len(joints) != len(drive) or len(joints) != len(timestamps)
            or len(joints) < 2 or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("full-scene plan arrays must be aligned and time ordered")
    command_times = time_scale * (timestamps - timestamps[0]) / 1e6 + 0.3
    command_times = np.concatenate(([0.0, 0.3], command_times[1:]))
    command_joints = np.concatenate((joints[:1], joints))
    command_drive = np.concatenate((drive[:1], drive))
    dwell_start = dwell_end = None
    if grasp_dwell_s:
        if grasp is None:
            raise ValueError("grasp dwell requires an object-aware plan")
        command_times, command_joints, command_drive, dwell_start, dwell_end = insert_grasp_dwell(
            command_times, command_joints, command_drive,
            int(grasp["phase_bounds"][1]) - 1, frame_start, grasp_dwell_s,
            hold_drive=float(drive[int(grasp["phase_bounds"][1]) - frame_start]),
        )
    duration = float(command_times[-1] + 0.5)

    force_gate = (BilateralForceGate(minimum_pad_force_n, force_hold_s)
                  if minimum_pad_force_n is not None else None)
    pad_geoms = ({model.geom("left_rubber_pad").id: "left",
                  model.geom("right_rubber_pad").id: "right"}
                 if force_gate is not None else {})
    object_geom = model.geom("tracked_object_geom").id if force_gate is not None else None
    peak_pad_force_n = {"left": 0.0, "right": 0.0}
    peak_simultaneous_pad_force_n = 0.0
    force_gate_passed = False
    aborted_before_lift = False

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

    object_names = ["tracked_object", *(f"tracked_object_{identifier}"
                                        for identifier in geometry["other_object_ids"])]
    object_body_ids = [model.body(name).id for name in object_names]
    initial_positions = {name: data.xpos[body_id].tolist()
                         for name, body_id in zip(object_names, object_body_ids)}
    base_position = model.body("robot_base").pos
    center = np.mean(data.xpos[object_body_ids], axis=0)
    lookat = (center + base_position) / 2
    lookat[2] = 0.15
    peak_joint_error = 0.0
    completed = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -30
        viewer.sync()
        print(f"MuJoCo GUI ready: {len(joints)} frames, {len(object_names)} free objects; starts in 2 s.", flush=True)
        time.sleep(2.0)
        start_wall = time.monotonic()
        step_index = 0
        while viewer.is_running() and data.time < duration:
            now = float(data.time)
            if force_gate is not None and now >= dwell_end and not force_gate_passed:
                aborted_before_lift = True
                break
            for joint_index in range(len(ARM_JOINTS)):
                data.ctrl[joint_index] = np.interp(
                    now, command_times, command_joints[:, joint_index],
                )
            data.ctrl[6] = np.interp(now, command_times, command_drive)
            mujoco.mj_step(model, data)
            if force_gate is not None and dwell_start <= data.time <= dwell_end:
                pad_force_n = {"left": 0.0, "right": 0.0}
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if contact.geom1 == object_geom:
                        side = pad_geoms.get(contact.geom2)
                    elif contact.geom2 == object_geom:
                        side = pad_geoms.get(contact.geom1)
                    else:
                        side = None
                    if side is not None:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        pad_force_n[side] += max(0.0, float(force[0]))
                for side in pad_force_n:
                    peak_pad_force_n[side] = max(peak_pad_force_n[side], pad_force_n[side])
                peak_simultaneous_pad_force_n = max(
                    peak_simultaneous_pad_force_n, min(pad_force_n.values()),
                )
                force_gate_passed = force_gate.observe(
                    pad_force_n["left"], pad_force_n["right"], model.opt.timestep,
                )
            peak_joint_error = max(peak_joint_error, float(np.max(
                np.abs(data.qpos[arm_indices] - data.ctrl[:6]),
            )))
            step_index += 1
            if step_index % 40 == 0:
                viewer.sync()
                time.sleep(max(0.0, start_wall + float(data.time) / speed - time.monotonic()))
        completed = bool(data.time >= duration)
        viewer.sync()
        result = {
            "plan": str(plan_path),
            "frames": len(joints),
            "object_ids": [object_id, *geometry["other_object_ids"]],
            "proxy_mass_kg_each": mass_kg,
            "proxy_table_z_m": geometry["table_z_m"],
            "collision_geometry": geometry["collision_geometry"],
            "rubber_pad_collision_proxy": rubber_pads,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "pressure_rad": pressure_rad,
            "fully_close": fully_close,
            "grasp_dwell_s": grasp_dwell_s,
            "minimum_pad_force_n": minimum_pad_force_n,
            "force_hold_s": force_hold_s if force_gate is not None else None,
            "force_gate_passed": force_gate_passed if force_gate is not None else None,
            "peak_pad_force_n": peak_pad_force_n if force_gate is not None else None,
            "peak_simultaneous_pad_force_n": (peak_simultaneous_pad_force_n
                                               if force_gate is not None else None),
            "aborted_before_lift": aborted_before_lift,
            "object_grasp": grasp is not None,
            "completed": completed,
            "physics_time_s": float(data.time),
            "peak_joint_tracking_error_rad": peak_joint_error,
            "initial_object_positions_m": initial_positions,
            "final_object_positions_m": {
                name: data.xpos[body_id].tolist()
                for name, body_id in zip(object_names, object_body_ids)
            },
        }
        print(json.dumps(result, indent=2), flush=True)
        if completed or aborted_before_lift:
            if aborted_before_lift:
                print("Grasp force gate failed; lift cancelled. Inspect the frozen pickup in MuJoCo.", flush=True)
            else:
                print("Playback finished. Close the MuJoCo window when done inspecting.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--object-id", type=int, default=4)
    parser.add_argument("--mass-kg", type=float, default=0.2)
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--single-object", action="store_true")
    parser.add_argument("--rubber-pads", action="store_true")
    parser.add_argument("--official-mesh", type=Path)
    parser.add_argument("--gt-object-poses", type=Path)
    parser.add_argument("--gt-anchor-frame", type=int)
    parser.add_argument("--object-name")
    parser.add_argument("--pressure-rad", type=float, default=0.0)
    parser.add_argument("--fully-close", action="store_true")
    parser.add_argument("--grasp-dwell-s", type=float, default=0.0)
    parser.add_argument("--min-pad-force-n", type=float)
    parser.add_argument("--force-hold-s", type=float, default=0.1)
    arguments = parser.parse_args()
    watch(arguments.plan, arguments.track, arguments.models,
          arguments.object_id, arguments.mass_kg, arguments.time_scale,
          arguments.speed, arguments.frame_start, arguments.frame_end,
          not arguments.single_object, arguments.rubber_pads,
          arguments.official_mesh, arguments.gt_object_poses,
          arguments.gt_anchor_frame, arguments.object_name,
          arguments.pressure_rad, arguments.grasp_dwell_s,
          arguments.min_pad_force_n, arguments.force_hold_s, arguments.fully_close)
