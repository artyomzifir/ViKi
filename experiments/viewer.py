"""Watch a recorded cube-transfer plan in the native MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from experiments.scene import ARM_JOINTS, load_scene


def watch(episode: Path, model_root: Path, speed: float,
          close_after_s: float | None = None) -> dict:
    if speed <= 0:
        raise ValueError("speed must be positive")
    plan_path = episode / "plan.h5"
    object_model_path = (episode / "intermediates/segmentation/"
                         "sam2.1_hiera_small/object_models.npz")
    model = load_scene(plan_path, object_model_path, model_root)
    data = mujoco.MjData(model)
    with h5py.File(plan_path) as plan:
        approach = np.asarray(plan["q_approach"], dtype=float)
        joints = np.asarray(plan["q"], dtype=float)
        drive = np.asarray(plan["gripper_joint_position"], dtype=float)
        timestamps = np.asarray(plan["timestamps"], dtype=float)
        frame_dt = float(plan["dt"][()])
    times = (timestamps - timestamps[0]) / 1e6
    if not np.all(np.diff(times) > 0):
        raise ValueError("plan timestamps must be strictly increasing")
    warmup_s = 0.3
    approach_times = warmup_s + np.arange(len(approach)) * frame_dt
    demonstration_times = warmup_s + len(approach) * frame_dt + times
    command_times = np.concatenate(([0.0, warmup_s], approach_times[1:],
                                    demonstration_times))
    command_joints = np.concatenate((approach[:1], approach, joints))
    command_drive = np.concatenate((np.zeros(len(approach) + 1), drive))
    duration = float(demonstration_times[-1] + 0.7)
    arm_qpos = [int(model.jnt_qposadr[model.joint(name).id]) for name in ARM_JOINTS]
    drive_qpos = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_qpos] = approach[0]
    data.ctrl[:6] = approach[0]
    data.ctrl[6] = 0.0
    mujoco.mj_forward(model, data)
    red_body = model.body("red_cube").id
    green_body = model.body("green_cube").id
    blue_body = model.body("blue_cube").id
    blue_geom = model.geom("blue_geom").id
    center = (data.xpos[red_body] + data.xpos[green_body]) / 2
    max_joint_error = 0.0
    max_drive_error = 0.0
    blue_peak_z = float(data.xpos[blue_body, 2])
    max_robot_penetration = 0.0
    start_wall = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [center[0], center[1], 0.1]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -30
        viewer.sync()
        print("MuJoCo window open; trajectory starts in 2 seconds.", flush=True)
        while viewer.is_running() and time.monotonic() - start_wall < 2:
            time.sleep(0.05)
        start_sim_wall = time.monotonic()
        step = 0
        while viewer.is_running() and data.time < duration:
            if close_after_s is not None and time.monotonic() - start_wall >= close_after_s:
                break
            now = float(data.time)
            for joint_index in range(len(ARM_JOINTS)):
                data.ctrl[joint_index] = np.interp(
                    now, command_times, command_joints[:, joint_index])
            data.ctrl[6] = np.interp(now, command_times, command_drive)
            mujoco.mj_step(model, data)
            max_joint_error = max(max_joint_error, float(np.max(
                np.abs(data.qpos[arm_qpos] - data.ctrl[:6]))))
            max_drive_error = max(max_drive_error, float(abs(
                data.qpos[drive_qpos] - data.ctrl[6])))
            blue_peak_z = max(blue_peak_z, float(data.xpos[blue_body, 2]))
            for contact in data.contact[:data.ncon]:
                if blue_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == blue_geom else contact.geom1
                if model.geom_bodyid[other] not in (
                    model.body("red_cube").id, model.body("green_cube").id, 0
                ):
                    max_robot_penetration = max(
                        max_robot_penetration, max(0.0, -float(contact.dist)))
            step += 1
            if step % 5 == 0:
                viewer.sync()
                deadline = start_sim_wall + float(data.time) / speed
                time.sleep(max(0.0, deadline - time.monotonic()))
        viewer.sync()
        result = {
            "episode": episode.name,
            "physics_time_s": float(data.time),
            "blue_peak_z_m": blue_peak_z,
            "max_robot_blue_penetration_m": max_robot_penetration,
            "valid_contact_physics": max_robot_penetration <= 0.004,
            "max_joint_tracking_error_rad": max_joint_error,
            "max_gripper_tracking_error_rad": max_drive_error,
            "valid_tracking": max_joint_error <= 0.1 and max_drive_error <= 0.1,
        }
        print(json.dumps(result, indent=2), flush=True)
        if close_after_s is None:
            print("Close the MuJoCo window when done inspecting.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--close-after-s", type=float)
    arguments = parser.parse_args()
    watch(arguments.episode, arguments.models, arguments.speed,
          arguments.close_after_s)
