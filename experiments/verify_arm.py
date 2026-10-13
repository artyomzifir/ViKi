"""Compare cached old-UR10 MuJoCo kinematics with ViKi's URDF model."""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pinocchio as pin


JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def verify(plan_path: Path, model_root: Path) -> dict:
    paths = sorted(glob.glob(str(model_root / "robot_descriptions/xacrodoc/"
                                 "ur10_official_description/*.urdf")))
    if not paths:
        raise FileNotFoundError("cached old-UR10 URDF is required; UR10e is not a substitute")
    urdf = Path(paths[0])
    with h5py.File(plan_path) as plan:
        if plan["robot_key"][()].decode() != "ur10":
            raise ValueError("plan does not target the old UR10")
        if tuple(json.loads(plan["joint_names_json"][()].decode())) != JOINT_NAMES:
            raise ValueError("plan joint order differs from old UR10 model")
        trajectory = np.asarray(plan["q"], dtype=float)
    pin_model = pin.buildModelFromUrdf(str(urdf))
    pin_data = pin_model.createData()
    with tempfile.TemporaryDirectory() as directory:
        normalized = Path(directory) / "ur10.urdf"
        normalized.write_text(urdf.read_text().replace("file://", ""))
        mj_model = mujoco.MjModel.from_xml_path(str(normalized))
    mj_data = mujoco.MjData(mj_model)
    mj_joints = tuple(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, index)
                      for index in range(mj_model.njnt))
    if mj_joints != JOINT_NAMES:
        raise ValueError(f"MuJoCo joint order differs: {mj_joints}")
    if pin_model.nq != mj_model.nq or pin_model.nq != len(JOINT_NAMES):
        raise ValueError("arm degrees of freedom differ")
    body_id = mj_model.body("wrist_3_link").id
    frame_id = pin_model.getFrameId("wrist_3_link")
    position_errors = []
    rotation_errors = []
    sample_indices = np.linspace(0, len(trajectory) - 1, min(15, len(trajectory)),
                                 dtype=int)
    for joint_positions in trajectory[sample_indices]:
        pin.forwardKinematics(pin_model, pin_data, joint_positions)
        pin.updateFramePlacements(pin_model, pin_data)
        mj_data.qpos[:] = joint_positions
        mujoco.mj_forward(mj_model, mj_data)
        reference = pin_data.oMf[frame_id]
        position_errors.append(float(np.linalg.norm(
            reference.translation - mj_data.xpos[body_id])))
        rotation_errors.append(float(np.linalg.norm(pin.log3(
            reference.rotation.T @ mj_data.xmat[body_id].reshape(3, 3)))))
    result = {
        "urdf": str(urdf),
        "frames_checked": int(len(sample_indices)),
        "max_position_error_m": max(position_errors),
        "max_rotation_error_rad": max(rotation_errors),
    }
    result["accepted"] = (result["max_position_error_m"] <= 1e-5
                          and result["max_rotation_error_rad"] <= 1e-4)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    arguments = parser.parse_args()
    report = verify(arguments.plan, arguments.models)
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)
