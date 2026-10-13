"""A recorded cube scene with the exact cached UR10/Robotiq geometry."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import h5py
import mujoco
import numpy as np

from experiments.assemble_urdf import assembled_xml


ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
OBJECT_NAMES = {2: "blue", 3: "red", 4: "green"}


def scene_xml(plan_path: Path, object_model_path: Path, model_root: Path,
              red_offset_xy: np.ndarray | None = None,
              green_offset_xy: np.ndarray | None = None) -> str:
    with h5py.File(plan_path) as plan:
        transform = np.asarray(plan["T_calibration_rig"], dtype=float)
        base_position = np.asarray(plan["base_position_calibration"], dtype=float)
        base_rpy = np.deg2rad(np.asarray(plan["base_rpy_deg_calibration"], dtype=float))
        if plan["robot_key"][()].decode() != "ur10":
            raise ValueError("only the old UR10 plan is supported")
    with np.load(object_model_path, allow_pickle=False) as objects:
        positions_rig = np.asarray(objects["translation_world"][0], dtype=float)
        object_ids = objects["object_ids"].tolist()
    positions = (transform[:3, :3] @ positions_rig.T).T + transform[:3, 3]
    xy = {OBJECT_NAMES[object_id]: positions[index, :2]
          for index, object_id in enumerate(object_ids) if object_id in OBJECT_NAMES}
    if set(xy) != {"blue", "red", "green"}:
        raise ValueError("scene needs blue manipulated, red support and green target")
    red_offset = np.zeros(2) if red_offset_xy is None else np.asarray(red_offset_xy, float)
    green_offset = np.zeros(2) if green_offset_xy is None else np.asarray(green_offset_xy, float)
    if (red_offset.shape != (2,) or green_offset.shape != (2,)
            or not np.isfinite(red_offset).all() or not np.isfinite(green_offset).all()
            or np.max(np.abs(red_offset)) > 0.05
            or np.max(np.abs(green_offset)) > 0.05):
        raise ValueError("cube XY offsets must be finite and within 50 mm per axis")
    xy["red"] = xy["red"] + red_offset
    xy["blue"] = xy["blue"] + red_offset
    xy["green"] = xy["green"] + green_offset
    if np.linalg.norm(xy["blue"] - xy["red"]) > 0.07:
        raise ValueError("blue cube is not initially above the red cube")
    center_xy = (xy["red"] + xy["green"]) / 2
    with tempfile.TemporaryDirectory() as directory:
        urdf_path = Path(directory) / "ur10_robotiq.urdf"
        urdf_path.write_text(assembled_xml(model_root))
        root = ET.fromstring(mujoco.MjSpec.from_file(str(urdf_path)).to_xml())
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
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
                  pos=f"{center_xy[0]} {center_xy[1]} -0.025",
                  size="0.35 0.6 0.025", friction="1.2 0.005 0.0001",
                  rgba="0.75 0.75 0.75 1")
    for name in ("red", "green", "blue"):
        position = np.array([*xy[name],
                             0.06 if name == "blue" else 0.02])
        body = ET.SubElement(world, "body", name=f"{name}_cube",
                             pos=" ".join(map(str, position)))
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box",
                      size="0.02 0.02 0.02", mass="0.064",
                      friction="1.2 0.005 0.0001",
                      rgba={"red": "0.8 0.1 0.1 1", "green": "0.1 0.7 0.1 1",
                            "blue": "0.1 0.1 0.8 1"}[name])
    contacts = ET.SubElement(root, "contact")
    gripper_bodies = [body.get("name") for body in robot.iter("body")
                      if "phalanx" in body.get("name", "") or "knuckle" in body.get("name", "")]
    for first, second in combinations(["wrist_3_link", *gripper_bodies], 2):
        ET.SubElement(contacts, "exclude", body1=first, body2=second)
    actuators = ET.SubElement(root, "actuator")
    for name in ARM_JOINTS:
        ET.SubElement(actuators, "position", name=f"cmd_{name}", joint=name,
                      kp="3000", kv="180")
    ET.SubElement(actuators, "position", name="cmd_finger", joint="finger_joint",
                  kp="100", kv="5")
    return ET.tostring(root, encoding="unicode")


def load_scene(plan_path: Path, object_model_path: Path,
               model_root: Path,
               red_offset_xy: np.ndarray | None = None,
               green_offset_xy: np.ndarray | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        scene_xml(plan_path, object_model_path, model_root,
                  red_offset_xy, green_offset_xy))
