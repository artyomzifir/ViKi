"""Assemble cached old-UR10 and ViKi Robotiq URDFs for simulator checks."""

from __future__ import annotations

import glob
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def cached_description(model_root: Path, name: str) -> Path:
    paths = sorted(glob.glob(str(model_root / "robot_descriptions/xacrodoc" /
                                 name / "*.urdf")))
    if not paths:
        raise FileNotFoundError(f"cached {name} URDF is required")
    return Path(paths[0])


def assembled_xml(model_root: Path, robot: str = "ur10") -> str:
    if robot not in {"ur10", "iiwa14"}:
        raise ValueError(f"unsupported simulation robot: {robot}")
    description = "ur10_official_description" if robot == "ur10" else "iiwa14_description"
    mount = "tool0" if robot == "ur10" else "iiwa_link_ee"
    arm = ET.parse(cached_description(model_root, description))
    gripper = ET.parse(cached_description(model_root, "robotiq_2f85_v4_description"))
    arm_root = arm.getroot()
    gripper_root = gripper.getroot()
    arm_links = {link.get("name") for link in arm_root.findall("link")}
    for link in gripper_root.findall("link"):
        if link.get("name") == "base_link":
            link.set("name", "robotiq_mount_base_link")
        if link.get("name") in arm_links:
            raise ValueError(f"duplicate arm/gripper link: {link.get('name')}")
        arm_root.append(link)
    for joint in gripper_root.findall("joint"):
        for connection in (joint.find("parent"), joint.find("child")):
            if connection is not None and connection.get("link") == "base_link":
                connection.set("link", "robotiq_mount_base_link")
        arm_root.append(joint)
    adapter = ET.SubElement(arm_root, "joint",
                            name="viki_flange_to_gripper_adapter", type="fixed")
    ET.SubElement(adapter, "parent", link=mount)
    ET.SubElement(adapter, "child", link="robotiq_mount_base_link")
    ET.SubElement(adapter, "origin", xyz="0 0 0.011", rpy="0 0 0")
    return ET.tostring(arm_root, encoding="unicode").replace("file://", "")


def load_combined_model(model_root: Path) -> mujoco.MjModel:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ur10_robotiq.urdf"
        path.write_text(assembled_xml(model_root))
        return mujoco.MjModel.from_xml_path(str(path))


if __name__ == "__main__":
    model = load_combined_model(Path("models"))
    print(f"nq={model.nq} njnt={model.njnt} ngeom={model.ngeom}")
    print([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
           for index in range(model.njnt)])
