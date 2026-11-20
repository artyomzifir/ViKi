"""Build a local UR10 + right Tesollo DG5F diagnostic model.

The vendor DG5F geometry and joint limits stay intact. The DG5F circular
mount sits on the visible UR10 wrist_3 end face along its +Z axis. The
twist around the axis is 0 degrees (orientation A, selected by the user). The
10 mm spacer is a provisional cylinder, not manufacturing CAD.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import trimesh

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'models/tesollo_dg5f_mujoco'
SOURCE = VENDOR / 'robot/urdf/dg5f_right.urdf'
ASSETS = VENDOR / 'robot/assets/dg5f_right'
CONVERTED = VENDOR / 'converted'
OUT = ROOT / 'data/_hocap_demo_source/ur10_tesollo.urdf'
PALM_ROLL_RAD = 0.0  # orientation A: straight axial mount, chosen after A-D comparison


def build():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'{SOURCE}: clone https://github.com/VAlikV/tesollo_dg5f_mujoco.git')
    arm_file = sorted((ROOT / 'models/robot_descriptions/xacrodoc/ur10_official_description').glob('*.urdf'))[0]
    arm = ET.parse(arm_file).getroot()
    hand = ET.parse(SOURCE).getroot()
    CONVERTED.mkdir(parents=True, exist_ok=True)
    converted = {}
    for mesh in hand.findall('.//mesh'):
        rel = mesh.attrib['filename']
        if not rel.startswith('meshes/dg5f_right/'):
            raise ValueError(f'unexpected DG5F mesh path: {rel}')
        src = ASSETS / rel.removeprefix('meshes/dg5f_right/')
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.suffix.lower() == '.dae':
            dst = CONVERTED / (src.stem + '.obj')
            if src not in converted:
                if not dst.is_file() or dst.stat().st_mtime_ns < src.stat().st_mtime_ns:
                    geo = trimesh.load(src, force='mesh')
                    if len(geo.vertices) < 20 or len(geo.faces) < 20:
                        raise ValueError(f'invalid visual mesh: {src}')
                    geo.export(dst)
                converted[src] = dst
            src = dst
        mesh.set('filename', '/app/' + str(src.relative_to(ROOT)))
    present = {link.get('name') for link in arm.findall('link')}
    for link in hand.findall('link'):
        if link.get('name') in present:
            raise ValueError(f'duplicate link: {link.get("name")}')
        arm.append(link)
    for joint in hand.findall('joint'):
        arm.append(joint)
    # Vendor UR10e uses wrist_3 +Y.  Our official UR10 visual/collision mesh
    # has its circular end face on wrist_3 +Z, at z=0 after its URDF origin.
    # Mount the DG5F circular base on that visible end face; the only
    # provisional geometry is the 10 mm straight cylinder between faces.
    mount_offset = 0.010
    adapter_length = 0.010
    adapter_center = adapter_length / 2
    import numpy as np
    from scipy.spatial.transform import Rotation
    mount_rotation = Rotation.from_euler('z', PALM_ROLL_RAD)
    mount_rpy = mount_rotation.as_euler('xyz')
    assert np.allclose(mount_rotation.apply([0, 0, 1]), [0, 0, 1], atol=1e-12)
    assert abs((adapter_center - adapter_length/2)) < 1e-12
    assert abs((adapter_center + adapter_length/2) - mount_offset) < 1e-12
    bridge = ET.SubElement(arm, 'link', name='tesollo_adapter')
    inertial = ET.SubElement(bridge, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=f'0 0 {adapter_center:.6f}', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value='0.10')
    ET.SubElement(inertial, 'inertia', ixx='0.0000204', ixy='0', ixz='0',
                  iyy='0.0000204', iyz='0', izz='0.0000392')
    for kind in ('visual', 'collision'):
        node = ET.SubElement(bridge, kind)
        ET.SubElement(node, 'origin', xyz=f'0 0 {adapter_center:.6f}', rpy='0 0 0')
        ET.SubElement(ET.SubElement(node, 'geometry'), 'cylinder', radius='0.028', length=f'{adapter_length:.6f}')
    joint = ET.SubElement(arm, 'joint', name='tool0_tesollo_adapter', type='fixed')
    ET.SubElement(joint, 'parent', link='tool0')
    ET.SubElement(joint, 'child', link='tesollo_adapter')
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    joint = ET.SubElement(arm, 'joint', name='tesollo_adapter_mount', type='fixed')
    ET.SubElement(joint, 'parent', link='tesollo_adapter')
    ET.SubElement(joint, 'child', link='rl_dg_mount')
    ET.SubElement(joint, 'origin', xyz=f'0 0 {mount_offset:.6f}',
                  rpy=' '.join(f'{v:.16g}' for v in mount_rpy))
    # As in the Shadow diagnostic, convert UR10's DAE visuals to OBJ for
    # Pinocchio and MuJoCo; collision bodies remain from the original URDF.
    for mesh in arm.findall('.//mesh'):
        name = mesh.get('filename', '').replace('file://', '')
        if not name.endswith('.dae'):
            continue
        src = Path(name.replace('/app/models/', str(ROOT / 'models/') + '/', 1))
        dst = CONVERTED / ('ur10_' + src.stem + '.obj')
        if not dst.is_file():
            geo = trimesh.load(src, force='mesh')
            geo.export(dst)
        mesh.set('filename', '/app/' + str(dst.relative_to(ROOT)))
    ET.SubElement(ET.SubElement(arm, 'mujoco'), 'compiler', discardvisual='false')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(ET.tostring(arm, encoding='unicode').replace('file://', ''))
    print(json.dumps(dict(model=str(OUT.relative_to(ROOT)),
                          sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                          converted_visuals=len(converted),
                          adapter='straight cylinder radius 28 mm length 10 mm at wrist3 Z=0..10 mm; provisional',
                          palm_roll_about_visible_flange_deg=float(np.rad2deg(PALM_ROLL_RAD)),
                          mount_axis_alignment=1.0), indent=2))


if __name__ == '__main__':
    build()
