"""Prepare an experimental UR10 + official Shadow Hand URDF for a local MuJoCo attempt.

Requires the downloaded models/shadow_hand tree and trimesh + pycollada.
Converted meshes remain ignored local assets; no robotic success is implied.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import trimesh

source=Path('models/shadow_hand/shadow_hand_right.urdf')
mesh_root=Path('models/shadow_hand/sr_common/sr_description')
output_root=Path('models/shadow_hand/converted')
output_root.mkdir(parents=True,exist_ok=True)
shadow=ET.parse(source).getroot()
for link in list(shadow.findall('link')):
    if link.get('name') in {'world','rh_forearm','rh_wrist'}:
        shadow.remove(link)
for joint in list(shadow.findall('joint')):
    if joint.get('name') in {'rh_world_joint','rh_WRJ1','rh_WRJ2'}:
        shadow.remove(joint)
for extra in list(shadow):
    if extra.tag in {'gazebo','transmission'}:
        shadow.remove(extra)
converted={}
for mesh in shadow.findall('.//mesh'):
    name=mesh.get('filename')
    if not name or not name.startswith('package://sr_description/'):
        continue
    rel=name.removeprefix('package://sr_description/')
    src=mesh_root/rel
    dst=output_root/(Path(rel).stem+'.obj')
    if src not in converted:
        geometry=trimesh.load(src,force='mesh')
        if len(geometry.vertices)<50 or len(geometry.faces)<50:
            raise ValueError(f'invalid Shadow mesh {src}')
        geometry.export(dst)
        converted[src]=dst
    mesh.set('filename',f'/app/models/shadow_hand/converted/{dst.name}')

arm_path=next(Path('models/robot_descriptions/xacrodoc/ur10_official_description').glob('*.urdf'))
arm=ET.parse(arm_path).getroot()
arm_names={x.get('name') for x in arm.findall('link')}
for link in shadow.findall('link'):
    if link.get('name') in arm_names: raise ValueError('duplicate link')
    arm.append(link)
for joint in shadow.findall('joint'):
    arm.append(joint)
adapter=ET.SubElement(arm,'joint',name='viki_flange_to_shadow_palm',type='fixed')
ET.SubElement(adapter,'parent',link='tool0')
ET.SubElement(adapter,'child',link='rh_palm')
ET.SubElement(adapter,'origin',xyz='0 0 0.055',rpy='0 0 0')
for mesh in arm.findall('.//mesh'):
    name=mesh.get('filename','').replace('file://','')
    if not name.endswith('.dae'): continue
    src=Path(name.replace('/app/models/','models/',1))
    dst=output_root/('ur10_'+src.stem+'.obj')
    if not dst.exists():
        geometry=trimesh.load(src,force='mesh')
        geometry.export(dst)
    mesh.set('filename',f'/app/models/shadow_hand/converted/{dst.name}')
# A visible, collidable flange adapter fills the declared 55-mm mount gap.
mount=ET.SubElement(arm,'link',name='shadow_adapter')
for kind in ['visual','collision']:
    part=ET.SubElement(mount,kind)
    ET.SubElement(part,'origin',xyz='0 0 0.0275',rpy='0 0 0')
    geom=ET.SubElement(part,'geometry')
    ET.SubElement(geom,'cylinder',radius='0.025',length='0.055')
fixed=ET.SubElement(arm,'joint',name='shadow_adapter_mount',type='fixed')
ET.SubElement(fixed,'parent',link='tool0')
ET.SubElement(fixed,'child',link='shadow_adapter')
compiler=ET.SubElement(arm,'mujoco')
ET.SubElement(compiler,'compiler',discardvisual='false')
out=Path('data/_hocap_demo_source/ur10_shadow.urdf')
out.write_text(ET.tostring(arm,encoding='unicode').replace('file://',''))
print('wrote',out,'meshes',len(converted),'bytes',out.stat().st_size)
