"""Unassisted dynamic replay of the saved UR10 + Tesollo DG5F object plan.

No object pose forcing, weld, or trajectory-dependent finger correction.
Official collision geometry; object meshes use MuJoCo's convex hull contacts.
"""
from pathlib import Path
import time
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from viki.retarget.robots import normalize_robot

EP=Path('/app/data/datasets/hocap_demo_object_v4/g10_object2_190')
parser=argparse.ArgumentParser()
parser.add_argument('--plan',default='tesollo_object_plan.npz')
parser.add_argument('--output',default='/app/rl-output/tesollo_object')
parser.add_argument('--gui',action='store_true')
parser.add_argument('--gui-hold-sec',type=float,default=900)
args=parser.parse_args()
OUT=Path(args.output)
OUT.mkdir(parents=True,exist_ok=True)
planpath=EP/args.plan
with np.load(planpath,allow_pickle=False) as p:
    plan={k:p[k] for k in p.files}
plan_report=json.loads(str(plan['report_json']))
plan_validated=plan_report['solver_status']=='framewise_ik_whole_trajectory_validated'
track=np.load(EP/'intermediates/segmentation/hocap-oracle-v1/object_models.npz')
root=ET.fromstring(mujoco.MjSpec.from_file('/app/data/_hocap_demo_source/ur10_tesollo.urdf').to_xml())
compiler=root.find('compiler')
compiler.set('angle','radian')
option=root.find('option')
if option is None: option=ET.SubElement(root,'option')
option.set('timestep','0.0005');option.set('integrator','implicitfast')
world=root.find('worldbody')
robot=ET.Element('body',name='robot_base',pos=' '.join(map(str,plan['base_position'])),
    euler=' '.join(map(str,np.deg2rad(plan['base_rpy_deg']))))
for child in list(world): world.remove(child);robot.append(child)
for body in robot.iter('body'): body.set('gravcomp','1')
# MuJoCo combines equal-priority geom friction with an element-wise maximum.
# Set only collidable robot geoms; visual meshes do not participate in contact.
for geom in robot.iter('geom'):
    if geom.get('contype') != '0' or geom.get('conaffinity') != '0':
        geom.set('friction','1.2 .005 .0001')
world.append(robot)
assets=root.find('asset')
if assets is None: assets=ET.SubElement(root,'asset')
table_z=0.
for i,oid in enumerate(track['object_ids']):
    mesh=EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{oid}.obj'
    verts=np.array([[float(x) for x in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
    pos=track['translation_world'][0,i];rot=track['rotation_world_object'][0,i]
    table_z=min(table_z,float((verts@rot.T+pos)[:,2].min()))
    ET.SubElement(assets,'mesh',name=f'object_mesh_{oid}',file=str(mesh))
    b=ET.SubElement(world,'body',name=f'object_{oid}',pos=' '.join(map(str,pos)),
        quat=' '.join(map(str,Rotation.from_matrix(rot).as_quat(scalar_first=True))))
    ET.SubElement(b,'freejoint',name=f'object_free_{oid}')
    colors=['.078 .722 .651 1','.655 .545 .98 1','.639 .902 .208 1','.984 .573 .235 1']
    ET.SubElement(b,'geom',name=f'object_geom_{oid}',type='mesh',mesh=f'object_mesh_{oid}',mass='.08',
        friction='1.2 .005 .0001',rgba=colors[i])
ET.SubElement(world,'geom',name='table',type='box',pos=f'0 0 {table_z-.025}',size='.7 .7 .025',rgba='.25 .28 .32 1',friction='1.0 .005 .0001',priority='1')
ET.SubElement(world,'light',pos='0 -1 2',dir='0 .3 -1',diffuse='.8 .8 .8')
contact=root.find('contact')
if contact is None: contact=ET.SubElement(root,'contact')
ET.SubElement(contact,'exclude',body1='forearm_link',body2='wrist_2_link')
act=ET.SubElement(root,'actuator')
names=list(normalize_robot('ur10').joint_names)+list(map(str,plan['tesollo_joint_names']))
efforts={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse('/app/data/_hocap_demo_source/ur10_tesollo.urdf').getroot().findall('joint') if j.find('limit') is not None}
for i,n in enumerate(names):
    ET.SubElement(act,'position',name='cmd_'+n,joint=n,kp='1800' if i<6 else '16',kv='120' if i<6 else '.15',
        forcerange=f'{-efforts[n]} {efforts[n]}')
visual=root.find('visual')
if visual is None: visual=ET.SubElement(root,'visual')
ET.SubElement(visual,'global',offwidth='1280',offheight='720')
ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.65 .65 .65',specular='.1 .1 .1')
xml=ET.tostring(root,encoding='unicode')
(OUT/'scene.xml').write_text(xml)
model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
addresses=np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
commands=np.c_[plan['arm_q'],plan['tesollo_q']]
data.qpos[addresses]=commands[0];data.ctrl[:]=commands[0]
mujoco.mj_forward(model,data)
renderer=mujoco.Renderer(model,height=600,width=640)
camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:]=[.02,-.05,.15];camera.distance=.85;camera.azimuth=225;camera.elevation=-32
overview=mujoco.MjvCamera();overview.type=mujoco.mjtCamera.mjCAMERA_FREE
overview.lookat[:]=[0,-.4,.3];overview.distance=2.0;overview.azimuth=135;overview.elevation=-25
opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1
# MuJoCo imports collision geoms into group 0, visual geoms into 1;
# objects and table are group 0, so keep both and hide only robot collisions.
for g in range(model.ngeom):
    if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name not in ['world','object_1','object_2','object_3','object_4']:
        model.geom_rgba[g,3]=0
opt.geomgroup[0]=1
writer=cv2.VideoWriter(str(OUT/'replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,600))
obj=model.body('object_2').id
positions=[];rotations=[];qactual=[];contacts=[];tablecontacts=[]
gui=None
if args.gui:
    import mujoco.viewer
    gui=mujoco.viewer.launch_passive(model,data)
    gui.cam.lookat[:]=overview.lookat;gui.cam.distance=1.8;gui.cam.azimuth=225;gui.cam.elevation=-25
    gui.sync()
    print('Native MuJoCo GUI opened:',gui.is_running(),flush=True)
    time.sleep(3)
for _ in range(round(.3/model.opt.timestep)): mujoco.mj_step(model,data)
start=data.xpos[obj].copy()
for t,command in enumerate(commands):
    frame_wall=time.monotonic()
    duration=(plan['timestamps'][t]-plan['timestamps'][t-1])/1e6 if t else 0.
    thumb=False;finger=False;tablehit=0
    steps=max(0,round(duration/model.opt.timestep))
    previous=commands[max(t-1,0)]
    for step in range(steps):
        # Sample at the saved timestamp: smoothly execute the preceding segment.
        data.ctrl[:]=previous+(command-previous)*(step+1)/steps
        mujoco.mj_step(model,data)
        for c in data.contact:
            b1,b2=[model.body(model.geom_bodyid[g]).name for g in [c.geom1,c.geom2]]
            if 'object_2' in (b1,b2):
                other=b2 if b1=='object_2' else b1
                thumb |= other.startswith('rl_dg_1')
                finger |= other.startswith(('rl_dg_2','rl_dg_3','rl_dg_4','rl_dg_5'))
            if model.geom('table').id in (c.geom1,c.geom2) and ('link' in b1 or 'link' in b2 or b1.startswith('rl_dg_') or b2.startswith('rl_dg_')): tablehit+=1
    if gui is not None:
        if not gui.is_running(): raise RuntimeError('GUI closed before validation finished')
        gui.sync()
        time.sleep(max(0.,duration*2-(time.monotonic()-frame_wall)))
    positions.append(data.xpos[obj].copy());rotations.append(data.xmat[obj].reshape(3,3).copy());qactual.append(data.qpos[addresses].copy())
    contacts.append([thumb,finger]);tablecontacts.append(tablehit)
    renderer.update_scene(data,camera,scene_option=opt)
    detail=renderer.render().copy()
    renderer.update_scene(data,overview,scene_option=opt)
    frame=np.concatenate([renderer.render().copy(),detail],axis=1)
    cv2.putText(frame,f'Tesollo DG5F Hand | saved object plan | frame {t+1}/190',(18,28),cv2.FONT_HERSHEY_SIMPLEX,.57,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(frame,'Dynamic replay: no weld / no object pose forcing',(18,53),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
    writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
writer.release();renderer.close()
positions=np.array(positions);qactual=np.array(qactual);contacts=np.array(contacts)
col=list(track['object_ids']).index(2)
error=np.linalg.norm(positions-track['translation_world'][:,col],axis=1)*1000
carry_start,carry_stop=json.loads(str(plan['report_json'])).get('object_relative_window',[42,126])
carry_contact_fraction=float(contacts[carry_start:carry_stop].all(axis=1).mean())
carry_p95=float(np.percentile(error[carry_start:carry_stop],95))
report=dict(schema='tesollo_unassisted_dynamic_replay_v1',frames=len(commands),plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),
    plan_solver_status=plan_report['solver_status'],plan_validated=plan_validated,
    object_position_mean_mm=float(error.mean()),object_position_median_mm=float(np.median(error)),object_position_p95_mm=float(np.percentile(error,95)),
    final_object_error_mm=float(error[-1]),max_lift_mm=float((positions[:,2]-start[2]).max()*1000),
    thumb_contact_frames=int(contacts[:,0].sum()),finger_contact_frames=int(contacts[:,1].sum()),opposing_contact_frames=int(contacts.all(axis=1).sum()),
    robot_table_contact_steps=int(sum(tablecontacts)),arm_tracking_median_rad=float(np.median(np.abs(qactual[:,:6]-commands[:,:6]))),
    hand_tracking_median_rad=float(np.median(np.abs(qactual[:,6:]-commands[:,6:]))),
    carry_opposing_contact_fraction=carry_contact_fraction,carry_object_error_p95_mm=carry_p95,
    object_mass_kg=.08,friction=1.2,object_friction=1.2,table_friction=1.0,robot_friction=1.2,table_contact_priority=1,gravity_compensation=True,
    servo_gains={'arm_kp':1800,'arm_kv':120,'hand_kp':16,'hand_kv':.15},actuator_force_limits='URDF effort',object_contact_geometry='convex hull of official aligned mesh',table_z=table_z,
    command_interpolation='linear between saved timestamps; state sampled at target timestamp',
    gui_enabled=args.gui,assistance='none',success=bool(plan_validated and (positions[:,2]-start[2]).max()>.02 and contacts.all(axis=1).sum()>10 and error[-1]<20 and carry_contact_fraction>=.8 and carry_p95<30 and sum(tablecontacts)==0))
np.savez_compressed(OUT/'trajectory.npz',object_positions=positions,object_rotations=rotations,q_actual=qactual,contacts=contacts)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

if gui is not None:
    print('Replay finished. Native GUI remains open for inspection.',flush=True)
    deadline=time.monotonic()+args.gui_hold_sec
    while gui.is_running() and time.monotonic()<deadline:
        gui.sync();time.sleep(.05)
    gui.close()
