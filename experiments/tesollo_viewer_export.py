"""Browser scene from the exact saved Tesollo DG5F commands, with source identity."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from viki.retarget.solver import PinocchioKinematics
from viki.retarget.robots import normalize_robot
from viki.retarget.frames import base_transform

def export(episode):
    source=episode/'tesollo_object_plan.npz'
    with np.load(source,allow_pickle=False) as p:
        values={k:p[k] for k in p.files}
    modelpath=Path('data/_hocap_demo_source/ur10_tesollo.urdf')
    model=pin.buildModelFromUrdf(str(modelpath))
    visual=pin.buildGeomFromUrdf(model,str(modelpath),pin.GeometryType.VISUAL)
    kin=PinocchioKinematics(SimpleNamespace(model=model,visual_model=visual), 'rl_dg_palm',
        collision_pairs=0,collision_min_distance_m=.001,
        actuated_joint_names=normalize_robot('ur10').joint_names,
        passive_joint_positions={str(n):values['tesollo_q'][:,i] for i,n in enumerate(values['tesollo_joint_names'])},
        gripper_prefix='rj_dg_')
    base=base_transform(values['base_position'],values['base_rpy_deg'])
    points=[]; placements=[]; rotations=[]
    for i,q in enumerate(values['arm_q']):
        kin.set_frame(i)
        points.append(kin.joint_points(q)@base[:3,:3].T+base[:3,3])
        placements.append(base@kin.joint_placements(q))
        rotations.append(base[:3,:3]@kin.pose(q)[1])
    visuals=kin.visual_geometries()
    for item in visuals:
        if 'mesh_path_abs' in item:
            absolute=item.pop('mesh_path_abs')
            prefix='/app/models/tesollo_dg5f_mujoco/'
            if not absolute.startswith(prefix): raise ValueError(absolute)
            item['mesh_url']='/api/pipeline/retarget/tesollo-mesh/'+absolute[len(prefix):]
    errors=np.linalg.norm(values['target_position']-values['achieved_position'],axis=1)
    status=json.loads(str(values['report_json']))['solver_status']
    label='UR10 + Tesollo DG5F' + (' (diagnostic, collision)' if status != 'framewise_ik_whole_trajectory_validated' else '')
    scene=dict(ready=True,id=episode.name,robot=label,robot_key='ur10',
        gripper_model='tesollo_hand_right',ee_frame='rl_dg_palm',solver_status=json.loads(str(values['report_json']))['solver_status'],
        n_frames=len(points),fps=1e6/np.median(np.diff(values['timestamps'])),base_position=values['base_position'].tolist(),
        base_rpy_deg=values['base_rpy_deg'].tolist(),base_transform=base.tolist(),reach_m=1.5,
        link_edges=kin.link_edges.tolist(),link_groups=list(kin.link_groups),point_groups=list(kin.point_groups),
        link_positions=np.array(points).tolist(),robot_visuals=visuals,robot_joint_placements=np.array(placements).tolist(),
        target_trajectory=values['target_position'].tolist(),achieved_trajectory=values['achieved_position'].tolist(),
        target_rotation=values['target_rotation'].tolist(),achieved_rotation=np.array(rotations).tolist(),
        position_error_m=errors.tolist(),gripper_closed=[False]*len(points),metrics=json.loads(str(values['report_json'])),
        plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest())
    (episode/'tesollo_object_viewer.json').write_text(json.dumps(scene,separators=(',',':')))

if __name__=='__main__': export(Path('data/datasets/hocap_demo_object_v4/g10_object2_190'))
