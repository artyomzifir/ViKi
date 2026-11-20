"""Bounded offline object-pose-aware placement on the frozen G10_2 Shadow scene.

The source robot/object pose is observed in an unforced nominal MuJoCo rollout.
That simulator-state oracle is used only to generate an exported 190-frame
open-loop command; candidate replay and the additional 2 s settle are unforced.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.hocap_shadow_polisher import EP,PLAN,TRACK,URDF,source,make_scene,rollout,sha
from experiments.hocap_shadow_palm_polisher import build_kinematics,sampled_joint_velocity_margin
from experiments.hocap_shadow_multifinger_polisher import (
    expected_export_hashes,full_digit_trace,kinematic_margins,prepared_data,
    advance_frame,digit_forces,static_hold)
from experiments.shadow_object_pose_placement_controls import (
    compose,inverse,object_to_palm,support_center_height,smoothstep,
    corrected_object_and_palm,settled_placement_gate)
from experiments.shadow_polisher_controls import command_margins,validate_command_export
from experiments.shadow_palm_controls import solve_arm_pose,validate_ik_errors
from viki.retarget.frames import base_rotation

INPUT=Path('/app/rl-output/shadow_multifinger_followup_20261025/final_commands.npz')
OBS_FRAME=90
END_FRAME=125
RELEASE_END=151
RETREAT_END=175
PROTOCOL={
    'schema':'shadow_object_pose_placement_protocol_v1',
    'scope':'single frozen 190-frame scene; offline MuJoCo-state oracle from source unforced rollout frame90; final command open-loop',
    'source_grasp':'followup MF +0.10, RF +0.08 rad; source finger positions and timestamps bitwise fixed',
    'object_target':'tracked final XY and rotation, center Z from official mesh lowest transformed vertex on original table top plus 0 or 5 mm clearance',
    'grasp_transform':'actual simulated object pose and actual palm FK at frame90; fixed object-to-palm SE(3)',
    'command_family':'C1 object-center position+orientation correction start frame90/100, full by125; orientation gain .75/1; support clearance0/5mm; corrected palm hold125..151; +50mm world-z retreat151..175',
    'candidate_count':8,'no_physics_changes':True,
    'placement_diagnostic_gate':{'final_xy_error_mm':30,'final_z_error_mm':20,
        'final_orientation_error_deg':15,'final_upright_tilt_deg':15,
        'last_1s_table_support_fraction':.95,'last_1s_hand_force_steps':0,
        'last_1s_robot_table_steps':0,'last_1s_max_speed_m_s':.03,
        'last_1s_position_drift_mm':5,'last_1s_rotation_drift_deg':5},
    'ranking':'placement gate first; then 1000 for each failed release/support/hand/table/settle check + XYmm + Zmm + 2*rotationdeg + 2*tiltdeg + 0.1*original full-rollout score',
    'original_target_and_full_scene_gates':'retained exactly from hocap_shadow_polisher.rollout; pickup lag remains separate',
}

EXPLORATORY_PROTOCOL={**PROTOCOL,
    'schema':'shadow_object_pose_placement_exploratory_clearance_v1',
    'reason':'the frozen 0/5 mm mesh clearances all violated the unchanged finger-floor screen near frames 120-125',
    'command_family':'same start and orientation axes; center 10 or 15 mm above mesh support, release with original fingers, hold palm125..151, retreat151..175',
    'candidate_count':8}

def hashes():
    return {'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),
            'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)},
            'input':sha(INPUT)}


def settings(family='fixed8'):
    if family not in ('fixed8','exploratory_clearance8'):
        raise ValueError('unknown placement candidate family')
    clearances=(0.,.005) if family=='fixed8' else (.01,.015)
    return [{'start_frame':start,'rotation_gain':gain,'support_clearance_m':clearance}
            for start in (90,100) for gain in (.75,1.) for clearance in clearances]


def pose_dict(position,rotation):
    return {'position_m':np.asarray(position).tolist(),
            'rotation_matrix':np.asarray(rotation).tolist()}


def load_input(plan,names,baseline,h):
    with np.load(INPUT,allow_pickle=False) as saved:
        expected={**expected_export_hashes(h),'palm_input_sha256':sha(Path('/app/rl-output/shadow_palm_polisher_20261025/final_commands.npz'))}
        commands,timestamps,factor=validate_command_export(saved,expected,names,plan['timestamps'],baseline.shape,baseline.dtype)
        residual=np.asarray(saved['multifinger_residual'],dtype=float)
    if factor!=1. or not np.array_equal(residual,np.array([.1,.08,0,0,0,0,0,0])):
        raise ValueError('expected exact reviewed three-digit followup input')
    return commands


def fk_world(kin,frame,arm_q,base_position,base_rotation_matrix):
    kin.set_frame(int(frame))
    p,r=kin.pose(np.asarray(arm_q,dtype=float))
    return base_position+base_rotation_matrix@p,base_rotation_matrix@r


def setup_geometry(kin,plan,track,trajectory,source_commands,table_z):
    br=base_rotation(plan['base_rpy_deg']);bp=np.asarray(plan['base_position'],dtype=float)
    obj_pos=trajectory['object_positions'][OBS_FRAME]
    obj_rot=trajectory['object_rotations'][OBS_FRAME]
    actual_palm=fk_world(kin,OBS_FRAME,trajectory['q_actual'][OBS_FRAME,:6],bp,br)
    rel=object_to_palm(obj_pos,obj_rot,*actual_palm)
    rel_inverse=inverse(*rel)
    source_palm=[];nominal_obj=[]
    for frame,command in enumerate(source_commands):
        palm=fk_world(kin,frame,command[:6],bp,br)
        source_palm.append(palm)
        nominal_obj.append(compose(*palm,*rel_inverse))
    col=list(map(int,track['object_ids'])).index(2)
    reference_rotation=track['rotation_world_object'][-1,col]
    mesh=EP/'intermediates/segmentation/hocap-oracle-v1/meshes/object_2.obj'
    vertices=np.array([[float(v) for v in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
    target_xy=track['translation_world'][-1,col,:2].astype(float)
    baseline_support_z=support_center_height(vertices,reference_rotation,table_z)
    return {'base_position':bp,'base_rotation':br,'observed_object':(obj_pos,obj_rot),
            'observed_palm':actual_palm,'object_to_palm':rel,
            'source_palm':source_palm,'nominal_object':nominal_obj,
            'reference_rotation':reference_rotation,'target_xy':target_xy,
            'support_center_z_m':baseline_support_z,'mesh_vertices':vertices,'table_z':table_z}


def construct_candidate(kin,plan,names,source_commands,lower,upper,velocity,geometry,setting):
    commands=source_commands.copy()
    start=setting['start_frame'];gain=setting['rotation_gain'];clearance=setting['support_clearance_m']
    if start not in (90,100) or gain not in (.75,1.) or clearance not in (0.,.005,.01,.015):
        raise ValueError('setting outside frozen candidate family')
    nominal_end_p,nominal_end_r=geometry['nominal_object'][END_FRAME]
    wanted_p=np.r_[geometry['target_xy'],geometry['support_center_z_m']+clearance]
    full_rot=geometry['reference_rotation']@nominal_end_r.T
    endpoint_turn=Rotation.from_rotvec(gain*Rotation.from_matrix(full_rot).as_rotvec()).as_matrix()
    wanted_r=endpoint_turn@nominal_end_r
    endpoint_delta_p=wanted_p-nominal_end_p
    endpoint_delta_r=wanted_r@nominal_end_r.T
    rel_p,rel_r=geometry['object_to_palm']
    bp=geometry['base_position'];br=geometry['base_rotation']
    ik_pos=np.zeros(190);ik_rot=np.zeros(190)
    target_positions=np.full((190,3),np.nan);target_rotations=np.full((190,3,3),np.nan)
    realized_positions=np.full((190,3),np.nan);realized_rotations=np.full((190,3,3),np.nan)
    end_palm=None
    for frame in range(start+1,190):
        if frame<=END_FRAME:
            weight=smoothstep(frame,start,END_FRAME)
            nominal_p,nominal_r=geometry['nominal_object'][frame]
            _,_,target_p,target_r=corrected_object_and_palm(nominal_p,nominal_r,
                rel_p,rel_r,endpoint_delta_p,endpoint_delta_r,weight)
            if frame==END_FRAME:end_palm=(target_p,target_r)
        elif frame<=RELEASE_END:
            target_p,target_r=end_palm
        else:
            retreat=smoothstep(frame,RELEASE_END,RETREAT_END)
            target_p=end_palm[0]+np.array([0.,0.,.05*retreat])
            target_r=end_palm[1]
        target_positions[frame]=target_p;target_rotations[frame]=target_r
        robot_p=br.T@(target_p-bp);robot_r=br.T@target_r
        seed=commands[frame-1,:6] if frame>start+1 else source_commands[frame,:6]
        q,_,_=solve_arm_pose(kin,frame,seed,robot_p,robot_r,max_iterations=24)
        commands[frame,:6]=q.astype(np.float32)
        realized_p,realized_r=fk_world(kin,frame,commands[frame,:6],bp,br)
        realized_positions[frame]=realized_p;realized_rotations[frame]=realized_r
        ik_pos[frame]=np.linalg.norm(realized_p-target_p)
        ik_rot[frame]=Rotation.from_matrix(target_r@realized_r.T).magnitude()
        if not validate_ik_errors([ik_pos[frame]],[ik_rot[frame]]):
            return None,{'status':'rejected_ik','frame':frame,
                'position_error_mm':ik_pos[frame]*1000,'orientation_error_deg':float(np.rad2deg(ik_rot[frame]))},None
    if not np.array_equal(commands[:start+1],source_commands[:start+1]) or not np.array_equal(commands[:,6:],source_commands[:,6:]):
        raise RuntimeError('pre-correction or finger commands changed')
    pm,vm=command_margins(commands,plan['timestamps'],lower,upper,velocity)
    details={'position_margin_rad':pm,'command_velocity_margin_rad_s':vm,
             'max_ik_position_mm':float(np.max(ik_pos)*1000),
             'max_ik_orientation_deg':float(np.rad2deg(np.max(ik_rot))),
             'endpoint_delta_position_mm':(endpoint_delta_p*1000).tolist(),
             'endpoint_delta_rotation_deg':float(np.rad2deg(Rotation.from_matrix(endpoint_delta_r).magnitude())),
             'target_endpoint_object':pose_dict(wanted_p,wanted_r)}
    if pm< -1e-6 or vm< -1e-6:
        return None,{'status':'rejected_limits',**details},None
    floor,self_margin,bad_frame=kinematic_margins(kin,names,commands,range(start+1,190))
    details.update(floor_margin_m=floor,screened_self_collision_margin_m=self_margin)
    if bad_frame is not None:
        return None,{'status':'rejected_floor_or_self','frame':bad_frame,**details},None
    diagnostics={'target_palm_position_world':target_positions,
                 'target_palm_rotation_world':target_rotations,
                 'realized_palm_position_world':realized_positions,
                 'realized_palm_rotation_world':realized_rotations,
                 'ik_position_error_m':ik_pos,'ik_orientation_error_rad':ik_rot}
    return commands,{'status':'feasible',**details},diagnostics


def terminal_settle(model,plan,track,names,commands,table_z):
    data,_=prepared_data(model,names,commands)
    obj=model.body('object_2').id
    for frame in range(190):advance_frame(model,data,plan,commands,frame)
    data.ctrl[:]=commands[-1]
    positions=[data.xpos[obj].copy()];rotations=[data.xmat[obj].reshape(3,3).copy()]
    speeds=[];supports=[];hands=[];tables=[];digit_forces_trace=[]
    for _ in range(round(2./model.opt.timestep)):
        mujoco.mj_step(model,data)
        positions.append(data.xpos[obj].copy());rotations.append(data.xmat[obj].reshape(3,3).copy())
        speeds.append(float(np.linalg.norm(data.cvel[obj,3:])))
        digit_forces_trace.append(digit_forces(model,data,obj))
        support=False;hand=False;table=False
        for ci,c in enumerate(data.contact):
            b1,b2=(model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2])
            n1,n2=model.body(b1).name,model.body(b2).name
            if obj not in (b1,b2) and 'world' not in (n1,n2):continue
            f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
            if f[0]<=1e-6:continue
            if obj in (b1,b2):
                other=n2 if b1==obj else n1
                if other=='world':support=True
                if other.startswith('rh_') and f[0]>.05:hand=True
            if 'world' in (n1,n2):
                other=n2 if n1=='world' else n1
                if other.startswith('rh_') or other.endswith('link'):table=True
        supports.append(support);hands.append(hand);tables.append(table)
    positions=np.asarray(positions);rotations=np.asarray(rotations);speeds=np.asarray(speeds)
    supports=np.asarray(supports);hands=np.asarray(hands);tables=np.asarray(tables)
    col=list(map(int,track['object_ids'])).index(2)
    ref_p=track['translation_world'][-1,col]
    ref_r=track['rotation_world_object'][-1,col]
    final_p=positions[-1];final_r=rotations[-1]
    xy_error=float(np.linalg.norm(final_p[:2]-ref_p[:2])*1000)
    z_error=float(abs(final_p[2]-ref_p[2])*1000)
    angle_error=float(np.rad2deg(Rotation.from_matrix(ref_r@final_r.T).magnitude()))
    tilt=float(np.rad2deg(np.arccos(np.clip(final_r[2,2],-1.,1.))))
    last=round(1./model.opt.timestep)
    drift=float(np.max(np.linalg.norm(positions[-last-1:]-positions[-last-1],axis=1))*1000)
    turn=float(np.rad2deg(np.max(Rotation.from_matrix(rotations[-last-1:]@rotations[-last-1].T).magnitude())))
    support_fraction=float(supports[-last:].mean())
    hand_steps=int(hands[-last:].sum());table_steps=int(tables[-last:].sum())
    max_speed=float(np.max(speeds[-last:]))
    evidence={'final_xy_error_mm':xy_error,'final_z_error_mm':z_error,
            'final_orientation_error_deg':angle_error,'final_upright_tilt_deg':tilt,
            'last_1s_table_support_fraction':support_fraction,
            'last_1s_hand_force_steps':hand_steps,'last_1s_robot_table_steps':table_steps,
            'last_1s_max_speed_m_s':max_speed,'last_1s_peak_position_drift_mm':drift,
            'last_1s_peak_rotation_drift_deg':turn,
            'final_object_position_m':final_p.tolist()}
    evidence['placement_diagnostic_pass']=settled_placement_gate(evidence)
    return evidence,{'object_position':positions,'object_rotation':rotations,'speed_m_s':speeds,
            'table_support_steps':supports,'hand_object_steps':hands,'robot_table_steps':tables,
            'digit_force_n':np.asarray(digit_forces_trace)}


def placement_rank(metrics,settle):
    bad_release=not metrics['release_no_hand_contact_last_15_frames'] or settle['last_1s_hand_force_steps']>0
    bad_support=not metrics['terminal_supported_last_15_frames'] or settle['last_1s_table_support_fraction']<.95
    bad_table=metrics['robot_table_contact_steps']>0 or settle['last_1s_robot_table_steps']>0
    bad_settle=(not metrics['terminal_settled_last_15_frames'] or settle['last_1s_max_speed_m_s']>=.03
                or settle['last_1s_peak_position_drift_mm']>=5 or settle['last_1s_peak_rotation_drift_deg']>=5)
    score=1000*sum((bad_release,bad_support,bad_table,bad_settle))
    score+=settle['final_xy_error_mm']+settle['final_z_error_mm']
    score+=2*(settle['final_orientation_error_deg']+settle['final_upright_tilt_deg'])+.1*metrics['score']
    return float(score)


def export(path,commands,plan,names,h,setting,index):
    np.savez_compressed(path,schema='shadow_polisher_commands_v1',
        commands=commands,timestamps=plan['timestamps'],joint_names=np.asarray(names),
        plan_sha256=h['plan'],track_sha256=h['track'],urdf_sha256=h['urdf'],
        mesh_hashes_json=json.dumps(h['meshes'],sort_keys=True),carry_timing_factor=1.,
        source_grasp_sha256=h['input'],placement_setting_json=json.dumps(setting,sort_keys=True),
        placement_candidate_index=int(index))


def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/app/rl-output/shadow_object_pose_placement_20261026')
    parser.add_argument('--replay',help='replay a saved open-loop candidate with its source hashes')
    parser.add_argument('--family',choices=('fixed8','exploratory_clearance8'),default='fixed8')
    args=parser.parse_args();out=Path(args.output)
    if any((out/n).exists() for n in ('manifest.json','feasibility.jsonl','rollout_ledger.jsonl')):
        raise RuntimeError('existing placement run; use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source();h=hashes()
    source_commands=load_input(plan,names,baseline,h)
    model,table_z,xml=make_scene(plan,track,names)
    kin=build_kinematics({**plan,'shadow_q':source_commands[:,6:]})
    src_metrics,src_trajectory=rollout(model,plan,track,names,source_commands,table_z,full=True)
    geometry=setup_geometry(kin,plan,track,src_trajectory,source_commands,table_z)
    geo_record={'observation_frame':OBS_FRAME,'observed_object':pose_dict(*geometry['observed_object']),
        'observed_palm':pose_dict(*geometry['observed_palm']),
        'object_to_palm':pose_dict(*geometry['object_to_palm']),
        'nominal_endpoint_object':pose_dict(*geometry['nominal_object'][END_FRAME]),
        'target_reference_object':pose_dict(np.r_[geometry['target_xy'],geometry['support_center_z_m']],geometry['reference_rotation']),
        'table_top_z_m':table_z,'mesh_local_min_m':geometry['mesh_vertices'].min(axis=0).tolist(),
        'mesh_local_max_m':geometry['mesh_vertices'].max(axis=0).tolist()}
    if args.replay:
        with np.load(args.replay,allow_pickle=False) as saved:
            expected={**expected_export_hashes(h),'source_grasp_sha256':h['input']}
            commands,timestamps,factor=validate_command_export(saved,expected,names,plan['timestamps'],baseline.shape,baseline.dtype)
            setting=json.loads(str(saved['placement_setting_json']))
        if factor!=1. or setting not in settings(args.family):raise ValueError('unrecognized placement setting or timing')
        rebuilt,details,_=construct_candidate(kin,plan,names,source_commands,lower,upper,velocity,geometry,setting)
        if rebuilt is None or not np.array_equal(rebuilt,commands):raise ValueError('exported placement command mismatch')
        metrics,trajectory=rollout(model,plan,track,names,commands,table_z,full=True)
        settle,settle_trace=terminal_settle(model,plan,track,names,commands,table_z)
        report={'metrics':metrics,'settle':settle,'source_hashes':h,
                'command_sha256':sha(Path(args.replay)),'construction':details}
        write_json(out/'replay_report.json',report)
        np.savez_compressed(out/'replay_trajectory.npz',**trajectory)
        np.savez_compressed(out/'replay_settle_trace.npz',**settle_trace)
        print(json.dumps(report,indent=2),flush=True)
        return
    protocol=PROTOCOL if args.family=='fixed8' else EXPLORATORY_PROTOCOL
    write_json(out/'protocol.json',{'protocol':protocol,'settings':settings(args.family)})
    write_json(out/'geometry.json',geo_record)
    (out/'scene.xml').write_text(xml)
    manifest={'schema':'shadow_object_pose_placement_pilot_v1','protocol':protocol,
        'source_hashes':h,'source_command_sha256':h['input'],'geometry':geo_record,
        'physics':'identical hocap_shadow_polisher.make_scene; all four objects free; no forcing or weld',
        'accepted':False}
    write_json(out/'manifest.json',manifest)
    control_settle,_=terminal_settle(model,plan,track,names,source_commands,table_z)
    write_json(out/'source_control.json',{'metrics':src_metrics,'settle':control_settle,
        'placement_score':placement_rank(src_metrics,control_settle)})
    feasible=[];ledger=[]
    for i,setting in enumerate(settings(args.family)):
        command,details,diag=construct_candidate(kin,plan,names,source_commands,lower,upper,velocity,geometry,setting)
        row={'index':i,'setting':setting,**details}
        if command is not None:
            cmd_path=out/f'candidate_{i}_commands.npz'
            export(cmd_path,command,plan,names,h,setting,i)
            np.savez_compressed(out/f'candidate_{i}_ik.npz',**diag)
            row.update(command_file=cmd_path.name,command_sha256=sha(cmd_path))
            feasible.append((i,command))
        with (out/'feasibility.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        print(json.dumps({'phase':'IK','index':i,'status':row['status'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    for i,command in feasible:
        metrics,trajectory=rollout(model,plan,track,names,command,table_z,full=True)
        actual_margin=sampled_joint_velocity_margin(trajectory['q_actual'],plan['timestamps'],velocity)
        settle,settle_trace=terminal_settle(model,plan,track,names,command,table_z)
        score=placement_rank(metrics,settle)
        row={'index':i,'metrics':metrics,'settle':settle,'placement_score':score,
             'sampled_actual_velocity_margin_rad_s':actual_margin,
             'actual_velocity_eligible':bool(actual_margin>=-1e-6)}
        np.savez_compressed(out/f'candidate_{i}_trajectory.npz',**trajectory)
        np.savez_compressed(out/f'candidate_{i}_settle_trace.npz',**settle_trace)
        ledger.append(row)
        with (out/'rollout_ledger.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        print(json.dumps({'phase':'physics','index':i,'score':round(score,2),'placement_pass':settle['placement_diagnostic_pass'],'target_pass':metrics['target_task_success'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    eligible=[x for x in ledger if x['actual_velocity_eligible']]
    if not eligible:
        manifest.update(feasible_count=len(feasible),evaluated_count=len(ledger),failure='no candidate passed actual sampled velocity screen',total_wall_s=time.monotonic()-started)
        write_json(out/'manifest.json',manifest);return
    selected=min(eligible,key=lambda x:(not x['settle']['placement_diagnostic_pass'],x['placement_score'],x['index']))
    index=selected['index'];setting=settings(args.family)[index]
    with np.load(out/f'candidate_{index}_commands.npz',allow_pickle=False) as saved:commands=saved['commands'].copy()
    export(out/'final_commands.npz',commands,plan,names,h,setting,index)
    final_metrics,final_traj=rollout(model,plan,track,names,commands,table_z,full=True)
    final_settle,final_settle_trace=terminal_settle(model,plan,track,names,commands,table_z)
    if final_metrics!=selected['metrics'] or final_settle!=selected['settle']:
        raise RuntimeError('selected exported plan changed on re-evaluation')
    digits,digit_trace=full_digit_trace(model,plan,names,commands)
    np.savez_compressed(out/'trajectory.npz',**final_traj,**digit_trace)
    np.savez_compressed(out/'settle_trace.npz',**final_settle_trace)
    original_static=static_hold(model,plan,names,commands)
    report={'schema':'shadow_object_pose_placement_report_v1','selected_index':index,'setting':setting,
        'metrics':final_metrics,'settle':final_settle,'digit_contact':digits,
        'static_grasp_hold':original_static,'source_control':{'metrics':src_metrics,'settle':control_settle},
        'pickup_p95_mm':float(np.percentile(np.linalg.norm(final_traj['object_positions'][42:60]-track['translation_world'][42:60,list(map(int,track['object_ids'])).index(2)],axis=1)*1000,95)),
        'accepted':bool(final_metrics['target_task_success'] and final_metrics['full_scene_success'] and final_settle['placement_diagnostic_pass']),
        'selected_command_sha256':sha(out/'final_commands.npz')}
    write_json(out/'report.json',report)
    manifest.update(feasible_count=len(feasible),evaluated_count=len(ledger),
        selected_index=index,selected_setting=setting,selected_metrics=final_metrics,
        selected_settle=final_settle,selected_command_sha256=sha(out/'final_commands.npz'),
        accepted=report['accepted'],total_wall_s=time.monotonic()-started)
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'selected_index':index,'placement_pass':final_settle['placement_diagnostic_pass'],
        'target_pass':final_metrics['target_task_success'],'full_scene_pass':final_metrics['full_scene_success'],
        'total_wall_s':manifest['total_wall_s']},indent=2),flush=True)

if __name__=='__main__':main()
