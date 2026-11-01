"""Two-setting simulator-state feedback generation, exported as open-loop Shadow commands.

This is an offline oracle pilot. During generation only, it observes the true
simulated object/palm state; independent replay reads only saved controls.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.hocap_shadow_object_pose_placement import (source,hashes,load_input,
    make_scene,build_kinematics,setup_geometry,fk_world,pose_dict,write_json)
from experiments.hocap_shadow_multifinger_polisher import prepared_data,advance_frame,digit_forces
from experiments.shadow_feedback_placement_controls import (palm_for_object_step,
    smooth_palm_command,release_ready,force_residual_step,residual_finger_command,controlled_touchdown_gate)
from experiments.shadow_object_pose_placement_controls import smoothstep,settled_placement_gate
from experiments.shadow_palm_controls import solve_arm_pose,validate_ik_errors
from experiments.shadow_polisher_controls import command_margins
from experiments.hocap_shadow_polisher import rollout,sha
from viki.retarget.frames import base_rotation

ANCHOR=110
MAX_FRAMES=260
SETTINGS=(
    {'name':'gentle','object_step_m':.002,'object_turn_deg':1.,'palm_accel_m':.001,'palm_angular_accel_deg':.5},
    {'name':'firm','object_step_m':.003,'object_turn_deg':1.5,'palm_accel_m':.0015,'palm_angular_accel_deg':.75},
)
CONTACT_SETTINGS=(
    {**SETTINGS[1],'contact_name':'gentle_force','force_gain_rad_per_n_frame':.003,'force_caps_rad':[.06,.12,.12]},
    {**SETTINGS[1],'contact_name':'firm_force','force_gain_rad_per_n_frame':.006,'force_caps_rad':[.08,.18,.12]},
)
CONTACT_PROTOCOL={'schema':'shadow_contact_feedback_placement_protocol_v1',
    'source':'same saved three-digit command and firm arm-feedback law as preceding pilot; prefix0..110 fixed',
    'family':'two measured frame110 TH/FF/MF force-maintenance integral settings only, 80% of each measured force as target',
    'settings':CONTACT_SETTINGS,
    'synergies':'THJ4 and FF/MF J1/J2/J3 weights1/.7/.35; declared per-digit residual caps, no URDF clipping',
    'support_handling':'freeze closure after 3 consecutive frames with >=95% positive table support; release after original 9-frame support/upright/slow gate; fade residual smoothly over exact original 125..151 opening intervals',
    'controlled_touchdown_diagnostic':'TH+two distinct digits >.05N in >=90% pre-support physics steps, no gap>.05s, first table-contact speed<.1m/s; added after preceding upright-but-dropped result; no threshold tuning on these two variants',
    'scope':'offline simulator-state oracle; independent exported open-loop replay; unchanged model, collisions and original source prefix',
    'max_frames':MAX_FRAMES}

PROTOCOL={'schema':'shadow_feedback_placement_protocol_v1','candidate_count':2,
    'source':'saved MF+.10 RF+.08 grasp; original commands/timestamps through frame110 bitwise fixed',
    'oracle':'actual MuJoCo object and actual palm FK observed after each generated frame, only during offline generation',
    'controller':'current object-to-palm SE3, object-centered ≤2/3mm and ≤1/1.5deg steps; acceleration-limited palm command from source frame109→110 velocity',
    'phases':'align while airborne near anchor Z and XY/reference upright; descend to official mesh support+3mm only after upright tilt<8deg, XY<10mm, reference rotation<10deg; hold frame110 fingers until positive table support≥95% for9 frames, tilt<8deg and speed<.03m/s; then source finger release frames125..151 at their exact original segment durations, palm hold, 50mm rise over24frames, 15frame settle; max260 frames',
    'settings':SETTINGS,'extra_settle_s':2.,'no_physics_or_object_forcing':True,
    'evaluation':'extended placement gate separate from original190-frame baseline gate; no reference phase remapping of generated frames',
    'limits':'original joint position, per-segment command velocity, IK≤1.5mm/1deg, candidate passive-finger floor/self screen; no clamp of joint commands'}


def observe(model,data,obj,kin,frame,addresses,bp,br):
    p=data.xpos[obj].copy();r=data.xmat[obj].reshape(3,3).copy()
    q=data.qpos[addresses].copy()
    palm=fk_world(kin,frame,q[:6],bp,br)
    speed=float(np.linalg.norm(data.cvel[obj,3:]))
    tilt=float(np.rad2deg(np.arccos(np.clip(r[2,2],-1.,1.))))
    return p,r,palm,speed,tilt


def force_flags(model,data,obj):
    target_table=False;robot_table=False;hand=False
    for ci,c in enumerate(data.contact):
        b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
        n1,n2=model.body(b1).name,model.body(b2).name
        if obj not in (b1,b2) and 'world' not in (n1,n2):continue
        f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
        if f[0]<=1e-6:continue
        if obj in (b1,b2):
            other=n2 if b1==obj else n1
            if other=='world':target_table=True
            if other.startswith('rh_') and f[0]>.05:hand=True
        if 'world' in (n1,n2):
            other=n2 if n1=='world' else n1
            if other.startswith('rh_') or other.endswith('link'):robot_table=True
    return target_table,robot_table,hand


def step_command(model,data,previous,current,dt,object_id):
    steps=round(dt/model.opt.timestep)
    supports=[];tables=[];hands=[];digits=[];speeds=[];triads=[]
    for step in range(steps):
        data.ctrl[:]=previous+(current-previous)*(step+1)/steps
        mujoco.mj_step(model,data)
        support,table,hand=force_flags(model,data,object_id)
        supports.append(support);tables.append(table);hands.append(hand)
        force_values=digit_forces(model,data,object_id)
        digits.append(force_values)
        triads.append(bool(force_values[0]>.05 and np.sum(force_values[1:]>.05)>=2))
        speeds.append(float(np.linalg.norm(data.cvel[object_id,3:])))
    return {'support_fraction':float(np.mean(supports)),'robot_table_steps':int(sum(tables)),
            'hand_step_fraction':float(np.mean(hands)),'digit_mean_force_n':np.mean(digits,axis=0).tolist(),
            'max_speed_m_s':float(max(speeds)),
            'triad_step_fraction':float(np.mean(triads)),
            'triad_boolean_steps':triads,'support_boolean_steps':supports}


def frame_state(model,data,obj,kin,frame,addresses,bp,br,diagnostic):
    p,r,palm,speed,tilt=observe(model,data,obj,kin,frame,addresses,bp,br)
    return {'frame':frame,'object_position_m':p.tolist(),'object_rotation_world':r.tolist(),
            'actual_palm_position_m':palm[0].tolist(),'actual_palm_rotation_world':palm[1].tolist(),
            'object_speed_m_s':speed,'upright_tilt_deg':tilt,**diagnostic}


def screen_kinematics(kin,frame,command,previous,dt,lower,upper,velocity,passive,names):
    passive[frame,:]=command[6:]
    index={int(kin.model.idx_qs[kin.model.getJointId(n)]):i for i,n in enumerate(names)}
    for qidx,array in kin._passive_positions:array[frame]=float(command[index[qidx]])
    kin._collision_configuration_cache=None
    kin.set_frame(frame)
    pm=float(min(np.min(command-lower),np.min(upper-command)))
    vm=float(np.min(velocity-np.abs((command-previous)/dt)))
    floor=float(np.min(kin.floor_margins(command[:6])))
    self_margin=float(np.min(kin.collision_margins(command[:6])))
    return {'position_margin_rad':pm,'velocity_margin_rad_s':vm,
            'floor_margin_m':floor,'screened_self_margin_m':self_margin}


def generate(model,plan,track,names,source_commands,lower,upper,velocity,setting,geometry):
    nominal_dt_us=int(np.median(np.diff(plan['timestamps'])))
    passive=np.repeat(source_commands[ANCHOR:ANCHOR+1,6:],MAX_FRAMES,axis=0)
    passive[:ANCHOR+1]=source_commands[:ANCHOR+1,6:]
    kin=build_kinematics({**plan,'shadow_q':passive})
    data,addresses=prepared_data(model,names,source_commands)
    obj=model.body('object_2').id;bp=geometry['base_position'];br=geometry['base_rotation']
    commands=[x.copy() for x in source_commands[:ANCHOR+1]]
    times=list(map(int,plan['timestamps'][:ANCHOR+1]))
    for frame in range(ANCHOR+1):advance_frame(model,data,plan,source_commands,frame)
    records=[frame_state(model,data,obj,kin,ANCHOR,addresses,bp,br,{'stage':'source'})]
    previous_palm=fk_world(kin,ANCHOR,commands[-1][:6],bp,br)
    before_palm=fk_world(kin,ANCHOR-1,commands[-2][:6],bp,br)
    anchor_actual_palm = fk_world(kin, ANCHOR, data.qpos[addresses][:6], bp, br)
    fixed_preload_position = anchor_actual_palm[1].T @ (previous_palm[0] - anchor_actual_palm[0])
    fixed_preload_rotation = anchor_actual_palm[1].T @ previous_palm[1]
    last_dp=previous_palm[0]-before_palm[0]
    last_dr=Rotation.from_matrix(previous_palm[1]@before_palm[1].T).as_rotvec()
    initial_palm_command_step=last_dp.copy()
    palm_max_step=max(.006,float(np.linalg.norm(last_dp)))
    palm_max_turn=max(np.deg2rad(2.),float(np.linalg.norm(last_dr)))
    anchor_z=float(data.xpos[obj,2])
    align_z=max(geometry['support_center_z_m']+.04,anchor_z-.015)
    ref_p=np.r_[geometry['target_xy'],geometry['support_center_z_m']]
    ref_r=geometry['reference_rotation']
    contact_enabled='force_gain_rad_per_n_frame' in setting
    contact_residual=np.zeros(3,dtype=float)
    anchor_force=digit_forces(model,data,obj)[:3]
    target_force=.8*anchor_force
    last_force=anchor_force.copy()
    closure_frozen_frame=None;first_support_frame=None;pre_support_gap_steps=0
    max_pre_support_gap_steps=0;pre_support_steps=0;pre_support_triad_steps=0
    grip_lost_before_support=False
    stage='align';release_index=0;retreat_index=0;post_index=0
    support_count=0;start_release=None;abort=None
    hold_palm=None;retreat_palm=None
    for frame in range(ANCHOR+1,MAX_FRAMES):
        p,r,actual_palm,speed,tilt=observe(model,data,obj,kin,frame-1,addresses,bp,br)
        orientation_error=float(np.rad2deg(Rotation.from_matrix(ref_r@r.T).magnitude()))
        xy_error=float(np.linalg.norm(p[:2]-ref_p[:2]))
        if stage=='align' and tilt<8 and xy_error<.01 and orientation_error<10 and speed<.1:
            stage='descend'
        if stage in ('align','descend') and release_ready(float(records[-1].get('support_fraction',0)),tilt,speed,support_count):
            stage='release';start_release=frame;release_index=0;hold_palm=previous_palm
        if stage in ('align','descend') and contact_enabled and support_count<3:
            contact_residual=force_residual_step(contact_residual,last_force,target_force,
                setting['force_gain_rad_per_n_frame'],setting['force_caps_rad'])
        if stage in ('align','descend') and contact_enabled and support_count>=3 and closure_frozen_frame is None:
            closure_frozen_frame=frame
        if stage in ('align','descend'):
            target_z=align_z if stage=='align' else geometry['support_center_z_m']+.003
            target=np.r_[ref_p[:2],target_z]
            raw_p,raw_r=palm_for_object_step(p,r,*actual_palm,target,ref_r,
                setting['object_step_m'],np.deg2rad(setting['object_turn_deg']))
            if setting.get('fixed_command_deflection', False):
                raw_p = raw_p + raw_r @ fixed_preload_position
                raw_r = raw_r @ fixed_preload_rotation
            if setting.get('preserve_command_deflection', False):
                # Apply the observed object correction to the preceding COMMAND.
                # Replacing it with actual palm pose discards servo preload.
                turn = raw_r @ actual_palm[1].T
                raw_p = raw_p + turn @ (previous_palm[0] - actual_palm[0])
                raw_r = turn @ previous_palm[1]
            target_p,target_r,last_dp,last_dr=smooth_palm_command(raw_p,raw_r,*previous_palm,
                last_dp,last_dr,palm_max_step,palm_max_turn,
                setting['palm_accel_m'],np.deg2rad(setting['palm_angular_accel_deg']))
            finger=(residual_finger_command(source_commands[ANCHOR,6:],names[6:],contact_residual)
                if contact_enabled else source_commands[ANCHOR,6:])
        elif stage=='release':
            target_p,target_r=hold_palm
            finger=(residual_finger_command(source_commands[125+release_index,6:],names[6:],
                contact_residual*(1-smoothstep(release_index,0,26))) if contact_enabled
                else source_commands[125+release_index,6:])
        elif stage=='retreat':
            w=smoothstep(retreat_index,0,24)
            target_p=retreat_palm[0]+np.array([0.,0.,.05*w]);target_r=retreat_palm[1]
            finger=source_commands[151,6:]
        else:
            target_p,target_r=previous_palm;finger=source_commands[151,6:]
        step_us=(int(plan['timestamps'][125+release_index]-plan['timestamps'][124+release_index])
                 if stage=='release' else nominal_dt_us)
        dt=step_us*1e-6
        robot_p=br.T@(target_p-bp);robot_r=br.T@target_r
        kin.set_frame(frame)
        q,_,_=solve_arm_pose(kin,frame,commands[-1][:6],robot_p,robot_r,max_iterations=24)
        command=commands[-1].copy();command[:6]=q.astype(np.float32);command[6:]=finger
        realized=fk_world(kin,frame,command[:6],bp,br)
        ik_pos=float(np.linalg.norm(realized[0]-target_p));ik_rot=float(Rotation.from_matrix(target_r@realized[1].T).magnitude())
        margin=screen_kinematics(kin,frame,command,commands[-1],dt,lower,upper,velocity,passive,names)
        if not validate_ik_errors([ik_pos],[ik_rot]) or min(margin.values())< -1e-6:
            abort={'frame':frame,'stage':stage,'ik_position_mm':ik_pos*1000,
                   'ik_rotation_deg':float(np.rad2deg(ik_rot)),**margin}
            break
        times.append(times[-1]+step_us);commands.append(command)
        diagnostic=step_command(model,data,commands[-2],command,dt,obj)
        if diagnostic['support_fraction']>=.95:support_count+=1
        else:support_count=0
        if contact_enabled:
            last_force=np.asarray(diagnostic['digit_mean_force_n'][:3],dtype=float)
            for supported,triad in zip(diagnostic['support_boolean_steps'],diagnostic['triad_boolean_steps']):
                if first_support_frame is None:
                    if supported:first_support_frame=frame
                    else:
                        pre_support_steps+=1;pre_support_triad_steps+=triad
                        pre_support_gap_steps=0 if triad else pre_support_gap_steps+1
                        max_pre_support_gap_steps=max(max_pre_support_gap_steps,pre_support_gap_steps)
                        if pre_support_gap_steps*model.opt.timestep>.05:grip_lost_before_support=True
        records.append(frame_state(model,data,obj,kin,frame,addresses,bp,br,
            {'stage':stage,'contact_residual_rad':contact_residual.tolist(),'target_palm_position_m':target_p.tolist(),
             'target_palm_rotation_world':target_r.tolist(),
             'realized_ik_position_error_mm':ik_pos*1000,
             'realized_ik_rotation_error_deg':float(np.rad2deg(ik_rot)),**margin,**diagnostic}))
        previous_palm=realized
        if stage=='release':
            release_index+=1
            if release_index>=27:
                stage='retreat';retreat_index=0;retreat_palm=realized
        elif stage=='retreat':
            retreat_index+=1
            if retreat_index>=25:stage='post';post_index=0
        elif stage=='post':
            post_index+=1
            if post_index>=15:break
    command_array=np.asarray(commands,dtype=np.float32);timestamp_array=np.asarray(times,dtype=np.int64)
    if not np.array_equal(command_array[:ANCHOR+1],source_commands[:ANCHOR+1]) or not np.array_equal(timestamp_array[:ANCHOR+1],plan['timestamps'][:ANCHOR+1]):
        raise RuntimeError('source prefix changed')
    details={'setting':setting,'stage_at_end':stage,'release_start_frame':start_release,
             'frames':len(command_array),'duration_s':float((times[-1]-times[0])*1e-6),
             'align_height_m':align_z,'anchor_observed_object':pose_dict(*geometry['observed_object']),
             'aborted':abort,'generation_trace':records,
             'initial_palm_command_step_m':initial_palm_command_step.tolist(),
             'command_margins':command_margins(command_array,timestamp_array,lower,upper,velocity),
             'anchor_digit_force_n':anchor_force.tolist(),'target_digit_force_n':target_force.tolist(),
             'max_contact_residual_rad':contact_residual.tolist(),
             'closure_frozen_frame':closure_frozen_frame,'first_support_frame':first_support_frame,
             'pre_support_triad_step_fraction':float(pre_support_triad_steps/pre_support_steps) if pre_support_steps else None,
             'pre_support_longest_no_triad_gap_s':max_pre_support_gap_steps*model.opt.timestep,
             'grip_lost_before_support':grip_lost_before_support}
    return command_array,timestamp_array,details


def replay(model,plan,track,names,commands,timestamps):
    if len(commands)!=len(timestamps) or not np.isfinite(commands).all() or np.any(np.diff(timestamps)<=0):raise ValueError('invalid extended command')
    data,addresses=prepared_data(model,names,commands)
    obj=model.body('object_2').id
    positions=[];rotations=[];frames=[]
    for frame,command in enumerate(commands):
        if frame:
            dt=float((timestamps[frame]-timestamps[frame-1])*1e-6)
            diagnostic=step_command(model,data,commands[frame-1],command,dt,obj)
        else:diagnostic={'support_fraction':0.,'robot_table_steps':0,'hand_step_fraction':0.,'digit_mean_force_n':[0.]*5,'max_speed_m_s':0.}
        positions.append(data.xpos[obj].copy());rotations.append(data.xmat[obj].reshape(3,3).copy());frames.append(diagnostic)
    positions=np.asarray(positions);rotations=np.asarray(rotations)
    data.ctrl[:]=commands[-1]
    settle_p=[];settle_r=[];settle_speed=[];settle_support=[];settle_hand=[];settle_table=[]
    for _ in range(round(2./model.opt.timestep)):
        mujoco.mj_step(model,data)
        support,table,hand=force_flags(model,data,obj)
        settle_p.append(data.xpos[obj].copy());settle_r.append(data.xmat[obj].reshape(3,3).copy())
        settle_speed.append(float(np.linalg.norm(data.cvel[obj,3:])))
        settle_support.append(support);settle_hand.append(hand);settle_table.append(table)
    sp=np.asarray(settle_p);sr=np.asarray(settle_r);ss=np.asarray(settle_speed)
    su=np.asarray(settle_support);sh=np.asarray(settle_hand);st=np.asarray(settle_table)
    col=list(map(int,track['object_ids'])).index(2)
    refp=track['translation_world'][-1,col];refr=track['rotation_world_object'][-1,col]
    last=round(1./model.opt.timestep)
    evidence={'final_xy_error_mm':float(np.linalg.norm(sp[-1,:2]-refp[:2])*1000),
        'final_z_error_mm':float(abs(sp[-1,2]-refp[2])*1000),
        'final_orientation_error_deg':float(np.rad2deg(Rotation.from_matrix(refr@sr[-1].T).magnitude())),
        'final_upright_tilt_deg':float(np.rad2deg(np.arccos(np.clip(sr[-1,2,2],-1,1)))),
        'last_1s_table_support_fraction':float(su[-last:].mean()),
        'last_1s_hand_force_steps':int(sh[-last:].sum()),
        'last_1s_robot_table_steps':int(st[-last:].sum()),
        'last_1s_max_speed_m_s':float(ss[-last:].max()),
        'last_1s_peak_position_drift_mm':float(np.max(np.linalg.norm(sp[-last:]-sp[-last],axis=1))*1000),
        'last_1s_peak_rotation_drift_deg':float(np.rad2deg(np.max(Rotation.from_matrix(sr[-last:]@sr[-last].T).magnitude())))}
    evidence['placement_diagnostic_pass']=settled_placement_gate(evidence)
    return evidence,{'object_positions':positions,'object_rotations':rotations,
        'frame_support_fraction':np.asarray([x['support_fraction'] for x in frames]),
        'frame_robot_table_steps':np.asarray([x['robot_table_steps'] for x in frames]),
        'frame_hand_step_fraction':np.asarray([x['hand_step_fraction'] for x in frames]),
        'frame_digit_mean_force_n':np.asarray([x['digit_mean_force_n'] for x in frames]),
        'settle_object_positions':sp,'settle_object_rotations':sr,'settle_speed_m_s':ss,
        'settle_support_steps':su,'settle_hand_steps':sh,'settle_robot_table_steps':st}


def audit_extended(model,names,commands,timestamps,velocity):
    """Read-only physical audit of the exported command, including bystanders."""
    data,addresses=prepared_data(model,names,commands)
    obj=model.body('object_2').id
    others=[model.body(f'object_{i}').id for i in (1,3,4)]
    others_start=np.array([data.xpos[i].copy() for i in others])
    max_neighbor=0.;robot_other_steps=0;robot_table_steps=0
    min_target_distance=0.;saturated_steps=0;actual=[]
    before_support_steps=before_support_triad=0;no_grip_gap=longest_no_grip_gap=0
    first_support_frame=None;first_support_speed=None
    for frame,command in enumerate(commands):
        steps=(round((timestamps[frame]-timestamps[frame-1])*1e-6/model.opt.timestep) if frame else 0)
        previous=commands[max(frame-1,0)]
        for step in range(steps):
            data.ctrl[:]=previous+(command-previous)*(step+1)/steps
            mujoco.mj_step(model,data)
            if frame>ANCHOR and first_support_frame is None:
                support,_,_=force_flags(model,data,obj)
                if support:
                    first_support_frame=frame
                    first_support_speed=float(np.linalg.norm(data.cvel[obj,3:]))
                else:
                    before_support_steps+=1
                    forces=digit_forces(model,data,obj)
                    triad=bool(forces[0]>.05 and np.sum(forces[1:]>.05)>=2)
                    before_support_triad+=triad
                    no_grip_gap=0 if triad else no_grip_gap+1
                    longest_no_grip_gap=max(longest_no_grip_gap,no_grip_gap)
            displacement=np.linalg.norm(np.array([data.xpos[i] for i in others])-others_start,axis=1)
            max_neighbor=max(max_neighbor,float(displacement.max()))
            low=model.actuator_forcerange[:,0];high=model.actuator_forcerange[:,1]
            saturated_steps+=bool(np.any(data.actuator_force<=low+1e-6) or np.any(data.actuator_force>=high-1e-6))
            robot_other=False;robot_table=False
            for ci,c in enumerate(data.contact):
                b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
                if obj in (b1,b2):min_target_distance=min(min_target_distance,float(c.dist))
                force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                if force[0]<=1e-6:continue
                n1,n2=model.body(b1).name,model.body(b2).name
                if (b1 in others or b2 in others) and (n1.startswith('rh_') or n2.startswith('rh_') or 'link' in n1 or 'link' in n2):robot_other=True
                if 'world' in (n1,n2) and (n1.startswith('rh_') or n2.startswith('rh_') or 'link' in n1 or 'link' in n2):robot_table=True
            robot_other_steps+=robot_other;robot_table_steps+=robot_table
        actual.append(data.qpos[addresses].copy())
    actual=np.asarray(actual)
    speed=np.abs(np.diff(actual,axis=0)/(np.diff(timestamps)[:,None]*1e-6))
    margin=float(np.min(velocity-speed))
    return {'max_neighbor_displacement_mm':max_neighbor*1000,
        'neighbor_5mm_gate_pass':bool(max_neighbor<.005),
        'robot_other_object_contact_steps':robot_other_steps,
        'robot_table_contact_steps':robot_table_steps,
        'min_target_contact_distance_mm':min_target_distance*1000,
        'effort_saturation_steps':saturated_steps,
        'sampled_actual_joint_velocity_margin_rad_s':margin,
        'first_support_after_anchor_frame':first_support_frame,
        'first_support_linear_speed_m_s':first_support_speed,
        'pre_support_triad_step_fraction':float(before_support_triad/before_support_steps) if before_support_steps else 0.,
        'pre_support_longest_no_triad_gap_s':longest_no_grip_gap*model.opt.timestep,
        'controlled_touchdown_diagnostic_pass':bool(first_support_frame is not None and before_support_steps>0 and
            controlled_touchdown_gate(before_support_triad/before_support_steps,
                longest_no_grip_gap*model.opt.timestep,first_support_speed,
                robot_table_steps,robot_other_steps))}


def export(path,commands,timestamps,names,h,setting,details):
    np.savez_compressed(path,schema='shadow_feedback_placement_commands_v1',commands=commands,
        timestamps=timestamps,joint_names=np.asarray(names),plan_sha256=h['plan'],
        track_sha256=h['track'],urdf_sha256=h['urdf'],mesh_hashes_json=json.dumps(h['meshes'],sort_keys=True),
        input_sha256=h['input'],anchor_frame=ANCHOR,setting_json=json.dumps(setting,sort_keys=True),
        release_start_frame=-1 if details['release_start_frame'] is None else details['release_start_frame'])


def load_export(path,names,h,source_commands,source_timestamps,lower,upper,velocity):
    with np.load(path,allow_pickle=False) as z:
        expected={'schema':'shadow_feedback_placement_commands_v1','plan_sha256':h['plan'],
            'track_sha256':h['track'],'urdf_sha256':h['urdf'],
            'mesh_hashes_json':json.dumps(h['meshes'],sort_keys=True),'input_sha256':h['input']}
        for key,value in expected.items():
            if key not in z or str(z[key])!=value:raise ValueError(f'provenance mismatch: {key}')
        if list(map(str,z['joint_names']))!=list(names) or int(z['anchor_frame'])!=ANCHOR:raise ValueError('names or anchor mismatch')
        setting=json.loads(str(z['setting_json']))
        if setting not in SETTINGS and setting not in CONTACT_SETTINGS:raise ValueError('setting mismatch')
        q=z['commands'];ts=z['timestamps']
        if q.ndim!=2 or q.shape[1]!=len(names) or q.dtype!=source_commands.dtype or len(q)!=len(ts) or len(q)>MAX_FRAMES or len(q)<ANCHOR+1 or ts.dtype.kind not in 'iu' or not np.isfinite(q).all():raise ValueError('command shape or dtype mismatch')
        if not np.array_equal(q[:ANCHOR+1],source_commands[:ANCHOR+1]) or not np.array_equal(ts[:ANCHOR+1],source_timestamps[:ANCHOR+1]):raise ValueError('prefix mismatch')
        nominal=int(np.median(np.diff(source_timestamps)))
        expected_deltas=np.full(len(ts)-ANCHOR-1,nominal,dtype=np.int64)
        release=int(z['release_start_frame'])
        if release>=0:
            if release<ANCHOR+1 or release+27>len(ts):raise ValueError('release index mismatch')
            for j in range(27):expected_deltas[release-ANCHOR-1+j]=source_timestamps[125+j]-source_timestamps[124+j]
        if not np.array_equal(np.diff(ts[ANCHOR:]),expected_deltas):raise ValueError('extended timestamps mismatch')
        pm,vm=command_margins(q,ts,lower,upper,velocity)
        if pm< -1e-6 or vm< -1e-6:raise ValueError('exported limits mismatch')
    return q,ts,setting,release


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/app/rl-output/shadow_feedback_placement_20261026')
    parser.add_argument('--replay',help='independently replay a saved command (no oracle/controller)')
    parser.add_argument('--contact-followup',action='store_true',help='run the separately declared two-setting force-maintenance follow-up')
    args=parser.parse_args();out=Path(args.output)
    if any((out/n).exists() for n in ('manifest.json','ledger.jsonl','replay_report.json')):raise RuntimeError('existing run; use fresh output')
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source();h=hashes()
    input_commands=load_input(plan,names,baseline,h)
    model,table_z,xml=make_scene(plan,track,names)
    if args.replay:
        commands,timestamps,setting,release=load_export(Path(args.replay),names,h,input_commands,plan['timestamps'],lower,upper,velocity)
        evidence,trajectory=replay(model,plan,track,names,commands,timestamps)
        physical_audit=audit_extended(model,names,commands,timestamps,velocity)
        report={'setting':setting,'release_start_frame':release,'settle':evidence,'physical_audit':physical_audit,
            'command_sha256':sha(Path(args.replay)),'source_hashes':h}
        write_json(out/'replay_report.json',report)
        np.savez_compressed(out/'replay_trajectory.npz',**trajectory)
        print(json.dumps(report,indent=2),flush=True);return
    protocol=CONTACT_PROTOCOL if args.contact_followup else PROTOCOL
    active_settings=CONTACT_SETTINGS if args.contact_followup else SETTINGS
    write_json(out/'protocol.json',protocol)
    (out/'scene.xml').write_text(xml)
    manifest={'schema':'shadow_feedback_placement_pilot_v1','protocol':protocol,
        'source_hashes':h,'physics':'unchanged make_scene','accepted':False}
    write_json(out/'manifest.json',manifest)
    source_metrics,source_trajectory=rollout(model,plan,track,names,input_commands,table_z,full=True)
    kin=build_kinematics({**plan,'shadow_q':input_commands[:,6:]})
    geometry=setup_geometry(kin,plan,track,source_trajectory,input_commands,table_z)
    write_json(out/'geometry.json',{'table_top_z_m':table_z,'mesh_supported_center_z_m':geometry['support_center_z_m'],
        'reference_rotation_world':geometry['reference_rotation'].tolist(),
        'reference_xy_m':geometry['target_xy'].tolist(),'source_frame90_object_to_palm':pose_dict(*geometry['object_to_palm'])})
    write_json(out/'source_control.json',source_metrics)
    rows=[]
    for i,setting in enumerate(active_settings):
        cmd,ts,details=generate(model,plan,track,names,input_commands,lower,upper,velocity,setting,geometry)
        trace=details.pop('generation_trace')
        write_json(out/f'candidate_{i}_generation.json',{'details':details,'frames':trace})
        row={'index':i,'setting':setting,'generation':details}
        if details['aborted'] is None:
            path=out/f'candidate_{i}_commands.npz';export(path,cmd,ts,names,h,setting,details)
            evidence,trajectory=replay(model,plan,track,names,cmd,ts)
            physical_audit=audit_extended(model,names,cmd,ts,velocity)
            np.savez_compressed(out/f'candidate_{i}_trajectory.npz',**trajectory)
            row.update(command_sha256=sha(path),settle=evidence,physical_audit=physical_audit,
                released=details['release_start_frame'] is not None,
                extended_placement_pass=bool(details['release_start_frame'] is not None and evidence['placement_diagnostic_pass']),
                controlled_touchdown_pass=bool(details['release_start_frame'] is not None and
                    evidence['placement_diagnostic_pass'] and physical_audit['controlled_touchdown_diagnostic_pass']))
        rows.append(row)
        with (out/'ledger.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'index':i,'frames':len(cmd),'release_frame':details['release_start_frame'],
            'aborted':details['aborted'],'placement_pass':row.get('extended_placement_pass'),
            'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    eligible=[x for x in rows if x.get('command_sha256')]
    if eligible:
        winner=min(eligible,key=lambda x:(not x['controlled_touchdown_pass'],not x['extended_placement_pass'],not x['released'],
            x['settle']['final_orientation_error_deg']+x['settle']['final_upright_tilt_deg']+
            x['settle']['final_xy_error_mm']+x['settle']['final_z_error_mm'],x['index']))
        idx=winner['index'];src=out/f'candidate_{idx}_commands.npz';(out/'final_commands.npz').write_bytes(src.read_bytes())
        report={'schema':'shadow_feedback_placement_report_v1','selected_index':idx,'selected':winner,
            'source_control':source_metrics,'original_task_gate_only_for_original_190_frame_source':source_metrics['target_task_success'],
            'extended_placement_diagnostic_pass':bool(winner['extended_placement_pass']),
            'controlled_touchdown_diagnostic_pass':bool(winner['controlled_touchdown_pass']),
            'accepted':False,
            'acceptance_note':'extended placement diagnostic passes; original190 task phases are not directly comparable and full-scene gate requires separate audit',
            'selected_command_sha256':sha(out/'final_commands.npz')}
        write_json(out/'report.json',report)
        manifest.update(selected_index=idx,accepted=report['accepted'],selected_command_sha256=report['selected_command_sha256'])
    else:manifest['failure']='both settings rejected before an eligible replay'
    manifest.update(total_wall_s=time.monotonic()-start,evaluated_count=len(rows))
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'selected_index':manifest.get('selected_index'),'accepted':manifest['accepted'],
        'wall_s':manifest['total_wall_s']},indent=2),flush=True)

if __name__=='__main__':main()
