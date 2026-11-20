"""Bounded open-loop open-and-retreat ablation after the multifinger grasp.

All objects remain free. No support feedback enters control; anchor frames and
release schedules are predeclared and every saved command is checked before replay.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.hocap_shadow_polisher import source,make_scene,rollout,sha,PLAN,TRACK,URDF,EP
from experiments.hocap_shadow_palm_polisher import build_kinematics
from experiments.hocap_shadow_multifinger_polisher import full_digit_trace,kinematic_margins
from experiments.shadow_polisher_controls import command_margins,validate_command_export
from experiments.shadow_palm_controls import solve_arm_pose,validate_ik_errors
from viki.retarget.frames import base_rotation


def smooth(frames,start,end):
    u=np.clip((frames-start)/(end-start),0.,1.)
    return u*u*(3-2*u)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='/app/rl-output/shadow_multifinger_polisher_20261025_v2/final_commands.npz')
    parser.add_argument('--output',default='/app/rl-output/shadow_placement_polisher_20261025')
    parser.add_argument('--finger-policy',choices=('straight','source'),default='straight')
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    plan,track,names,baseline,lower,upper,velocity=source()
    hashes={'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)}}
    expected={'schema':'shadow_polisher_commands_v1','plan_sha256':hashes['plan'],'track_sha256':hashes['track'],'urdf_sha256':hashes['urdf'],'mesh_hashes_json':json.dumps(hashes['meshes'],sort_keys=True)}
    with np.load(args.input,allow_pickle=False) as saved:
        source_commands,timestamps,factor=validate_command_export(saved,expected,names,plan['timestamps'],baseline.shape,baseline.dtype)
    if factor!=1.:raise ValueError('this placement pilot keeps the original timestamps')
    protocols=[{'anchor':125,'open_end':145,'retreat_start':145,'retreat_end':170},
               {'anchor':130,'open_end':150,'retreat_start':150,'retreat_end':175},
               {'anchor':125,'open_end':145,'retreat_start':125,'retreat_end':150}]
    if args.finger_policy=='source':
        protocols=[{'anchor':anchor,'open_end':151,'retreat_start':151,'retreat_end':175} for anchor in (120,125,130)]
    declaration={'schema':'shadow_placement_pilot_v1','input_sha256':sha(Path(args.input)),'source_hashes':hashes,'schedules':protocols,'retreat_world_z_m':.05,'opening':'FF/MF/RF/LF J1 J2 J3 -> 0 rad; THJ4 -> .3 rad; other finger joints frozen at anchor','selection':'target_task_success first, then old frozen full-rollout score; all thresholds and physics unchanged','limits':'all saved frames checked against joint/command segment velocity on saved timestamps and existing screened floor/self-collision constraints; no closed-loop support forcing'}
    declaration['finger_policy']=args.finger_policy
    if args.finger_policy=='source':declaration['opening']='original input finger trajectory preserved bitwise; only arm stop and retreat change'
    (out/'protocol.json').write_text(json.dumps(declaration,indent=2)+'\n')
    model,table_z,xml=make_scene(plan,track,names);(out/'scene.xml').write_text(xml)
    base_metrics=rollout(model,plan,track,names,source_commands,table_z)
    (out/'input_metrics.json').write_text(json.dumps(base_metrics,indent=2)+'\n')
    time=np.arange(190,dtype=float);ledger=[];valid=[]
    for index,schedule in enumerate(protocols):
        anchor=schedule['anchor'];commands=source_commands.copy();opened=commands[anchor,6:].copy()
        for digit in ('FF','MF','RF','LF'):
            for joint in (1,2,3):opened[names.index(f'rh_{digit}J{joint}')-6]=0.
        opened[names.index('rh_THJ4')-6]=.3
        w=smooth(time,anchor,schedule['open_end'])
        if args.finger_policy=='straight':
            commands[anchor:,6:]=((1-w[anchor:,None])*source_commands[anchor,6:]+w[anchor:,None]*opened).astype(np.float32)
        commands[anchor:,:6]=source_commands[anchor,:6]
        candidate_plan=dict(plan);candidate_plan['shadow_q']=commands[:,6:]
        kin=build_kinematics(candidate_plan);kin.set_frame(anchor);p0,r0=kin.pose(commands[anchor,:6])
        up=base_rotation(plan['base_rpy_deg']).T@np.array([0.,0.,.05]);retreat=smooth(time,schedule['retreat_start'],schedule['retreat_end'])
        errors=[];angles=[];failure=None
        for frame in range(anchor+1,190):
            if retreat[frame]==0:continue
            target=p0+up*retreat[frame]
            q,_,_=solve_arm_pose(kin,frame,commands[frame-1,:6],target,r0)
            commands[frame,:6]=q.astype(np.float32);kin.set_frame(frame);ap,ar=kin.pose(commands[frame,:6])
            errors.append(float(np.linalg.norm(ap-target)));angles.append(float(Rotation.from_matrix(r0@ar.T).magnitude()))
            if not validate_ik_errors(errors[-1:],angles[-1:]):failure={'reason':'IK','frame':frame};break
        pm,vm=command_margins(commands,timestamps,lower,upper,velocity)
        row={'index':index,'schedule':schedule,'position_margin_rad':pm,'velocity_margin_rad_s':vm,'max_ik_position_mm':max(errors,default=0.)*1000,'max_ik_orientation_deg':float(np.rad2deg(max(angles,default=0.)))}
        if failure or min(pm,vm)<-1e-6:
            row.update(status='rejected_ik_or_limits',failure=failure)
        else:
            floor,self_margin,bad_frame=kinematic_margins(kin,names,commands,range(190))
            row.update(floor_margin_m=floor,self_collision_margin_m=self_margin)
            if bad_frame is not None:row.update(status='rejected_geometry',frame=bad_frame)
            else:
                assert np.array_equal(commands[:anchor],source_commands[:anchor])
                file=out/f'candidate_{index}_commands.npz'
                np.savez_compressed(file,**expected,commands=commands,timestamps=timestamps,joint_names=np.asarray(names),carry_timing_factor=1.,placement_schedule_json=json.dumps(schedule),finger_policy=args.finger_policy,input_sha256=declaration['input_sha256'])
                metrics,trajectory=rollout(model,plan,track,names,commands,table_z,full=True);digits,trace=full_digit_trace(model,plan,names,commands)
                np.savez_compressed(out/f'candidate_{index}_trajectory.npz',**trajectory,**trace)
                row.update(status='evaluated',command_file=file.name,command_sha256=sha(file),metrics=metrics,digits=digits)
                valid.append(row)
        ledger.append(row);(out/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(row),flush=True)
    if valid:
        best=min(valid,key=lambda r:(not r['metrics']['target_task_success'],r['metrics']['score'],r['index']))
        (out/'selected_commands.npz').write_bytes((out/best['command_file']).read_bytes())
        result={'selected_index':best['index'],'selected_metrics':best['metrics'],'input_metrics':base_metrics,'improves_input_score':best['metrics']['score']<base_metrics['score'],'accepted':bool(best['metrics']['target_task_success'] and best['metrics']['full_scene_success'])}
    else:result={'selected_index':None,'accepted':False,'reason':'all fixed placement candidates rejected'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
