"""Geometry-only finger proximity audit against the original HO-Cap annotation.

No force/contact truth is inferred: capsule radii are the fitted hand's proxy,
object distance uses MuJoCo's convex hull of the aligned official object mesh.
"""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation

EP=Path('/app/data/datasets/hocap_demo_object_v4/g10_object2_190')
OUT=Path('/app/rl-output/source_finger_audit_20261026')
DIGITS=('thumb','index','middle','ring','little')


def main():
    OUT.mkdir(exist_ok=True)
    clean_path=EP/'cln.npz';gt_path=EP/'gt/joints3d_world.npz';pose_path=EP/'gt/object_poses_world.npz';track_path=EP/'intermediates/segmentation/hocap-oracle-v1/object_models.npz';mesh=track_path.parent/'meshes/object_2.obj';align_path=mesh.parent/'alignment.json'
    c=np.load(clean_path,allow_pickle=False);g=np.load(gt_path,allow_pickle=False);gt=np.load(pose_path,allow_pickle=False);t=np.load(track_path,allow_pickle=False)
    assert np.array_equal(g['frame_index'],np.arange(190)) and np.array_equal(gt['frame_index'],np.arange(190))
    alignment=next(x for x in json.loads(align_path.read_text()) if x['id']==2)
    object_index=list(map(str,gt['object_ids'])).index('G10_2');track_index=list(map(int,t['object_ids'])).index(2)
    gt_world_track=np.asarray(gt['poses_world'][:,object_index],float)@np.linalg.inv(np.array(alignment['track_from_mesh']))
    gt_caps=np.stack([g['joints_world'][:,[j,j+1]] for j in (1,2,3,5,6,7,9,10,11,13,14,15,17,18,19)],axis=1)
    gt_valid=np.stack([g['valid'][:,[j,j+1]].all(axis=1) for j in (1,2,3,5,6,7,9,10,11,13,14,15,17,18,19)],axis=1)
    radii=c['hand_fit_capsule_radii'][1:].astype(float);assert len(radii)==15
    geoms=''.join(f'<geom name="bone_{i}" type="capsule" size=".01 .01" contype="0" conaffinity="0"/>' for i in range(15))
    xml=f'<mujoco><asset><mesh name="object" file="{mesh}"/></asset><worldbody><body name="object"><geom name="object" type="mesh" mesh="object" contype="0" conaffinity="0"/></body>{geoms}</worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);body=model.body('object').id;obj=model.geom('object').id;geomids=[model.geom(f'bone_{i}').id for i in range(15)]
    # Compiled world geoms initially share the world frame; disable that fast path
    # before assigning their per-frame capsule transforms.
    model.geom_sameframe[geomids]=0
    sources=[('annotated_joints_gt_object',gt_caps,gt_valid,gt_world_track[:,:3,3],gt_world_track[:,:3,:3]),('fitted_capsules_tracked_object',c['hand_fit_capsules'][:,1:],np.ones((190,15),bool),t['translation_world'][:,track_index],t['rotation_world_object'][:,track_index]),('fitted_capsules_gt_object',c['hand_fit_capsules'][:,1:],np.ones((190,15),bool),gt_world_track[:,:3,3],gt_world_track[:,:3,:3])]
    results=[];arrays={}
    for label,caps,valid,positions,rotations in sources:
        gaps=np.full((190,15),np.nan)
        for frame in range(190):
            if not gt['valid'][frame,object_index]:continue
            model.body_pos[body]=positions[frame];model.body_quat[body]=Rotation.from_matrix(rotations[frame]).as_quat(scalar_first=True)
            for j,geom in enumerate(geomids):
                ends=np.asarray(caps[frame,j],float);delta=ends[1]-ends[0];length=np.linalg.norm(delta)
                if not valid[frame,j] or not np.isfinite(ends).all() or length<1e-8:continue
                axis=delta/length;q=np.r_[1+axis[2],-axis[1],axis[0],0.]
                q=np.array([0.,1.,0.,0.]) if np.linalg.norm(q)<1e-8 else q/np.linalg.norm(q)
                model.geom_pos[geom]=ends.mean(axis=0);model.geom_quat[geom]=q;model.geom_size[geom,:2]=[radii[j],length/2];model.geom_rbound[geom]=radii[j]+length/2
            mujoco.mj_forward(model,data)
            for j,geom in enumerate(geomids):
                if valid[frame,j]:
                    if not np.allclose(data.geom_xpos[geom],caps[frame,j].mean(axis=0),atol=1e-7):raise RuntimeError('capsule world transform mismatch')
                    gaps[frame,j]=mujoco.mj_geomDistance(model,data,geom,obj,.5,None)
        arrays[label]=gaps
        per=gaps.reshape(190,5,3).min(axis=2)*1000;distal=gaps[:,2::3]*1000;rows=[]
        for j,digit in enumerate(DIGITS):
            values=per[42:126,j];dv=distal[42:126,j];finite=np.isfinite(values);values=values[finite]
            rows.append({'digit':digit,'valid_carry_frames':int(len(values)),'min_any_phalanx_gap_median_mm':float(np.median(values)),'gap_p10_p90_mm':np.percentile(values,[10,90]).tolist(),'distal_gap_median_mm':float(np.nanmedian(dv)),'fraction_gap_le_threshold_mm':{str(k):float(np.mean(values<=k)) for k in (0,3,5,10)},'proxy_penetration_below_minus10mm_fraction':float(np.mean(values< -10))})
        results.append({'source':label,'digits':rows,'sample_min_gap_mm':{str(k):per[k].tolist() for k in (42,60,90,120,145)}})
    delta_pos=np.linalg.norm(gt_world_track[:,:3,3]-t['translation_world'][:,track_index],axis=1)*1000
    delta_rot=np.rad2deg(Rotation.from_matrix(gt_world_track[:,:3,:3]@np.swapaxes(t['rotation_world_object'][:,track_index],1,2)).magnitude())
    inputs=[clean_path,gt_path,pose_path,track_path,mesh,align_path]
    report={'schema':'source_finger_proximity_audit_v1','digits':DIGITS,'carry_frames':[42,126],'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'radii_mm':(radii*1000).tolist(),'method':'minimum signed capsule-to-convex-hull gap over each digit three phalanges; GT joints use fitted radii; fitted capsules use stored radii; tracked vs GT objects compared independently','limits':['Proximity proxy only, not measured contact force.','Semi-automatic joint/object annotations and hand thickness approximation can disagree with true skin contact.','Negative gaps indicate proxy penetration, not pressure.','Occluded digits in RGB do not prove absent contact.'],'object_pose_alignment_difference':{'median_position_mm':float(np.median(delta_pos)),'max_position_mm':float(delta_pos.max()),'median_rotation_deg':float(np.median(delta_rot)),'max_rotation_deg':float(delta_rot.max())},'results':results}
    (OUT/'finger_proximity.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'finger_gaps.npz',**arrays)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
