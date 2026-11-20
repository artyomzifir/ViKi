import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.shadow_feedback_placement_controls import (
    bounded_step,orient_toward,palm_for_object_step,smooth_palm_command,release_ready)


def test_bounded_object_center_step_moves_palm_with_lever_arm():
    target=Rotation.from_euler('z',90,degrees=True).as_matrix()
    p,r=palm_for_object_step(np.zeros(3),np.eye(3),[.1,0,0],np.eye(3),
                              np.zeros(3),target,.002,np.deg2rad(90))
    np.testing.assert_allclose(p,[0,.1,0],atol=1e-12)
    np.testing.assert_allclose(r,target,atol=1e-12)
    assert np.linalg.norm(bounded_step([10,0,0],.002))==pytest.approx(.002)
    assert Rotation.from_matrix(orient_toward(np.eye(3),target,np.deg2rad(1))).magnitude()==pytest.approx(np.deg2rad(1))


def test_command_braking_is_acceleration_bounded():
    p,r,dp,dr=smooth_palm_command([0,0,0],np.eye(3),[0,0,0],np.eye(3),
        np.array([.004,0,0]),np.zeros(3),.006,np.deg2rad(2),.001,np.deg2rad(.5))
    np.testing.assert_allclose(p,[.003,0,0])
    np.testing.assert_allclose(dp,[.003,0,0])
    np.testing.assert_allclose(r,np.eye(3))
    np.testing.assert_allclose(dr,np.zeros(3))


def test_release_requires_supported_upright_slow_interval():
    assert release_ready(1.,7.9,.029,9)
    assert not release_ready(.5,0,0,30)
    assert not release_ready(1.,9,0,30)
    assert not release_ready(1.,0,.04,30)
    assert not release_ready(1.,0,0,8)
    with pytest.raises(ValueError):release_ready(np.nan,0,0,9)


def test_force_residual_bounded_and_named_digit_synergy():
    from experiments.shadow_feedback_placement_controls import force_residual_step,residual_finger_command
    q=force_residual_step([0.,0.,0.],[0.,1.,.5],[3.,2.,1.],.01,[.06,.12,.12])
    np.testing.assert_allclose(q,[.03,.01,.005])
    q=force_residual_step(q,[0.,0.,0.],[3.,2.,1.],1.,[.06,.12,.12])
    np.testing.assert_allclose(q,[.06,.12,.12])
    names=['rh_THJ4','rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_MFJ1','rh_MFJ2','rh_MFJ3']
    source=np.zeros(len(names),dtype=np.float32)
    out=residual_finger_command(source,names,[.06,.12,.08])
    np.testing.assert_allclose(out,[.06,.12,.084,.042,.08,.056,.028],atol=1e-8)
    assert np.array_equal(source,np.zeros(len(names)))
    with pytest.raises(ValueError):force_residual_step([0,0,0],[0,0,0],[1,1,1],.01,[0,1,1])


def test_controlled_touchdown_rejects_lucky_upright_drop():
    from experiments.shadow_feedback_placement_controls import controlled_touchdown_gate
    assert controlled_touchdown_gate(.98,.01,.04,0,0)
    assert not controlled_touchdown_gate(.64,.287,.694,0,0)
    assert not controlled_touchdown_gate(1.,0.,.12,0,0)
    assert not controlled_touchdown_gate(1.,0.,.03,1,0)
    with pytest.raises(ValueError):controlled_touchdown_gate(1.1,0.,.03,0,0)


def test_extended_export_guards_prefix_provenance_and_timing(tmp_path):
    pytest.importorskip('mujoco')
    from experiments.hocap_shadow_feedback_placement import export,load_export,SETTINGS
    names=[f'joint_{i}' for i in range(28)]
    original=np.zeros((190,28),dtype=np.float32)
    source_ts=np.arange(190,dtype=np.int64)*33333
    q=np.zeros((115,28),dtype=np.float32)
    ts=np.r_[source_ts[:111],source_ts[110]+np.arange(1,5)*33333]
    hashes={'plan':'p','track':'t','urdf':'u','meshes':{'2':'m'},'input':'i'}
    path=tmp_path/'command.npz'
    export(path,q,ts,names,hashes,SETTINGS[0],{'release_start_frame':None})
    lower=np.full(28,-1.);upper=np.full(28,1.);speed=np.full(28,2.)
    got,got_ts,_,release=load_export(path,names,hashes,original,source_ts,lower,upper,speed)
    assert np.array_equal(got,q) and np.array_equal(got_ts,ts) and release==-1
    with pytest.raises(ValueError,match='provenance'):
        load_export(path,names,{**hashes,'track':'tampered'},original,source_ts,lower,upper,speed)
    tampered=tmp_path/'bad_time.npz'
    with np.load(path,allow_pickle=False) as z:
        values={key:z[key] for key in z.files}
    values['timestamps']=ts.copy();values['timestamps'][-1]+=1
    np.savez_compressed(tampered,**values)
    with pytest.raises(ValueError,match='timestamps'):
        load_export(tampered,names,hashes,original,source_ts,lower,upper,speed)
