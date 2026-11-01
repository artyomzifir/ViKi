import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.shadow_object_pose_placement_controls import (
    compose, inverse, object_to_palm, support_center_height, smoothstep,
    corrected_object_and_palm, settled_placement_gate,
)


def test_relative_grasp_pose_round_trip():
    object_pose=(np.array([.12,.04,.2]),Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix())
    palm_pose=(np.array([.16,-.07,.23]),Rotation.from_euler('xyz',[-.1,.3,.2]).as_matrix())
    relative=object_to_palm(*object_pose,*palm_pose)
    p,r=compose(*object_pose,*relative)
    np.testing.assert_allclose(p,palm_pose[0],atol=1e-12)
    np.testing.assert_allclose(r,palm_pose[1],atol=1e-12)
    zero,identity=compose(*relative,*inverse(*relative))
    np.testing.assert_allclose(zero,np.zeros(3),atol=1e-12)
    np.testing.assert_allclose(identity,np.eye(3),atol=1e-12)


def test_object_center_rotation_moves_palm_at_fixed_lever_arm():
    quarter=Rotation.from_euler('z',90,degrees=True).as_matrix()
    obj_p,obj_r,palm_p,palm_r=corrected_object_and_palm(
        np.array([1.,2.,3.]),np.eye(3),np.array([.1,0,0]),np.eye(3),
        np.zeros(3),quarter,1.)
    np.testing.assert_allclose(obj_p,[1.,2.,3.])
    np.testing.assert_allclose(palm_p,[1.,2.1,3.],atol=1e-12)
    np.testing.assert_allclose(obj_r,quarter,atol=1e-12)
    np.testing.assert_allclose(palm_r,quarter,atol=1e-12)


def test_mesh_bottom_reaches_original_table_top():
    vertices=np.array([[-.01,0,-.075],[.02,0,.04],[0,.02,0]])
    r=Rotation.from_euler('y',12,degrees=True).as_matrix()
    table_z=-.001
    center=support_center_height(vertices,r,table_z,.005)
    bottom=np.min((vertices@r.T)[:,2])+center
    assert bottom==pytest.approx(table_z+.005)
    with pytest.raises(ValueError):support_center_height(vertices,r,table_z,-.001)


def test_gate_endpoints_and_input_rejection():
    assert smoothstep(90,90,125)==0
    assert smoothstep(125,90,125)==1
    assert 0<smoothstep(107,90,125)<1
    with pytest.raises(ValueError):smoothstep(90,125,90)
    with pytest.raises(ValueError):corrected_object_and_palm(
        np.zeros(3),np.eye(3),np.zeros(3),np.eye(3),np.zeros(3),np.eye(3),1.01)


def test_settle_gate_requires_sustained_support_release_upright_stability():
    good={'final_xy_error_mm':4.,'final_z_error_mm':2.,
          'final_orientation_error_deg':3.,'final_upright_tilt_deg':4.,
          'last_1s_table_support_fraction':1.,'last_1s_hand_force_steps':0,
          'last_1s_robot_table_steps':0,'last_1s_max_speed_m_s':.004,
          'last_1s_peak_position_drift_mm':1.,'last_1s_peak_rotation_drift_deg':1.}
    assert settled_placement_gate(good)
    for changed in ({'last_1s_table_support_fraction':.5},
                    {'last_1s_hand_force_steps':1},
                    {'final_upright_tilt_deg':94},
                    {'last_1s_peak_position_drift_mm':8},
                    {'last_1s_peak_rotation_drift_deg':8},
                    {'last_1s_max_speed_m_s':.05}):
        assert not settled_placement_gate({**good,**changed})
    with pytest.raises(ValueError):settled_placement_gate({**good,'final_xy_error_mm':np.nan})


def test_frozen_candidate_families_and_export_provenance(tmp_path):
    pytest.importorskip('mujoco')
    from experiments.hocap_shadow_object_pose_placement import settings, export
    from experiments.shadow_polisher_controls import validate_command_export
    fixed=settings();followup=settings('exploratory_clearance8')
    assert len(fixed)==len(followup)==8
    assert {x['support_clearance_m'] for x in fixed}=={0.,.005}
    assert {x['support_clearance_m'] for x in followup}=={.01,.015}
    assert not set(map(str,fixed)) & set(map(str,followup))
    command=np.zeros((190,28),dtype=np.float32)
    timestamps=np.arange(190,dtype=np.int64)*33333
    plan={'timestamps':timestamps}
    names=[f'joint_{i}' for i in range(28)]
    hashes={'plan':'p','track':'t','urdf':'u','meshes':{'2':'m'},'input':'i'}
    path=tmp_path/'command.npz'
    export(path,command,plan,names,hashes,followup[0],0)
    expected={'plan_sha256':'p','track_sha256':'t','urdf_sha256':'u',
              'mesh_hashes_json':'{"2": "m"}','source_grasp_sha256':'i'}
    with np.load(path,allow_pickle=False) as saved:
        copied,ts,factor=validate_command_export(saved,expected,names,timestamps,command.shape,command.dtype)
        assert np.array_equal(copied,command)
        assert np.array_equal(ts,timestamps) and factor==1
        with pytest.raises(ValueError,match='provenance'):
            validate_command_export(saved,{**expected,'track_sha256':'tampered'},names,timestamps,command.shape,command.dtype)
        with pytest.raises(ValueError,match='names'):
            validate_command_export(saved,expected,names[::-1],timestamps,command.shape,command.dtype)
