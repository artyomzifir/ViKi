"""Object-frame residual and IK acceptance invariants."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.shadow_palm_controls import (
    object_frame_palm_target, palm_gate, solve_arm_pose,
    validate_ik_errors, zero_palm_commands,
)


def test_zero_residual_preserves_float32_commands_bitwise():
    original=np.random.default_rng(7).normal(size=(190,28)).astype(np.float32)
    snapshot=original.copy()
    copied=zero_palm_commands(original)
    assert copied.dtype==np.float32 and copied is not original
    assert np.array_equal(copied,snapshot)
    assert np.array_equal(original,snapshot)
    with pytest.raises(ValueError):zero_palm_commands(original.astype(np.float64))


def test_object_translation_basis_and_palm_center_rotation_pivot():
    object_r=Rotation.from_euler('z',90,degrees=True).as_matrix()
    palm_p=np.array([.1,.2,.3]);palm_r=np.eye(3)
    delta=np.array([.01,0,0,np.deg2rad(8),0,0])
    p,r=object_frame_palm_target(palm_p,palm_r,object_r,delta,1.)
    assert p==pytest.approx([.1,.21,.3],abs=1e-12)
    expected=object_r@Rotation.from_rotvec(delta[3:]).as_matrix()@object_r.T
    assert r==pytest.approx(expected,abs=1e-12)
    p0,r0=object_frame_palm_target(palm_p,palm_r,object_r,np.zeros(6),1.)
    assert p0==pytest.approx(palm_p) and r0==pytest.approx(palm_r)
    pure_rotation=delta.copy();pure_rotation[:3]=0
    center,_=object_frame_palm_target(palm_p,palm_r,object_r,pure_rotation,1.)
    assert center==pytest.approx(palm_p)


def test_gate_reaches_full_correction_at_first_contact_and_releases():
    gate=palm_gate()
    assert np.all(gate[:11]==0)
    assert gate[23]==1 and gate[42]==1 and gate[110]==1
    assert np.all(gate[125:]==0)
    assert np.max(np.abs(np.diff(gate)))<.13
    with pytest.raises(ValueError):palm_gate(100)


class LinearArm:
    q_min=np.full(6,-1.)
    q_max=np.full(6,1.)
    def set_frame(self,frame):self.frame=frame
    def pose_jacobian(self,q):
        p=q[:3].copy();r=Rotation.from_rotvec(q[3:]).as_matrix()
        return p,r,np.c_[np.eye(3),np.zeros((3,3))],np.c_[np.zeros((3,3)),np.eye(3)]
    def pose(self,q):return q[:3].copy(),Rotation.from_rotvec(q[3:]).as_matrix()


def test_arm_ik_achieves_small_six_dimensional_target_and_checks_error():
    fake=LinearArm()
    target_p=np.array([.01,-.005,.012])
    target_r=Rotation.from_rotvec([.04,-.03,.06]).as_matrix()
    q,pos,ori=solve_arm_pose(fake,17,np.zeros(6),target_p,target_r)
    assert fake.frame==17
    assert np.linalg.norm(q[:3]-target_p)<.0015
    assert validate_ik_errors([pos],[ori])
    assert not validate_ik_errors([.002],[0])
    assert not validate_ik_errors([0],[np.deg2rad(2)])
    with pytest.raises(ValueError):validate_ik_errors([np.nan],[0])


def test_sampled_actual_joint_velocity_margin_detects_limit_violation():
    from experiments.shadow_palm_controls import sampled_joint_velocity_margin
    q=np.array([[0.],[.1],[.2]])
    ts=np.array([0,100_000,200_000])
    assert sampled_joint_velocity_margin(q,ts,[2.])==pytest.approx(1.)
    assert sampled_joint_velocity_margin(q,ts,[.5])==pytest.approx(-.5)
    with pytest.raises(ValueError):sampled_joint_velocity_margin(q,[0,100_000,100_000],[2.])
