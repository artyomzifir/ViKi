import numpy as np
import pytest

from experiments.shadow_multifinger_controls import BOUNDS, active_digit_forces, finger_gate, multifinger_commands, static_hold_metrics


NAMES = [f'arm{i}' for i in range(6)] + [
    f'rh_{digit}J{joint}' for digit,joints in [('FF',(1,2,3,4)),('MF',(1,2,3,4)),
        ('RF',(1,2,3,4)),('LF',(1,2,3,4,5)),('TH',(1,2,3,4,5))] for joint in joints]


def source():
    return np.zeros((190,28),dtype=np.float32)


def test_identity_is_bitwise_and_source_is_untouched():
    original=source();original[30,8]=np.float32(.12345679)
    before=original.copy()
    result=multifinger_commands(original,NAMES,np.zeros(8))
    assert result.dtype==np.float32
    assert np.array_equal(result,original)
    assert result is not original
    assert np.array_equal(original,before)


def test_acquisition_gate_and_digit_semantics():
    original=source();p=np.zeros(8);p[0]=.2;p[4]=.08;p[6]=.12;p[7]=-.04
    out=multifinger_commands(original,NAMES,p)
    g=finger_gate()
    assert g[10]==0 and g[30]==1 and g[125]==1 and g[151]==0
    assert np.array_equal(out[:11],original[:11])
    assert np.array_equal(out[151:],original[151:])
    assert np.array_equal(out[:,:6],original[:,:6])
    at=30
    assert np.isclose(out[at,NAMES.index('rh_MFJ1')],.2)
    assert np.isclose(out[at,NAMES.index('rh_MFJ2')],.14)
    assert np.isclose(out[at,NAMES.index('rh_MFJ3')],.07)
    assert np.isclose(out[at,NAMES.index('rh_RFJ4')],.08)
    assert np.isclose(out[at,NAMES.index('rh_LFJ5')],.12)
    assert np.isclose(out[at,NAMES.index('rh_THJ2')],-.04)
    assert out[23,NAMES.index('rh_MFJ1')]>0


def test_rejects_out_of_bounds_and_bad_source():
    p=np.zeros(8);p[0]=BOUNDS[0,1]+1e-4
    with pytest.raises(ValueError):multifinger_commands(source(),NAMES,p)
    with pytest.raises(ValueError):multifinger_commands(source().astype(float),NAMES,np.zeros(8))
    with pytest.raises(ValueError):multifinger_commands(source(),NAMES[:-1]+[NAMES[0]],np.zeros(8))


def evidence(force_value=.1,slip_mm=0.,drop_mm=0.,last_drift_mm=0.,turn_deg=0.):
    rel=np.zeros((4001,3));rel[-1,0]=slip_mm/1000
    rel[2000,0]=(slip_mm-last_drift_mm)/1000
    obj=np.zeros((4001,3));obj[:,2]=.2;obj[-1,2]=.2-drop_mm/1000
    rotations=np.repeat(np.eye(3)[None],4001,axis=0)
    angle=np.deg2rad(turn_deg)
    rotations[-1]=np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],[0,0,1]])
    forces=np.zeros((4000,5));forces[:,[0,1,2]]=force_value
    return rel,rotations,obj,forces,30.,np.zeros(4000,dtype=bool),np.zeros(4000,dtype=bool)


def test_static_gate_requires_sustained_distinct_digits_and_pose_stability():
    args=evidence()
    assert static_hold_metrics(*args)['static_pass']
    lost=tuple(x.copy() if hasattr(x,'copy') else x for x in args);lost[3][-2000:,2]=0
    assert not static_hold_metrics(*lost)['static_pass']
    assert not static_hold_metrics(*evidence(slip_mm=11))['static_pass']
    assert not static_hold_metrics(*evidence(drop_mm=11))['static_pass']
    assert not static_hold_metrics(*evidence(slip_mm=8,last_drift_mm=6))['static_pass']
    assert not static_hold_metrics(*evidence(turn_deg=11))['static_pass']
    pinned=tuple(x.copy() if hasattr(x,'copy') else x for x in args)
    pinned[5][0]=True
    assert not static_hold_metrics(*pinned)['static_pass']
    not_lifted=list(args);not_lifted[4]=10.
    assert not static_hold_metrics(*not_lifted)['static_pass']


def test_contact_threshold_is_per_digit_at_same_step():
    force=np.array([[.1,.1,.1,0,0],[.1,.1,.049,.1,0],[.1,.1,0,0,0]])
    active,triad=active_digit_forces(force)
    assert triad.tolist()==[True,True,False]
    assert not active[1,2]


def test_collision_screen_receives_candidate_passive_fingers():
    pytest.importorskip('mujoco')
    from experiments.hocap_shadow_multifinger_polisher import set_passive_fingers

    class Model:
        idx_qs = np.arange(28)
        def getJointId(self, name):
            return NAMES.index(name)

    class Kin:
        model = Model()
        _passive_positions = [(i,np.full(190,-1.)) for i in range(6,28)]
        _collision_configuration_cache = object()

    commands=source()
    commands[:,NAMES.index('rh_MFJ1')]=np.linspace(0,.2,190,dtype=np.float32)
    kin=Kin()
    set_passive_fingers(kin,NAMES,commands)
    by_index=dict(kin._passive_positions)
    assert np.array_equal(by_index[NAMES.index('rh_MFJ1')],commands[:,NAMES.index('rh_MFJ1')])
    assert all(not np.any(values==-1) for values in by_index.values())
    assert kin._collision_configuration_cache is None
