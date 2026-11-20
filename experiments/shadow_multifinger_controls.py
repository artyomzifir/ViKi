"""Pure bounded finger-command family for the frozen G10_2 Shadow pilot."""

import numpy as np
from scipy.spatial.transform import Rotation

# MF/RF/LF flexion, MF/RF/LF abduction, little cup, thumb opposition.
BOUNDS = np.array([[-.08,.24],[-.08,.30],[-.08,.32],
                   [-.20,0.],[-.03,.16],[-.10,.16],
                   [0.,.18],[-.12,.12]], dtype=float)
DIGITS = ('th','ff','mf','rf','lf')
FORCE_THRESHOLD_N = .05


def finger_gate(n_frames=190):
    if n_frames != 190:
        raise ValueError('multifinger pilot requires the frozen 190-frame plan')
    x=np.arange(n_frames,dtype=float)
    def smooth(a,b):
        u=np.clip((x-a)/(b-a),0.,1.)
        return u*u*(3-2*u)
    return smooth(10,30)*(1-smooth(125,151))


def multifinger_commands(palm_commands, names, residual):
    """Copy float32 commands; add independent acquisition-stage digit residuals."""
    source=np.asarray(palm_commands)
    residual=np.asarray(residual,dtype=float)
    if (source.shape!=(190,28) or source.dtype!=np.float32 or not np.isfinite(source).all()
            or len(names)!=28 or len(set(names))!=28 or residual.shape!=(8,)
            or not np.isfinite(residual).all() or np.any(residual<BOUNDS[:,0])
            or np.any(residual>BOUNDS[:,1])):
        raise ValueError('invalid source, joint names, or bounded residual')
    result=source.copy()
    if np.all(residual==0):
        return result
    gate=finger_gate()
    lookup={name:i for i,name in enumerate(names)}
    for axis,digit in enumerate(('MF','RF','LF')):
        for joint,weight in ((1,1.),(2,.7),(3,.35)):
            result[:,lookup[f'rh_{digit}J{joint}']] += (gate*residual[axis]*weight).astype(np.float32)
    for axis,digit in enumerate(('MF','RF','LF'),start=3):
        result[:,lookup[f'rh_{digit}J4']] += (gate*residual[axis]).astype(np.float32)
    result[:,lookup['rh_LFJ5']] += (gate*residual[6]).astype(np.float32)
    result[:,lookup['rh_THJ2']] += (gate*residual[7]).astype(np.float32)
    if not np.array_equal(result[:11],source[:11]) or not np.array_equal(result[151:],source[151:]) or not np.array_equal(result[:,:6],source[:,:6]):
        raise RuntimeError('multifinger residual altered arm or outside-gate commands')
    return result


def active_digit_forces(forces, threshold=FORCE_THRESHOLD_N):
    values=np.asarray(forces,dtype=float)
    if values.ndim!=2 or values.shape[1]!=5 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('expected nonnegative per-step five-digit force array')
    active=values>threshold
    triad=active[:,0] & (active[:,1:].sum(axis=1)>=2)
    return active,triad


def static_hold_metrics(relative_positions, relative_rotations, object_positions, forces,
                        lift_from_settled_mm, target_support_steps, robot_table_steps,
                        dt_s=.0005):
    """Airborne, unsupported two-second hold, including maximum transient drift."""
    rel=np.asarray(relative_positions,dtype=float)
    rot=np.asarray(relative_rotations,dtype=float)
    obj=np.asarray(object_positions,dtype=float)
    force=np.asarray(forces,dtype=float)
    support=np.asarray(target_support_steps,dtype=bool)
    table=np.asarray(robot_table_steps,dtype=bool)
    if (rel.ndim!=2 or rel.shape[1]!=3 or obj.shape!=rel.shape
            or rot.shape!=(len(rel),3,3) or force.shape!=(len(rel)-1,5)
            or support.shape!=(len(force),) or table.shape!=(len(force),)
            or not np.isfinite(rot).all() or not np.isfinite(rel).all()
            or not np.isfinite(obj).all() or not np.isfinite(force).all()
            or not np.isfinite(lift_from_settled_mm)
            or abs((len(rel)-1)*dt_s-2.)>1e-8):
        raise ValueError('invalid two-second static hold evidence')
    active,triad=active_digit_forces(force)
    last=round(1./dt_s)
    slip_trace=np.linalg.norm(rel-rel[0],axis=1)*1000
    turn_trace=np.rad2deg(Rotation.from_matrix(rot@rot[0].T).magnitude())
    drift_trace=np.linalg.norm(rel[-last-1:]-rel[-last-1],axis=1)*1000
    slip=float(slip_trace[-1]);peak_slip=float(np.max(slip_trace))
    drop=float(max(0.,(obj[0,2]-obj[-1,2])*1000))
    peak_drop=float(max(0.,np.max((obj[0,2]-obj[:,2])*1000)))
    drift=float(drift_trace[-1]);peak_drift=float(np.max(drift_trace))
    turn=float(turn_trace[-1]);peak_turn=float(np.max(turn_trace))
    triad_fraction=float(np.mean(triad[-last:]))
    airborne=bool(lift_from_settled_mm>20. and not support.any() and not table.any())
    # Feasibility penalty prevents a tabletop pinch from outranking an airborne grip.
    score=float(min(300.,peak_slip)+min(300.,peak_drop)+min(100.,peak_drift)
                +min(100.,peak_turn)+100*(1-triad_fraction)+1000*(not airborne))
    return {'lift_from_settled_mm':float(lift_from_settled_mm),
            'slip_2s_mm':slip,'peak_slip_2s_mm':peak_slip,
            'z_drop_2s_mm':drop,'peak_z_drop_2s_mm':peak_drop,
            'last_1s_drift_mm':drift,'peak_last_1s_drift_mm':peak_drift,
            'relative_rotation_2s_deg':turn,'peak_relative_rotation_2s_deg':peak_turn,
            'target_support_step_fraction_2s':float(support.mean()),
            'target_support_step_fraction_last_1s':float(support[-last:].mean()),
            'robot_table_contact_steps_2s':int(table.sum()),
            'airborne_unpinned':airborne,
            'triad_last_1s_fraction':triad_fraction,
            'digit_active_last_1s_fraction':active[-last:].mean(axis=0).tolist(),
            'mean_force_last_1s_n':force[-last:].mean(axis=0).tolist(),
            'static_score':score,
            'static_pass':bool(airborne and peak_slip<10. and peak_drop<10.
                               and peak_drift<5. and peak_turn<10.
                               and triad_fraction>=.8)}
