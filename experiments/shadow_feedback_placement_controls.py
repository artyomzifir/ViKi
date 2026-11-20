"""Pure bounded SE(3) steps and event gates for the offline Shadow pilot."""

import numpy as np
from scipy.spatial.transform import Rotation


def bounded_step(vector, limit):
    value=np.asarray(vector,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all() or not np.isfinite(limit) or limit<=0:
        raise ValueError('invalid bounded step')
    magnitude=float(np.linalg.norm(value))
    return value if magnitude<=limit else value*(limit/magnitude)


def orient_toward(current,target,max_angle_rad):
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    if current.shape!=(3,3) or target.shape!=(3,3) or not np.isfinite(current).all() or not np.isfinite(target).all() or not np.isfinite(max_angle_rad) or max_angle_rad<=0:
        raise ValueError('invalid orientation step')
    error=Rotation.from_matrix(target@current.T).as_rotvec()
    step=bounded_step(error,max_angle_rad)
    return Rotation.from_rotvec(step).as_matrix()@current


def palm_for_object_step(object_position,object_rotation,palm_position,palm_rotation,
                         target_object_position,target_object_rotation,position_step_m,rotation_step_rad):
    """Use CURRENT grasp-relative pose so a small object-center change moves palm."""
    p=np.asarray(object_position,dtype=float);r=np.asarray(object_rotation,dtype=float)
    pp=np.asarray(palm_position,dtype=float);pr=np.asarray(palm_rotation,dtype=float)
    wanted=np.asarray(target_object_position,dtype=float)
    if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in ((p,(3,)),(r,(3,3)),(pp,(3,)),(pr,(3,3)),(wanted,(3,)))):
        raise ValueError('invalid object/palm pose')
    dp=bounded_step(wanted-p,position_step_m)
    nr=orient_toward(r,target_object_rotation,rotation_step_rad)
    turn=nr@r.T
    return pp+dp+(turn-np.eye(3))@(pp-p),turn@pr


def smooth_palm_command(raw_position,raw_rotation,previous_position,previous_rotation,
                        last_step_position,last_step_rotation,max_position_step_m,max_rotation_step_rad,
                        max_position_acceleration_m,max_rotation_acceleration_rad):
    """Bound world-palm speed and frame-to-frame command acceleration."""
    p=np.asarray(previous_position,dtype=float);r=np.asarray(previous_rotation,dtype=float)
    rawp=np.asarray(raw_position,dtype=float);rawr=np.asarray(raw_rotation,dtype=float)
    lastp=np.asarray(last_step_position,dtype=float);lastr=np.asarray(last_step_rotation,dtype=float)
    if not all(np.isfinite(x).all() for x in (p,r,rawp,rawr,lastp,lastr)) or any(x.shape!=shape for x,shape in ((p,(3,)),(r,(3,3)),(rawp,(3,)),(rawr,(3,3)),(lastp,(3,)),(lastr,(3,)))):
        raise ValueError('invalid command pose')
    desired_dp=bounded_step(rawp-p,max_position_step_m)
    dp=lastp+bounded_step(desired_dp-lastp,max_position_acceleration_m)
    dp=bounded_step(dp,max_position_step_m)
    desired_dr=bounded_step(Rotation.from_matrix(rawr@r.T).as_rotvec(),max_rotation_step_rad)
    dr=lastr+bounded_step(desired_dr-lastr,max_rotation_acceleration_rad)
    dr=bounded_step(dr,max_rotation_step_rad)
    return p+dp,Rotation.from_rotvec(dr).as_matrix()@r,dp,dr


def release_ready(support_fraction,tilt_deg,speed_m_s,consecutive_support_frames):
    if not all(np.isfinite(x) for x in (support_fraction,tilt_deg,speed_m_s,consecutive_support_frames)):
        raise ValueError('invalid support evidence')
    return bool(support_fraction>=.95 and tilt_deg<8 and speed_m_s<.03 and consecutive_support_frames>=9)


def force_residual_step(current,measured,target,gain,caps):
    """Bounded integral closure on TH, FF, MF; saturation is an explicit control bound."""
    q=np.asarray(current,dtype=float);f=np.asarray(measured,dtype=float)
    t=np.asarray(target,dtype=float);c=np.asarray(caps,dtype=float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (q,f,t,c)) or not np.isfinite(gain) or gain<=0 or np.any(q<0) or np.any(c<=0) or np.any(q>c) or np.any(t<0):
        raise ValueError('invalid digit feedback')
    return np.minimum(c,q+gain*np.maximum(0.,t-f))


def residual_finger_command(source_fingers,names,residual):
    """Apply validated flexion synergies; never clamp against URDF limits here."""
    fingers=np.asarray(source_fingers).copy();res=np.asarray(residual,dtype=float)
    if fingers.shape!=(len(names),) or res.shape!=(3,) or not np.isfinite(fingers).all() or not np.isfinite(res).all():
        raise ValueError('invalid finger command')
    lookup={n:i for i,n in enumerate(names)}
    if len(lookup)!=len(names):raise ValueError('duplicate joint names')
    fingers[lookup['rh_THJ4']]+=res[0]
    for digit,amount in (('FF',res[1]),('MF',res[2])):
        for joint,weight in ((1,1.),(2,.7),(3,.35)):
            fingers[lookup[f'rh_{digit}J{joint}']]+=weight*amount
    return fingers


def controlled_touchdown_gate(triad_fraction,longest_gap_s,first_support_speed_m_s,
                              robot_table_steps,robot_other_steps):
    """Review-added validity check: upright landing after a dropped grasp is failure."""
    values=(triad_fraction,longest_gap_s,first_support_speed_m_s,robot_table_steps,robot_other_steps)
    if any(x is None or not np.isfinite(x) or x<0 for x in values) or triad_fraction>1:
        raise ValueError('invalid touchdown evidence')
    return bool(triad_fraction>=.9 and longest_gap_s<=.05 and
        first_support_speed_m_s<.1 and robot_table_steps==0 and robot_other_steps==0)
