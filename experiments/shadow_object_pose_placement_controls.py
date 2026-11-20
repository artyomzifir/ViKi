"""Rigid object-center geometry for the opt-in Shadow placement pilot."""

import numpy as np
from scipy.spatial.transform import Rotation


def compose(a_position,a_rotation,b_position,b_rotation):
    ap=np.asarray(a_position,dtype=float);ar=np.asarray(a_rotation,dtype=float)
    bp=np.asarray(b_position,dtype=float);br=np.asarray(b_rotation,dtype=float)
    if ap.shape!=(3,) or bp.shape!=(3,) or ar.shape!=(3,3) or br.shape!=(3,3) or not all(np.isfinite(x).all() for x in (ap,bp,ar,br)):
        raise ValueError('invalid rigid pose')
    return ap+ar@bp,ar@br


def inverse(position,rotation):
    p=np.asarray(position,dtype=float);r=np.asarray(rotation,dtype=float)
    if p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('invalid rigid pose')
    return -r.T@p,r.T


def object_to_palm(object_position,object_rotation,palm_position,palm_rotation):
    ip,ir=inverse(object_position,object_rotation)
    return compose(ip,ir,palm_position,palm_rotation)


def support_center_height(vertices,rotation,table_z,clearance_m=0.):
    v=np.asarray(vertices,dtype=float);r=np.asarray(rotation,dtype=float)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or r.shape!=(3,3)
            or not np.isfinite(v).all() or not np.isfinite(r).all()
            or not np.isfinite(table_z) or not np.isfinite(clearance_m)
            or clearance_m<0):
        raise ValueError('invalid support geometry')
    return float(table_z-np.min((v@r.T)[:,2])+clearance_m)


def smoothstep(frame,start,end):
    if not (np.isfinite(frame) and np.isfinite(start) and np.isfinite(end) and start<end):
        raise ValueError('invalid smooth gate')
    u=np.clip((frame-start)/(end-start),0.,1.)
    return float(u*u*(3-2*u))


def corrected_object_and_palm(nominal_object_position,nominal_object_rotation,
                              object_to_palm_position,object_to_palm_rotation,
                              endpoint_displacement_world,endpoint_rotation_world,weight):
    """Rotate about the object origin, then place the palm via measured grasp SE(3)."""
    p=np.asarray(nominal_object_position,dtype=float)
    r=np.asarray(nominal_object_rotation,dtype=float)
    dp=np.asarray(endpoint_displacement_world,dtype=float)
    dr=np.asarray(endpoint_rotation_world,dtype=float)
    if (p.shape!=(3,) or r.shape!=(3,3) or dp.shape!=(3,) or dr.shape!=(3,3)
            or not all(np.isfinite(x).all() for x in (p,r,dp,dr)) or not 0<=weight<=1):
        raise ValueError('invalid object correction')
    position=p+weight*dp
    turn=Rotation.from_rotvec(weight*Rotation.from_matrix(dr).as_rotvec()).as_matrix()
    rotation=turn@r
    palm_p,palm_r=compose(position,rotation,object_to_palm_position,object_to_palm_rotation)
    return position,rotation,palm_p,palm_r


def settled_placement_gate(evidence):
    """Additional two-second placement diagnostic, separate from original task gates."""
    numeric=('final_xy_error_mm','final_z_error_mm','final_orientation_error_deg',
             'final_upright_tilt_deg','last_1s_table_support_fraction',
             'last_1s_max_speed_m_s','last_1s_peak_position_drift_mm',
             'last_1s_peak_rotation_drift_deg')
    counts=('last_1s_hand_force_steps','last_1s_robot_table_steps')
    if (not all(key in evidence and np.isfinite(evidence[key]) for key in numeric+counts)
            or any(evidence[key]<0 for key in numeric+counts)
            or evidence['last_1s_table_support_fraction']>1):
        raise ValueError('invalid settle evidence')
    e=evidence
    return bool(e['final_xy_error_mm']<30 and e['final_z_error_mm']<20
                and e['final_orientation_error_deg']<15 and e['final_upright_tilt_deg']<15
                and e['last_1s_table_support_fraction']>=.95
                and e['last_1s_hand_force_steps']==0 and e['last_1s_robot_table_steps']==0
                and e['last_1s_max_speed_m_s']<.03
                and e['last_1s_peak_position_drift_mm']<5
                and e['last_1s_peak_rotation_drift_deg']<5)
