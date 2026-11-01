"""Object-frame palm residual and arm-only IK for the frozen G10_2 pilot."""

import numpy as np
from scipy.spatial.transform import Rotation


PALM_BOUNDS = np.array([[-.012, .012]]*3 +
                       [[-np.deg2rad(8), np.deg2rad(8)]]*3, dtype=float)


def palm_gate(n_frames=190):
    if n_frames != 190:
        raise ValueError('palm pilot requires the frozen 190-frame plan')
    x = np.arange(n_frames, dtype=float)
    def smooth(a, b):
        u = np.clip((x-a)/(b-a), 0, 1)
        return u*u*(3-2*u)
    return smooth(10,23)*(1-smooth(110,125))


def object_frame_palm_target(original_position, original_rotation, object_rotation_robot, residual, weight):
    """Apply XYZ translation and XYZ rotation vector in the tracked object frame."""
    residual = np.asarray(residual, dtype=float)
    position = np.asarray(original_position, dtype=float)
    rotation = np.asarray(original_rotation, dtype=float)
    object_rotation = np.asarray(object_rotation_robot, dtype=float)
    if (residual.shape != (6,) or position.shape != (3,) or rotation.shape != (3,3)
            or object_rotation.shape != (3,3) or not np.isfinite(residual).all()
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.isfinite(object_rotation).all() or not 0 <= weight <= 1
            or np.any(residual < PALM_BOUNDS[:,0]) or np.any(residual > PALM_BOUNDS[:,1])):
        raise ValueError('invalid object-frame palm residual or pose')
    shifted = position + weight * object_rotation @ residual[:3]
    local_turn = Rotation.from_rotvec(weight*residual[3:]).as_matrix()
    turned = object_rotation @ local_turn @ object_rotation.T @ rotation
    return shifted, turned


def solve_arm_pose(kinematics, frame, original_q, target_position, target_rotation,
                   *, max_iterations=14, position_tol_m=.0015, orientation_tol_rad=np.deg2rad(1)):
    """Damped 6-D Jacobian IK, seeded from the saved arm command."""
    kinematics.set_frame(frame)
    q = np.asarray(original_q, dtype=float).copy()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('invalid source arm q')
    for _ in range(max_iterations):
        position, rotation, jp, jw = kinematics.pose_jacobian(q)
        dp = np.asarray(target_position) - position
        dw = Rotation.from_matrix(np.asarray(target_rotation) @ rotation.T).as_rotvec()
        if np.linalg.norm(dp) <= position_tol_m*.5 and np.linalg.norm(dw) <= orientation_tol_rad*.5:
            break
        jacobian = np.vstack((jp, jw))
        error = np.r_[dp, dw]
        damping = 1e-5*np.eye(6)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
        delta = np.clip(delta, -.18, .18)
        q = np.clip(q + delta, kinematics.q_min+1e-8, kinematics.q_max-1e-8)
    achieved_position, achieved_rotation = kinematics.pose(q)
    position_error = float(np.linalg.norm(achieved_position-target_position))
    orientation_error = float(np.linalg.norm(Rotation.from_matrix(
        np.asarray(target_rotation) @ achieved_rotation.T).as_rotvec()))
    return q, position_error, orientation_error


def validate_ik_errors(position_errors, orientation_errors,
                       position_tol_m=.0015, orientation_tol_rad=np.deg2rad(1)):
    position_errors = np.asarray(position_errors, dtype=float)
    orientation_errors = np.asarray(orientation_errors, dtype=float)
    if (position_errors.shape != orientation_errors.shape or not np.isfinite(position_errors).all()
            or not np.isfinite(orientation_errors).all() or np.any(position_errors < 0)
            or np.any(orientation_errors < 0)):
        raise ValueError('invalid IK error evidence')
    return bool(np.all(position_errors <= position_tol_m) and
                np.all(orientation_errors <= orientation_tol_rad))


def zero_palm_commands(original):
    """Copy a frozen float32 command plan without arithmetic promotion."""
    commands = np.asarray(original)
    if commands.shape != (190,28) or commands.dtype != np.float32 or not np.isfinite(commands).all():
        raise ValueError('expected finite 190x28 float32 source commands')
    return commands.copy()


def sampled_joint_velocity_margin(q_actual, timestamps, velocity_limits):
    """URDF speed margin between saved physics samples (not a stepwise maximum)."""
    q = np.asarray(q_actual,dtype=float)
    ts = np.asarray(timestamps,dtype=float)
    limits = np.asarray(velocity_limits,dtype=float)
    if (q.ndim!=2 or len(q)<2 or ts.shape!=(len(q),) or limits.shape!=(q.shape[1],)
            or not np.isfinite(q).all() or not np.isfinite(ts).all()
            or not np.isfinite(limits).all() or np.any(np.diff(ts)<=0) or np.any(limits<=0)):
        raise ValueError('invalid sampled joint-speed evidence')
    speed=np.abs(np.diff(q,axis=0))/(np.diff(ts)[:,None]/1e6)
    return float(np.min(limits-speed))
