"""Independent command replay using the XML built by the original replay runner.

Does not import any placement generator, controller, or its rollout helpers.
The scene and exported command are the only physics inputs. All objects are free.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commands', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=Path('/app/rl-output/shadow_object_pose_parent_original_20261026/scene.xml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.commands, allow_pickle=False) as saved:
        commands, times = saved['commands'], saved['timestamps']
        names = list(map(str, saved['joint_names']))
    if commands.dtype != np.float32 or commands.shape != (len(times), len(names)) or not np.isfinite(commands).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid command export')
    model = mujoco.MjModel.from_xml_path(str(args.scene)); data = mujoco.MjData(model)
    addresses = np.array([model.jnt_qposadr[model.joint(name).id] for name in names])
    obj = model.body('object_2').id
    table = model.geom('table').id
    data.qpos[addresses] = commands[0]; data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    for _ in range(round(.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    digits = ('th', 'ff', 'mf', 'rf', 'lf')
    positions, rotations, q_actual = [], [], []
    step_frames, forces, support, speeds, before_speeds, robot_table, neighbor_contact = [], [], [], [], [], [], []
    obj_ids = [model.body(f'object_{i}').id for i in (1, 2, 3, 4)]
    object_start = data.xpos[obj_ids].copy()
    neighbor_positions, penetrations = [], []
    if args.render:
        import cv2
        renderer = mujoco.Renderer(model, height=600, width=800)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [.02, -.05, .15]; cam.distance = .85; cam.azimuth = 225; cam.elevation = -32
        for geom in range(model.ngeom):
            if model.geom_contype[geom] and model.body(model.geom_bodyid[geom]).name not in ('world', 'object_1', 'object_2', 'object_3', 'object_4'):
                model.geom_rgba[geom, 3] = 0
        writer = cv2.VideoWriter(str(args.output / 'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 600))
    def one_step(frame):
        before = np.linalg.norm(data.cvel[obj, 3:])
        mujoco.mj_step(model, data)
        df = np.zeros(5); ts = False; rt = False; nc = False; penetration = 0.
        for ci, contact in enumerate(data.contact):
            b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
            n1, n2 = model.body(b1).name, model.body(b2).name
            f = np.zeros(6); mujoco.mj_contactForce(model, data, ci, f)
            if obj in (b1, b2):
                other = n2 if b1 == obj else n1
                penetration = min(penetration, float(contact.dist))
                if f[0] > 1e-6 and table in (contact.geom1, contact.geom2): ts = True
                for k, digit in enumerate(digits):
                    if other.startswith('rh_' + digit): df[k] += max(0., float(f[0]))
            if f[0] > 1e-6:
                if table in (contact.geom1, contact.geom2) and any(n.startswith('rh_') or n.endswith('link') for n in (n1, n2)): rt = True
                if any(n in ('object_1','object_3','object_4') for n in (n1,n2)) and any(n.startswith('rh_') or n.endswith('link') for n in (n1,n2)): nc = True
        step_frames.append(frame); forces.append(df); support.append(ts); robot_table.append(rt); neighbor_contact.append(nc)
        speeds.append(float(np.linalg.norm(data.cvel[obj, 3:]))); before_speeds.append(float(before))
        neighbor_positions.append(data.xpos[obj_ids].copy()); penetrations.append(penetration)
    def render(frame):
        if args.render:
            renderer.update_scene(data, cam)
            im = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(im, f'Independent open-loop replay | frame {frame}', (14, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (250,250,250), 1)
            writer.write(im)
    for frame, command in enumerate(commands):
        count = round((times[frame] - times[frame - 1]) / 1e6 / model.opt.timestep) if frame else 0
        previous = commands[max(frame - 1, 0)]
        for step in range(count):
            data.ctrl[:] = previous + (command - previous) * (step + 1) / count
            one_step(frame)
        positions.append(data.xpos[obj].copy()); rotations.append(data.xmat[obj].reshape(3,3).copy()); q_actual.append(data.qpos[addresses].copy())
        render(frame)
    settle_positions, settle_rotations = [], []
    data.ctrl[:] = commands[-1]
    for step in range(round(2. / model.opt.timestep)):
        one_step(len(commands))
        settle_positions.append(data.xpos[obj].copy()); settle_rotations.append(data.xmat[obj].reshape(3,3).copy())
        if step % 67 == 0: render(f'{len(commands)} + hold')
    if args.render: writer.release(); renderer.close()
    sf, force, su = np.asarray(step_frames), np.asarray(forces), np.asarray(support)
    pos, rot = np.asarray(positions), np.asarray(rotations)
    sp, sr = np.asarray(settle_positions), np.asarray(settle_rotations)
    s = np.asarray(speeds); pre_s = np.asarray(before_speeds)
    table_array = np.asarray(robot_table); neighbor_array = np.asarray(neighbor_positions)
    triad = (force[:,0] > .05) & ((force[:,1:] > .05).sum(1) >= 2)
    after_anchor = np.flatnonzero((sf > 110) & su)
    touchdown = int(after_anchor[0]) if len(after_anchor) else None
    carry = (sf > 110) & (np.arange(len(sf)) < touchdown) if touchdown is not None else sf > 110
    longest = run = 0
    for good in triad[carry]:
        run = 0 if good else run + 1; longest = max(longest, run)
    track = np.load('/app/data/datasets/hocap_demo_object_v4/g10_object2_190/intermediates/segmentation/hocap-oracle-v1/object_models.npz')
    col = list(track['object_ids']).index(2); refp = track['translation_world'][-1,col]; refr = track['rotation_world_object'][-1,col]
    last = round(1. / model.opt.timestep)
    report = {'command_sha256':hashlib.sha256(args.commands.read_bytes()).hexdigest(), 'original_scene_sha256':hashlib.sha256(args.scene.read_bytes()).hexdigest(),
              'frames':len(commands), 'duration_s':float((times[-1]-times[0])/1e6),
              'settled_xy_error_mm':float(np.linalg.norm(sp[-1,:2]-refp[:2])*1000), 'settled_z_error_mm':float(abs(sp[-1,2]-refp[2])*1000),
              'settled_rotation_error_deg':float(np.rad2deg(Rotation.from_matrix(refr@sr[-1].T).magnitude())),
              'settled_upright_tilt_deg':float(np.rad2deg(np.arccos(np.clip(sr[-1,2,2],-1,1)))),
              'last_1s_table_support_fraction':float(su[-last:].mean()), 'last_1s_any_hand_force_steps':int((force[-last:].sum(1)>1e-6).sum()),
              'last_1s_max_speed_m_s':float(s[-last:].max()), 'robot_table_force_steps':int(table_array.sum()),
              'robot_neighbor_force_steps':int(sum(neighbor_contact)),
              'neighbor_peak_drift_mm':float(np.linalg.norm(neighbor_array[:,[0,2,3]]-object_start[[0,2,3]],axis=2).max()*1000),
              'minimum_target_contact_gap_mm':float(min(penetrations)*1000),
              'post_anchor_touchdown_frame':int(sf[touchdown]) if touchdown is not None else None,
              'touchdown_speed_before_step_m_s':float(pre_s[touchdown]) if touchdown is not None else None,
              'touchdown_speed_after_step_m_s':float(s[touchdown]) if touchdown is not None else None,
              'pre_touchdown_triad_fraction':float(triad[carry].mean()), 'pre_touchdown_max_triad_gap_s':longest*model.opt.timestep}
    report['controlled_touchdown_pass'] = bool(touchdown is not None and report['pre_touchdown_triad_fraction']>=.9 and report['pre_touchdown_max_triad_gap_s']<=.05 and max(pre_s[touchdown],s[touchdown])<.1)
    np.savez_compressed(args.output/'trajectory.npz', object_positions=pos, object_rotations=rot, q_actual=q_actual,
                        settle_object_positions=sp, settle_object_rotations=sr, physics_step_frame=sf,
                        physics_digit_forces_n=force, physics_table_support=su, physics_object_speed=s, physics_object_pre_step_speed=pre_s)
    (args.output/'review.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
