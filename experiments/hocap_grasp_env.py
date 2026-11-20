"""Simulation-only HO-Cap pickup task for a UR10/Robotiq RL policy."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from experiments.contact_attachment import ContactAttachment
from experiments.hocap_grasp_replay import _scene
from experiments.object_trajectory_reward import ObjectTrajectoryReference
from experiments.scene import ARM_JOINTS
from experiments.verify_gripper import MIMIC_MULTIPLIERS, PAD_OFFSET_M


class HocapGraspEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        plan_path: Path,
        track_path: Path,
        model_root: Path,
        mesh_path: Path,
        gt_pose_path: Path,
        object_name: str,
        object_id: int,
        gt_anchor_frame: int,
        reset_frame: int,
        max_steps: int | None = None,
        reward_mode: str = "pickup",
        render_mode: str | None = None,
        contact_attach: bool = False,
        all_objects: bool = False,
        release_opening_rad: float = 0.12,
        other_mesh_paths: dict[int, Path] | None = None,
    ):
        super().__init__()
        if reward_mode not in ("pickup", "object_track"):
            raise ValueError("invalid reward mode")
        if max_steps is None:
            max_steps = 250 if reward_mode == "object_track" else 125
        if render_mode not in (None, "human") or max_steps < 1:
            raise ValueError("invalid render mode or episode length")
        self.render_mode = render_mode
        self.contact_attach = contact_attach
        self.all_objects = all_objects
        self.reward_mode = reward_mode
        self.model, self.geometry = _scene(
            plan_path, track_path, model_root, object_id, 0.2,
            rubber_pads=True, all_objects=all_objects,
            mesh_path=mesh_path, gt_pose_path=gt_pose_path,
            gt_frame=gt_anchor_frame, object_name=object_name,
            other_mesh_paths=other_mesh_paths,
        )
        with np.load(track_path, allow_pickle=False) as track:
            if not 0 <= reset_frame < len(track["translation_world"]):
                raise ValueError("reset frame is outside the object track")
            object_index = list(track["object_ids"]).index(object_id)
            self.object_position = np.asarray(
                track["translation_world"][reset_frame, object_index], dtype=float,
            )
            rotation = np.asarray(
                track["rotation_world_object"][reset_frame, object_index], dtype=float,
            )
            self.object_quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
            self.other_object_poses = {}
            for other_id in self.geometry["other_object_ids"]:
                other_index = list(track["object_ids"]).index(other_id)
                self.other_object_poses[other_id] = (
                    np.asarray(track["translation_world"][reset_frame, other_index], dtype=float),
                    Rotation.from_matrix(
                        track["rotation_world_object"][reset_frame, other_index],
                    ).as_quat(scalar_first=True),
                )
        import h5py
        with h5py.File(plan_path) as plan:
            self.reference_joint = np.asarray(plan["q"][reset_frame], dtype=float)
            if reward_mode == "object_track":
                timestamps = np.asarray(plan["timestamps"])
                with np.load(track_path, allow_pickle=False) as track:
                    positions = np.asarray(track["translation_world"][:, object_index], dtype=float)
                if len(timestamps) != len(positions):
                    raise ValueError("plan and object track must describe the same frames")
                self.object_reference = ObjectTrajectoryReference(
                    timestamps, positions, reset_frame,
                )
                if max_steps * 0.04 < self.object_reference.duration_s + 0.52:
                    raise ValueError("episode is too short to follow and settle the object track")
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.step_count = 0
        self.stable_lift_steps = 0
        self.stable_goal_steps = 0
        self.has_lifted = False
        self.viewer = None
        self.arm_joints = [self.model.joint(name).id for name in ARM_JOINTS]
        self.arm_qpos = np.array([self.model.jnt_qposadr[joint] for joint in self.arm_joints])
        self.arm_dof = np.array([self.model.jnt_dofadr[joint] for joint in self.arm_joints])
        self.finger_joint = self.model.joint("finger_joint").id
        self.finger_qpos = int(self.model.jnt_qposadr[self.finger_joint])
        self.finger_dof = int(self.model.jnt_dofadr[self.finger_joint])
        self.object_joint = self.model.joint("tracked_object_free").id
        self.object_qpos = int(self.model.jnt_qposadr[self.object_joint])
        self.object_dof = int(self.model.jnt_dofadr[self.object_joint])
        self.object_body = self.model.body("tracked_object").id
        self.object_geom = self.model.geom("tracked_object_geom").id
        self.table_geom = self.model.geom("table").id
        self.other_object_geoms = {
            self.model.geom(f"tracked_object_{other_id}_geom").id: other_id
            for other_id in self.geometry["other_object_ids"]
        }
        self.other_object_joints = {
            other_id: int(self.model.jnt_qposadr[
                self.model.joint(f"tracked_object_{other_id}_free").id
            ])
            for other_id in self.geometry["other_object_ids"]
        }
        self.pad_geoms = {
            self.model.geom("left_rubber_pad").id: "left",
            self.model.geom("right_rubber_pad").id: "right",
        }
        self.wrist_body = self.model.body("wrist_3_link").id
        self.attachment = (
            ContactAttachment(
                self.model, self.data, "tracked_object_grasp",
                "wrist_3_link", "tracked_object",
                release_opening_rad=release_opening_rad,
            ) if contact_attach else None
        )
        self.pad_bodies = [
            self.model.body("left_distal_phalanx").id,
            self.model.body("right_distal_phalanx").id,
        ]
        self.nominal_mass = float(self.model.body_mass[self.object_body])
        self.nominal_inertia = self.model.body_inertia[self.object_body].copy()
        self.object_extent = np.asarray(self.geometry["proxy_half_size_m"], dtype=float) * 2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        observation_size = 43 if reward_mode == "object_track" else 38
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32)
        self.control_steps = round(0.04 / self.model.opt.timestep)
        if not np.isclose(self.control_steps * self.model.opt.timestep, 0.04):
            raise ValueError("physics timestep must divide the 25 Hz control period")

    def _pad_positions(self) -> list[np.ndarray]:
        return [
            self.data.xpos[body].copy()
            + self.data.xmat[body].reshape(3, 3) @ PAD_OFFSET_M
            for body in self.pad_bodies
        ]

    def _contacts(self) -> tuple[dict[str, float], bool, bool]:
        force_n = {"left": 0.0, "right": 0.0}
        table_support = False
        non_pad_contact = False
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other_geom = (contact.geom2 if contact.geom1 == self.object_geom
                          else contact.geom1)
            if other_geom == self.table_geom:
                table_support = True
            side = self.pad_geoms.get(other_geom)
            if side is not None:
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                force_n[side] += max(0.0, float(force[0]))
            elif other_geom != self.table_geom:
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                non_pad_contact |= force[0] >= 0.2
        return force_n, table_support, non_pad_contact

    def _pad_distances(self) -> np.ndarray:
        return np.asarray([
            max(0.0, mujoco.mj_geomDistance(
                self.model, self.data, geom_id, self.object_geom, 0.2, None,
            ))
            for geom_id in self.pad_geoms
        ], dtype=float)

    def _other_object_contacts(self) -> list[int]:
        contacted = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            for other_geom, other_id in self.other_object_geoms.items():
                if other_geom not in (contact.geom1, contact.geom2):
                    continue
                opposite = contact.geom2 if contact.geom1 == other_geom else contact.geom1
                if opposite == self.table_geom or opposite in self.other_object_geoms:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                if force[0] >= 0.2:
                    contacted.add(other_id)
        return sorted(contacted)

    def _robot_table_contacts(self) -> list[str]:
        contacted = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if self.table_geom not in (contact.geom1, contact.geom2):
                continue
            other_geom = (contact.geom2 if contact.geom1 == self.table_geom
                          else contact.geom1)
            if other_geom == self.object_geom or other_geom in self.other_object_geoms:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if force[0] >= 0.2:
                contacted.add(self.model.body(self.model.geom_bodyid[other_geom]).name)
        return sorted(contacted)

    def _observation(self, force_n: dict[str, float]) -> np.ndarray:
        object_position = self.data.xpos[self.object_body]
        pads = self._pad_positions()
        values = np.concatenate((
            self.data.qpos[self.arm_qpos],
            self.data.qvel[self.arm_dof],
            [self.data.qpos[self.finger_qpos], self.data.qvel[self.finger_dof]],
            pads[0] - object_position,
            pads[1] - object_position,
            self.data.qvel[self.object_dof:self.object_dof + 3],
            self.object_extent,
            self._pad_distances(),
            self.reference_joint - self.data.qpos[self.arm_qpos],
            [object_position[2] - self.initial_height,
             force_n["left"], force_n["right"],
             self.step_count / self.max_steps],
        )).astype(np.float32)
        if self.reward_mode == "object_track":
            values = np.concatenate((
                values,
                self.object_reference.target(self.step_count * 0.04) - object_position,
                [float(self.has_lifted), self.stable_goal_steps / 13],
            )).astype(np.float32)
        if values.shape != self.observation_space.shape or not np.isfinite(values).all():
            raise RuntimeError("invalid HO-Cap RL observation")
        return values

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.stable_lift_steps = 0
        self.stable_goal_steps = 0
        self.has_lifted = False
        mass = float(self.np_random.uniform(0.1, 0.3))
        friction = float(self.np_random.uniform(0.7, 1.5))
        self.model.body_mass[self.object_body] = mass
        self.model.body_inertia[self.object_body] = self.nominal_inertia * mass / self.nominal_mass
        self.model.geom_friction[self.object_geom, 0] = friction
        mujoco.mj_resetData(self.model, self.data)
        if self.attachment is not None:
            self.attachment.reset()
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.arm_qpos] = self.reference_joint
        for name, multiplier in MIMIC_MULTIPLIERS.items():
            joint = self.model.joint(name).id
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.0 * multiplier
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = (
            self.object_position + [*self.np_random.uniform(-0.005, 0.005, 2), 0.0]
        )
        self.data.qpos[self.object_qpos + 3:self.object_qpos + 7] = self.object_quaternion
        for other_id, (position, quaternion) in self.other_object_poses.items():
            address = self.other_object_joints[other_id]
            self.data.qpos[address:address + 3] = position
            self.data.qpos[address + 3:address + 7] = quaternion
        self.data.ctrl[:6] = self.reference_joint
        self.data.ctrl[6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.initial_position = self.data.xpos[self.object_body].copy()
        self.initial_other_positions = {
            other_id: self.data.xpos[self.model.body(f"tracked_object_{other_id}").id].copy()
            for other_id in self.other_object_joints
        }
        self.initial_height = float(self.initial_position[2])
        forces, _, _ = self._contacts()
        if self.render_mode == "human":
            self.render()
        return self._observation(forces), {"mass_kg": mass, "friction": friction}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("action must contain five finite controls")
        previous_pad_distance_sum = (float(self._pad_distances().sum())
                                     if self.reward_mode == "object_track" else 0.0)
        action = np.clip(action, -1.0, 1.0)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model, self.data, jacobian_position, jacobian_rotation, self.wrist_body,
        )
        jacobian = np.vstack((
            jacobian_position[:, self.arm_dof],
            jacobian_rotation[:, self.arm_dof],
        ))
        desired_twist = np.array([
            *(.005 * action[:3]), 0.0, 0.0, .03 * action[3],
        ])
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + .001 * np.eye(6), desired_twist,
        )
        arm_targets = self.data.ctrl[:6] + np.clip(joint_delta, -.05, .05)
        for index, joint in enumerate(self.arm_joints):
            if self.model.jnt_limited[joint]:
                arm_targets[index] = np.clip(arm_targets[index], *self.model.jnt_range[joint])
        self.data.ctrl[:6] = arm_targets
        self.data.ctrl[6] = np.clip(self.data.ctrl[6] + 0.05 * action[4], 0.0, 0.8)
        interval_other_contacts = set()
        interval_robot_table_contacts = set()
        interval_non_pad_contact = False
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)
            if self.all_objects:
                interval_other_contacts.update(self._other_object_contacts())
                interval_robot_table_contacts.update(self._robot_table_contacts())
                interval_non_pad_contact |= self._contacts()[2]
        self.step_count += 1
        forces, table_support, non_pad_contact = self._contacts()
        non_pad_contact |= interval_non_pad_contact
        other_contacts = sorted(interval_other_contacts.union(self._other_object_contacts()))
        robot_table_contacts = sorted(interval_robot_table_contacts.union(
            self._robot_table_contacts() if self.all_objects else [],
        ))
        other_shifts = {
            other_id: float(np.linalg.norm(
                self.data.xpos[self.model.body(f"tracked_object_{other_id}").id]
                - initial_position,
            ))
            for other_id, initial_position in self.initial_other_positions.items()
        }
        if self.attachment is not None:
            self.attachment.observe(
                forces, self._pad_positions(), non_pad_contact,
                float(self.data.ctrl[6]), float(self.data.qpos[self.finger_qpos]),
                self.control_steps * self.model.opt.timestep,
            )
        object_position = self.data.xpos[self.object_body]
        lift = float(object_position[2] - self.initial_height)
        shift_xy = float(np.linalg.norm((object_position - self.initial_position)[:2]))
        bilateral = min(forces.values()) >= 0.2
        pad_distances = self._pad_distances()
        airborne = lift >= 0.025 and not table_support
        assisted_hold = self.attachment is not None and self.attachment.attached
        self.stable_lift_steps = self.stable_lift_steps + 1 if airborne and (bilateral or assisted_hold) else 0
        if self.reward_mode == "object_track":
            first_stable_lift = not self.has_lifted and self.stable_lift_steps >= 13
            self.has_lifted |= self.stable_lift_steps >= 13
            elapsed_s = self.step_count * 0.04
            goal_distance_m = float(np.linalg.norm(
                object_position - self.object_reference.positions_m[-1],
            ))
            object_speed_m_s = float(np.linalg.norm(
                self.data.qvel[self.object_dof:self.object_dof + 3],
            ))
            settled = (
                self.has_lifted and elapsed_s >= self.object_reference.duration_s
                and goal_distance_m <= 0.03 and table_support
                and object_speed_m_s <= 0.03
                and all(force < 0.05 for force in forces.values())
            )
            self.stable_goal_steps = self.stable_goal_steps + 1 if settled else 0
            success = self.stable_goal_steps >= 13
            lost = (shift_xy > 0.6 or non_pad_contact or bool(other_contacts)
                    or bool(robot_table_contacts)
                    or object_position[2] < self.geometry["table_z_m"] - 0.03)
            outside_m = self.object_reference.outside_m(object_position, elapsed_s)
            tracking_penalty = -0.25 * outside_m / (outside_m + 0.05)
            reward = (
                -0.02 + 0.8 * (previous_pad_distance_sum - float(pad_distances.sum()))
                + tracking_penalty + 15.0 * first_stable_lift
                - 0.001 * float(np.dot(action, action))
            )
            if success:
                reward += 100.0
            elif lost:
                reward -= 250.0
            elif self.step_count >= self.max_steps:
                reward -= 25.0
        else:
            success = self.stable_lift_steps >= 13
            lost = shift_xy > 0.12 or object_position[2] < self.geometry["table_z_m"] - 0.03
            reward = (
                -0.02 - 0.8 * float(pad_distances.sum())
                + 0.12 * float(sum(force > 0.05 for force in forces.values()))
                + (0.5 * min(max(lift, 0.0) / 0.04, 1.0) if bilateral else 0.0)
                - 1.5 * max(shift_xy - 0.015, 0.0)
                - 0.001 * float(np.dot(action, action))
            )
            if success:
                reward += 25.0
            elif lost:
                reward -= 20.0
        terminated = bool(success or lost)
        truncated = self.step_count >= self.max_steps
        if self.render_mode == "human":
            self.render()
        info = {
            "success": bool(success), "lift_m": lift, "object_shift_xy_m": shift_xy,
            "bilateral_contact": bool(bilateral), "table_support": bool(table_support),
            "stable_lift_steps": self.stable_lift_steps,
            "pad_distances_m": pad_distances.tolist(),
            "pad_forces_n": forces,
            "other_object_contact_ids": other_contacts,
            "robot_table_contact_bodies": robot_table_contacts,
            "other_object_shift_m": other_shifts,
            "contact_attachment_active": bool(assisted_hold),
            "contact_attachment_released": bool(
                self.attachment is not None and self.attachment.released
            ),
        }
        if self.reward_mode == "object_track":
            info.update({
                "outside_track_m": outside_m,
                "tracking_penalty": tracking_penalty,
                "goal_distance_m": goal_distance_m,
                "has_lifted": self.has_lifted,
                "stable_goal_steps": self.stable_goal_steps,
                "non_pad_contact": bool(non_pad_contact),
            })
        return self._observation(forces), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
