"""
viki.server.robot_viz
--------------------
MJPEG stream generator for the comprehensive robot trajectory visualisation.

Shows: world origin + board, camera positions + gaze, robot FK arm, human
wrist trail, robot EE trail, reach sphere, base-to-EE line, debug overlay.
All elements are toggleable; view can centre on world origin or robot base.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from viki.config import SKELETON_SMOOTHED_DIR
from viki.viz.mjpeg import mjpeg_chunk, placeholder
from viki.viz.robot_viz_shared import (
    VizConfig,
    camera_gaze_dir,
    camera_world_pos,
    find_urdf,
    fk_positions,
    get_neutral_ee,
    get_reach_radius,
    get_robot_world_pos,
    load_debug_data,
    load_extrinsics,
    load_skeleton_wrist,
    project_to_sphere,
    resolve_robot_alias,
    to_robot_frame,
    EXTRINSICS_FILE,
)


def robot_trajectory_stream(
    h5_path: Path,
    cfg: VizConfig | None = None,
) -> Iterator[bytes]:
    """Yield MJPEG frames showing the comprehensive robot trajectory viz.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 trajectory file.
    cfg : VizConfig or None
        Visualisation configuration (toggles, centering, axes length).

    Yields
    ------
    bytes — MJPEG chunk (JPEG image).
    """
    if cfg is None:
        cfg = VizConfig()

    if not h5_path.exists():
        while True:
            yield mjpeg_chunk(placeholder(640, 480, f"File not found: {h5_path.name}"))
            time.sleep(1)

    # ── Load HDF5 archive ──────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        q_keys = [k for k in ("q_scene_smooth", "q_scene_raw", "q_approach") if k in f]
        if not q_keys:
            while True:
                yield mjpeg_chunk(placeholder(640, 480, f"No joint data in {h5_path.name}"))
                time.sleep(1)
        q_key = q_keys[0]
        q_all = f[q_key][:]
        robot_name: str = f["robot"][()]
        ee_frame: str = f["ee_frame"][()]
        fps = float(f["fps"][()]) if "fps" in f else 15.0
        if isinstance(robot_name, bytes):
            robot_name = robot_name.decode()
        if isinstance(ee_frame, bytes):
            ee_frame = ee_frame.decode()
        base_offset_from_h5 = f["base_offset"][:] if "base_offset" in f else None
    n_frames, _n_joints = q_all.shape

    # Use the offset actually applied during IK (stored in HDF5),
    # falling back to the config-level ROBOT_BASE_OFFSET for legacy files.
    actual_offset = get_robot_world_pos(np.zeros(3))
    if base_offset_from_h5 is not None:
        actual_offset = np.asarray(base_offset_from_h5, dtype=np.float64)

    robot_alias = resolve_robot_alias(robot_name)
    reach_radius = get_reach_radius(robot_alias)

    # ── Load skeleton wrist positions ──────────────────────────────────
    wrist_positions: np.ndarray | None = None
    skel_dir = Path(SKELETON_SMOOTHED_DIR)
    skel_path = skel_dir / h5_path.with_suffix(".npz").name
    if skel_path.exists():
        wrist_positions, _ = load_skeleton_wrist(skel_path)

    # ── Load extrinsics ─────────────────────────────────────────────────
    extrinsics_data = load_extrinsics(EXTRINSICS_FILE)

    # ── Load debug overlay ──────────────────────────────────────────────
    debug_data = load_debug_data()

    # ── Load robot URDF & compute FK ────────────────────────────────────
    urdf_path = find_urdf(robot_alias)
    joint_positions_all: np.ndarray | None = None
    ee_positions: np.ndarray | None = None
    if urdf_path is not None:
        try:
            import pinocchio as pin
            model = pin.buildModelFromUrdf(str(urdf_path))
            data = pin.Data(model)
            joint_positions_all, ee_positions = fk_positions(model, data, q_all, ee_frame)
            # Shift FK positions from URDF frame to world frame
            joint_positions_all = joint_positions_all + actual_offset
            ee_positions = ee_positions + actual_offset
        except Exception:
            pass

    # ── Neutral EE ──────────────────────────────────────────────────────
    p_neutral_robot = get_neutral_ee(robot_alias)
    p_neutral_board = p_neutral_robot + actual_offset

    # ── Process debug data ──────────────────────────────────────────────
    debug_world: np.ndarray | None = None
    debug_robot: np.ndarray | None = None
    debug_base: np.ndarray | None = None
    if debug_data is not None:
        dw = np.array(debug_data.get("world_positions", []), dtype=np.float64)
        dr = np.array(debug_data.get("robot_positions", []), dtype=np.float64)
        if len(dw) > 0 and len(dr) > 0:
            debug_world = dw
            debug_robot = dr
            debug_base = np.array(debug_data.get("robot_base_offset", [0, 0, 0]), dtype=np.float64)

    # ── Prepare reach sphere mesh (static) ─────────────────────────────
    base_offset_world = actual_offset
    sphere_u, sphere_v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    sphere_x = reach_radius * np.cos(sphere_u) * np.sin(sphere_v) + base_offset_world[0]
    sphere_y = reach_radius * np.sin(sphere_u) * np.sin(sphere_v) + base_offset_world[1]
    sphere_z = reach_radius * np.cos(sphere_v) + base_offset_world[2]

    # ── Process debug˙robot positions through reach projection ─────────
    debug_robot_projected: np.ndarray | None = None
    if debug_robot is not None and len(debug_robot) > 0:
        proj_list = []
        for p in debug_robot:
            proj, _ = project_to_sphere(p, base_offset_world, reach_radius)
            proj_list.append(proj)
        debug_robot_projected = np.array(proj_list)

    # ── Process FK EE positions through reach projection ────────────────
    ee_projected: np.ndarray | None = None
    ee_was_projected: list[bool] = []
    if ee_positions is not None:
        proj_list = []
        for p in ee_positions:
            pp, wp = project_to_sphere(p, base_offset_world, reach_radius)
            proj_list.append(pp)
            ee_was_projected.append(wp)
        ee_projected = np.array(proj_list)

    # ── Figure setup ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Compute bounds for auto-scaling
    all_pts: list[np.ndarray] = []
    if extrinsics_data:
        for entry in extrinsics_data:
            cp = camera_world_pos(entry.get("rvec", [0, 0, 0]), entry.get("tvec", [0, 0, 0]))
            all_pts.append(cp)
    if wrist_positions is not None and len(wrist_positions) > 0:
        all_pts.append(wrist_positions[0])
        all_pts.append(wrist_positions[-1])
    if ee_projected is not None and len(ee_projected) > 0:
        all_pts.append(ee_projected[0])
        all_pts.append(ee_projected[-1])
    if debug_world is not None and len(debug_world) > 0:
        all_pts.append(debug_world[0])
        all_pts.append(debug_world[-1])
    if debug_robot_projected is not None and len(debug_robot_projected) > 0:
        all_pts.append(debug_robot_projected[0])
        all_pts.append(debug_robot_projected[-1])
    all_pts.append(base_offset_world)
    all_pts.append(p_neutral_board)
    all_pts.append(np.array([0.0, 0.0, 0.0]))

    if all_pts:
        stacked = np.stack(all_pts)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)
        span = max((hi - lo).max(), 1.0) * 0.6
        center = (lo + hi) / 2.0
    else:
        center = np.zeros(3)
        span = 1.0

    # Override centre point based on config
    if cfg.center_on == "robot":
        centre_at = base_offset_world
    else:
        centre_at = np.zeros(3)

    # ── Static artists (persistent) ────────────────────────────────────
    # Board plane
    board_half = 0.2
    bx = np.linspace(-board_half, board_half, 10)
    by = np.linspace(-board_half, board_half, 10)
    bx, by = np.meshgrid(bx, by)
    bz = np.zeros_like(bx)
    board_surf = ax.plot_surface(bx, by, bz, alpha=0.15, color="gray")
    world_origin = ax.scatter(0, 0, 0, c="red", marker="s", s=60, label="World Origin (Board)")

    # Camera artists (one scatter + quiver per camera)
    cam_scatter: list[Any] = []
    cam_quivers: list[Any] = []
    cam_labels: list[Any] = []
    if extrinsics_data:
        for entry in extrinsics_data:
            dev_id = entry.get("device_id", "?")
            rvec = entry.get("rvec", [0, 0, 0])
            tvec = entry.get("tvec", [0, 0, 0])
            cp = camera_world_pos(rvec, tvec)
            s = ax.scatter(cp[0], cp[1], cp[2], s=80, label=f"Camera {dev_id}")
            cam_scatter.append(s)
            gd = camera_gaze_dir(rvec)
            q = ax.quiver(
                cp[0], cp[1], cp[2],
                gd[0], gd[1], gd[2],
                length=0.3, color="blue", arrow_length_ratio=0.1,
            )
            cam_quivers.append(q)
            t = ax.text(cp[0], cp[1], cp[2], f" {dev_id}", fontsize=8)
            cam_labels.append(t)

    # Robot base
    robot_base_scatter = ax.scatter(
        base_offset_world[0], base_offset_world[1], base_offset_world[2],
        c="black", marker="s", s=120, label="Robot Base",
    )

    # Neutral EE
    neutral_scatter = ax.scatter(
        p_neutral_board[0], p_neutral_board[1], p_neutral_board[2],
        c="orange", marker="*", s=100, label="Neutral EE",
    )

    # Reach sphere wireframe
    sphere_wire = ax.plot_wireframe(
        sphere_x, sphere_y, sphere_z,
        color="gray", alpha=0.15, linewidth=0.5,
    )

    # ── Dynamic artists (updated per frame) ────────────────────────────
    (human_trail_line,) = ax.plot([], [], [], c="magenta", label="Human Wrist Path")
    (robot_trail_line,) = ax.plot([], [], [], c="cyan", label="Robot EE Path")

    (human_current,) = ax.plot([], [], [], "o", c="magenta", ms=8)
    (robot_current,) = ax.plot([], [], [], "X", c="cyan", ms=8)

    (base_to_ee_line,) = ax.plot([], [], [], color="black", linestyle="--", alpha=0.6)

    # Debug overlay lines
    (debug_world_line,) = ax.plot([], [], [], c="darkorange", linewidth=2, linestyle="--", label="Debug: Human Wrist")
    (debug_robot_line,) = ax.plot([], [], [], c="darkblue", linewidth=2, linestyle="--", label="Debug: Robot EE")
    (debug_base_dot,) = ax.plot([], [], [], "s", c="black", ms=8)
    (debug_base_line,) = ax.plot([], [], [], color="black", linestyle=":", alpha=0.4)

    # FK arm lines (one segment per link pair + EE dot)
    fk_lines: list[Any] = []
    fk_ee_dot: Any = None
    if joint_positions_all is not None:
        n_pts = joint_positions_all.shape[1]
        for _ in range(n_pts - 1):
            (line,) = ax.plot([], [], [], "o-", color="tab:blue", lw=2, ms=3)
            fk_lines.append(line)
        (fk_ee_dot,) = ax.plot([], [], [], "o", color="red", ms=5)
        fk_lines.append(fk_ee_dot)

    # Axis labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{h5_path.stem} — {robot_name}")
    f_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

    # ── Legend ──────────────────────────────────────────────────────────
    legend = ax.legend(loc="upper left", fontsize=7, ncol=2)

    # ── Streaming loop ──────────────────────────────────────────────────
    interval_s = 1.0 / max(fps, 1)
    frame_idx = 0

    while True:
        if frame_idx >= n_frames:
            yield mjpeg_chunk(placeholder(640, 480, "Done — trajectory complete"))
            time.sleep(2)
            continue

        # --- Update axes limits ---
        # Dynamic: auto-scale to visible content
        visible_pts: list[np.ndarray] = []
        if wrist_positions is not None and cfg.show_human_trail:
            visible_pts.append(wrist_positions[min(frame_idx, len(wrist_positions) - 1)])
        if ee_projected is not None and cfg.show_robot_trail:
            visible_pts.append(ee_projected[frame_idx])
        visible_pts.append(base_offset_world)
        visible_pts.append(np.zeros(3))
        if extrinsics_data and cfg.show_cameras:
            for entry in extrinsics_data:
                visible_pts.append(camera_world_pos(entry.get("rvec", [0, 0, 0]), entry.get("tvec", [0, 0, 0])))

        if visible_pts:
            vstacked = np.stack(visible_pts)
            vlo = vstacked.min(axis=0)
            vhi = vstacked.max(axis=0)
            vspan = max((vhi - vlo).max(), cfg.axes_length) * 0.6
        else:
            vspan = cfg.axes_length

        ax.set_xlim(centre_at[0] - vspan, centre_at[0] + vspan)
        ax.set_ylim(centre_at[1] - vspan, centre_at[1] + vspan)
        ax.set_zlim(centre_at[2] - vspan, centre_at[2] + vspan)

        # --- Visibility toggles ---
        board_surf.set_visible(cfg.show_board)
        world_origin.set_visible(True)  # always on

        for s in cam_scatter:
            s.set_visible(cfg.show_cameras)
        for q in cam_quivers:
            q.set_visible(cfg.show_cameras)
        for t in cam_labels:
            t.set_visible(cfg.show_cameras)

        robot_base_scatter.set_visible(True)
        neutral_scatter.set_visible(cfg.show_neutral_ee)
        sphere_wire.set_visible(cfg.show_reach_sphere)

        human_trail_line.set_visible(cfg.show_human_trail)
        robot_trail_line.set_visible(cfg.show_robot_trail)
        human_current.set_visible(cfg.show_human_trail)
        robot_current.set_visible(cfg.show_robot_trail)
        base_to_ee_line.set_visible(cfg.show_base_to_ee)

        debug_world_line.set_visible(cfg.show_debug_overlay)
        debug_robot_line.set_visible(cfg.show_debug_overlay)
        debug_base_dot.set_visible(cfg.show_debug_overlay)
        debug_base_line.set_visible(cfg.show_debug_overlay)

        for l in fk_lines:
            l.set_visible(cfg.show_fk_arm)

        # --- Update human trail ---
        if wrist_positions is not None and cfg.show_human_trail:
            end = min(frame_idx + 1, len(wrist_positions))
            seg = wrist_positions[:end]
            human_trail_line.set_data(seg[:, 0], seg[:, 1])
            human_trail_line.set_3d_properties(seg[:, 2])
            human_current.set_data([seg[-1, 0]], [seg[-1, 1]])
            human_current.set_3d_properties([seg[-1, 2]])
        else:
            human_trail_line.set_data([], [])
            human_trail_line.set_3d_properties([])
            human_current.set_data([], [])
            human_current.set_3d_properties([])

        # --- Update robot EE trail (projected) ---
        if ee_projected is not None and cfg.show_robot_trail:
            end = min(frame_idx + 1, len(ee_projected))
            seg = ee_projected[:end]
            robot_trail_line.set_data(seg[:, 0], seg[:, 1])
            robot_trail_line.set_3d_properties(seg[:, 2])
            robot_current.set_data([seg[-1, 0]], [seg[-1, 1]])
            robot_current.set_3d_properties([seg[-1, 2]])
        else:
            robot_trail_line.set_data([], [])
            robot_trail_line.set_3d_properties([])
            robot_current.set_data([], [])
            robot_current.set_3d_properties([])

        # --- Base-to-EE line ---
        if cfg.show_base_to_ee and ee_projected is not None and frame_idx < len(ee_projected):
            ee_p = ee_projected[frame_idx]
            base_to_ee_line.set_data(
                [base_offset_world[0], ee_p[0]],
                [base_offset_world[1], ee_p[1]],
            )
            base_to_ee_line.set_3d_properties([base_offset_world[2], ee_p[2]])
        else:
            base_to_ee_line.set_data([], [])
            base_to_ee_line.set_3d_properties([])

        # --- Debug overlay ---
        if cfg.show_debug_overlay:
            if debug_world is not None:
                debug_world_line.set_data(debug_world[:, 0], debug_world[:, 1])
                debug_world_line.set_3d_properties(debug_world[:, 2])
            if debug_robot_projected is not None:
                debug_robot_line.set_data(debug_robot_projected[:, 0], debug_robot_projected[:, 1])
                debug_robot_line.set_3d_properties(debug_robot_projected[:, 2])
            if debug_base is not None:
                debug_base_dot.set_data([debug_base[0]], [debug_base[1]])
                debug_base_dot.set_3d_properties([debug_base[2]])
                if debug_robot_projected is not None and len(debug_robot_projected) > 0:
                    dr0 = debug_robot_projected[0]
                    debug_base_line.set_data([debug_base[0], dr0[0]], [debug_base[1], dr0[1]])
                    debug_base_line.set_3d_properties([debug_base[2], dr0[2]])
        else:
            debug_world_line.set_data([], [])
            debug_world_line.set_3d_properties([])
            debug_robot_line.set_data([], [])
            debug_robot_line.set_3d_properties([])
            debug_base_dot.set_data([], [])
            debug_base_dot.set_3d_properties([])
            debug_base_line.set_data([], [])
            debug_base_line.set_3d_properties([])

        # --- FK arm ---
        if cfg.show_fk_arm and joint_positions_all is not None and frame_idx < len(joint_positions_all):
            jp = joint_positions_all[frame_idx]
            for i in range(len(fk_lines) - 1):
                if i < len(jp) - 1:
                    fk_lines[i].set_data([jp[i][0], jp[i + 1][0]], [jp[i][1], jp[i + 1][1]])
                    fk_lines[i].set_3d_properties([jp[i][2], jp[i + 1][2]])
            if ee_positions is not None and frame_idx < len(ee_positions):
                fk_lines[-1].set_data([ee_positions[frame_idx][0]], [ee_positions[frame_idx][1]])
                fk_lines[-1].set_3d_properties([ee_positions[frame_idx][2]])
        else:
            for l in fk_lines:
                l.set_data([], [])
                l.set_3d_properties([])

        f_text.set_text(f"Frame {frame_idx + 1} / {n_frames}")

        # --- Render & yield ---
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, [2, 1, 0]]
        yield mjpeg_chunk(img, 85)
        frame_idx += 1
        time.sleep(interval_s)
