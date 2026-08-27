"""Debug logging for retargeting pipeline.

Saves world-frame and robot-frame wrist positions during retargeting
so the visualization can overlay the actual IK input targets.

Every call to save_retarget_debug overwrites the previous file.
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

DEBUG_PATH = Path("data/retarget_debug.json")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def save_retarget_debug(
    *,
    world_positions: np.ndarray,
    robot_positions: np.ndarray,
    robot_base_offset: list[float] | np.ndarray,
    target_offset: list[float] | np.ndarray,
    trajectory_scale: float,
    recenter_to_neutral: bool,
    robot_name: str = "",
    sample_file: str = "",
) -> None:
    """Overwrite data/retarget_debug.json with the current retargeting state."""
    if isinstance(robot_base_offset, np.ndarray):
        robot_base_offset = robot_base_offset.tolist()
    if isinstance(target_offset, np.ndarray):
        target_offset = target_offset.tolist()

    data = {
        "world_positions": np.asarray(world_positions).tolist(),
        "robot_positions": np.asarray(robot_positions).tolist(),
        "robot_base_offset": list(robot_base_offset),
        "target_offset": list(target_offset),
        "trajectory_scale": float(trajectory_scale),
        "recenter_to_neutral": bool(recenter_to_neutral),
        "robot_name": robot_name,
        "sample_file": sample_file,
        "frame_count": int(len(world_positions)),
    }
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_PATH, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder)


def load_retarget_debug() -> dict[str, Any] | None:
    """Load the last saved debug data, or None if unavailable."""
    try:
        with open(DEBUG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def debug_viz_plot(
    debug: dict[str, Any],
    ax: Any | None = None,
) -> Any:
    """Plot world vs robot positions on a 3D axis. Returns the axis."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    world = np.array(debug["world_positions"], dtype=np.float64)
    robot = np.array(debug["robot_positions"], dtype=np.float64)
    base = np.array(debug["robot_base_offset"], dtype=np.float64)
    target_nudge = np.array(debug["target_offset"], dtype=np.float64)

    ax.scatter(0, 0, 0, c="red", marker="s", s=80, label="World origin (board)")

    ax.scatter(
        base[0], base[1], base[2],
        c="black", marker="s", s=150, label="Robot base (world)",
    )

    ax.plot(
        world[:, 0], world[:, 1], world[:, 2],
        c="magenta", alpha=0.7, label="Human wrist (world frame)",
    )
    ax.scatter(
        world[0, 0], world[0, 1], world[0, 2],
        c="magenta", marker="o", s=60,
    )

    ax.plot(
        robot[:, 0], robot[:, 1], robot[:, 2],
        c="cyan", alpha=0.7, label="Robot EE target (robot frame)",
    )
    ax.scatter(
        robot[0, 0], robot[0, 1], robot[0, 2],
        c="cyan", marker="X", s=60,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    title_parts = [f"Robot: {debug.get('robot_name', '?')}"]
    if debug.get("sample_file"):
        title_parts.append(f" | {Path(debug['sample_file']).name}")
    ax.set_title(" | ".join(title_parts))
    ax.legend(loc="upper left", fontsize=8)

    all_pts = np.vstack([world, robot, base.reshape(1, 3), [[0, 0, 0]]])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    span = max((hi - lo).max(), 1.0) * 0.6
    center = (lo + hi) / 2.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.grid(True)

    return ax


def render_debug_viz_png() -> bytes | None:
    """Render the debug plot to PNG bytes, or None if no debug data."""
    debug = load_retarget_debug()
    if debug is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    debug_viz_plot(debug, ax)
    fig.canvas.draw()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
