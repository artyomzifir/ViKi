import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Your own imports ---
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame
from viki.calibration.models import CalibrationExtrinsics
from viki.skeleton.fusion import fuse

# ----------------------------------------------------------------------
# 1. Build mock data
# ----------------------------------------------------------------------
dev_ids = ["cam_left", "cam_right"]

# Camera‑space landmark points (x, y, z) in metres
cam_points = {
    "cam_left": {
        LM.WRIST: np.array([1.0, 1.0, 1.0], dtype=np.float32),
        LM.INDEX_MCP: np.array([0.5, 0.1, 2.0], dtype=np.float32),
    },
    "cam_right": {
        LM.WRIST: np.array([2.0, 2.0, 2.0], dtype=np.float32),
        LM.INDEX_MCP: np.array([-0.5, 1.5, 0.5], dtype=np.float32),
    },
}

lms = {
    dev_id: Landmarks3D(points=pts, device_id=dev_id, timestamp_us=0)
    for dev_id, pts in cam_points.items()
}

# ----------------------------------------------------------------------
# 2. Extrinsics (rvec, tvec → 4x4 matrix inside CalibrationExtrinsics)
# ----------------------------------------------------------------------
# Identity: rvec=0, tvec=0
extr_left = CalibrationExtrinsics(
    rvec=np.zeros(3, dtype=np.float64), tvec=np.zeros(3, dtype=np.float64)
)

# 30° rotation around Y, translation (1.5, 0, 0.2)
theta = np.deg2rad(30)
rvec_right = np.array([0.0, theta, 0.0], dtype=np.float64)

tvec_right = np.array([1.5, 0.0, 0.2], dtype=np.float64)
extr_right = CalibrationExtrinsics(rvec=rvec_right, tvec=tvec_right)

extrinsics = {"cam_left": extr_left, "cam_right": extr_right}

# ----------------------------------------------------------------------
# 3. Run fusion
# ----------------------------------------------------------------------
fused = fuse(dev_ids, lms, extrinsics, timestamp_us=42)
assert fused is not None, "Fusion returned None"

# ----------------------------------------------------------------------
# 4. Compute per‑device world points (for verification / plotting)
# ----------------------------------------------------------------------
world_per_device = {}
for dev_id in dev_ids:
    T = extrinsics[dev_id].trasnform_matrix  # yes, the typo
    world_per_device[dev_id] = {}
    for lm in cam_points[dev_id]:
        vec = cam_points[dev_id][lm]
        pos_mtx = np.eye(4, dtype=np.float64)
        pos_mtx[:3, 3] = vec
        # replicate the (currently ineffective) Z‑flip for visual consistency
        pos_mtx[:3, 2] = -pos_mtx[:3, 2]
        world_vec = (T @ pos_mtx)[:3, 3]
        world_per_device[dev_id][lm] = world_vec

# ----------------------------------------------------------------------
# 5. 3D Plot
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

dev_colors = {"cam_left": "blue", "cam_right": "red"}
lm_markers = {LM.WRIST: "o", LM.INDEX_MCP: "s"}

# Individual device observations (small spheres)
for dev_id in dev_ids:
    xs, ys, zs = [], [], []
    for lm in world_per_device[dev_id]:
        p = world_per_device[dev_id][lm]
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
    ax.scatter(
        xs,
        ys,
        zs,
        c=dev_colors[dev_id],
        label=f"{dev_id} observation",
        s=40,
        edgecolors="k",
        alpha=0.8,
    )

# Fused points (large gold stars)
i = 0
for lm in fused.points:
    p = fused.points[lm]
    ax.scatter(
        *p,
        c="gold" if i % 2 else "blue",
        marker="*",
        s=200,
        edgecolors="k",
        label="Fused (mean)" if lm == LM.WRIST else "",
    )
    i += 1

# Thin lines connecting observations to the fused point
for lm in fused.points:
    fused_pt = fused.points[lm]
    for dev_id in dev_ids:
        if lm in world_per_device[dev_id]:
            obs_pt = world_per_device[dev_id][lm]
            ax.plot(
                [obs_pt[0], fused_pt[0]],
                [obs_pt[1], fused_pt[1]],
                [obs_pt[2], fused_pt[2]],
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(
    "Fusion test – rvec/tvec extrinsics\nFused = average of camera‑space → world points"
)
ax.legend()

# Make axes equal
max_range = (
    max(
        abs(ax.get_xlim()[1] - ax.get_xlim()[0]),
        abs(ax.get_ylim()[1] - ax.get_ylim()[0]),
        abs(ax.get_zlim()[1] - ax.get_zlim()[0]),
    )
    / 2.0
)
mid_x = np.mean(ax.get_xlim())
mid_y = np.mean(ax.get_ylim())
mid_z = np.mean(ax.get_zlim())
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()
