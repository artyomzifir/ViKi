"""
Interactive debug tool for depth projection.

Connects to a real Kinect, runs MediaPipe hand detection, and saves
composite images showing how each color landmark maps to the depth image.

Usage:
    docker compose run --rm terminal \\
        python tests/interactive_tests/depth_projection_debug.py

Press SPACE to capture a frame, ESC or q to quit.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from viki.capture.manager import CameraManager
from viki.capture.base import Frame
from viki.skeleton.camera_prep import prepare_frame
from viki.skeleton.detectors.hand_pose import MediaPipeHand
from viki.skeleton.models import LM


_DEBUG_DIR = "data/debug"
os.makedirs(_DEBUG_DIR, exist_ok=True)

# MediaPipe hand connections for drawing
_HAND_CONNS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def draw_hand_skeleton(img: np.ndarray, points: dict, color=(0, 255, 0)):
    """Draw 2D hand skeleton on an image."""
    for a, b in _HAND_CONNS:
        if a in points and b in points and not np.isnan(points[a]).any():
            pa = tuple(points[a][:2].astype(int))
            pb = tuple(points[b][:2].astype(int))
            cv2.line(img, pa, pb, color, 1)
    for idx, p in points.items():
        if not np.isnan(p).any():
            pt = tuple(p[:2].astype(int))
            cv2.circle(img, pt, 2 if idx != 0 else 4, color, -1)
            cv2.putText(img, str(idx), (pt[0] + 3, pt[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def make_debug_image(
    color_bgr: np.ndarray,
    depth_col: np.ndarray,
    landmarks_2d: dict,
    depth_K,
    backend,
) -> np.ndarray:
    """Build a composite debug image showing the projection for each landmark.

    Returns a BGR image with:
      - Top: color + 2D skeleton
      - Middle: pseudo-color depth + projected points + ROI circles
      - Per-landmark zoomed patches along the bottom
    """
    H, W = color_bgr.shape[:2]
    dh, dw = depth_col.shape[:2]

    # Scale depth to match color width for side-by-side
    depth_resized = cv2.resize(depth_col, (W, H))

    # --- Top row: color with landmarks ---
    color_viz = color_bgr.copy()
    draw_hand_skeleton(color_viz, landmarks_2d, color=(0, 255, 0))
    cv2.putText(color_viz, "COLOR + 2D DETECTION", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Middle row: depth with projected points ---
    depth_viz = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
    # Overlay projected points and ROIs
    r = 15  # ROI radius
    projection_data = []
    for lm_idx, uv in landmarks_2d.items():
        if np.isnan(uv).any():
            continue
        u, v = float(uv[0]), float(uv[1])

        # Project to depth space with 1m guess
        res = backend.project_color_to_depth(u, v, 1.0)
        if res is None:
            continue
        ud, vd = res
        # Scale depth coords (ud, vd) to color image size for overlay
        ud_scaled = ud * W / dw if dw else ud
        vd_scaled = vd * H / dh if dh else vd
        pd = (int(ud_scaled), int(vd_scaled))

        # Draw projected point
        cv2.circle(depth_viz, pd, 2, (0, 0, 255), -1)
        # Draw ROI circle
        cv2.circle(depth_viz, pd, r, (0, 255, 255), 1)
        # Draw line from color point to depth projection
        pc = (int(u), int(v))
        cv2.line(depth_viz, pc, pd, (255, 0, 255), 1)

        projection_data.append({
            "idx": lm_idx,
            "u": u, "v": v,
            "ud": ud, "vd": vd,
            "ru": int(round(ud)), "rv": int(round(vd)),
        })

    cv2.putText(depth_viz, "DEPTH + PROJECTED (red dot) + ROI (yellow circle)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Composite side-by-side ---
    gap = 2
    composite_h = H + gap + H
    composite = np.zeros((composite_h, W + gap + W, 3), dtype=np.uint8)
    composite[:H, :W] = color_viz
    composite[:H, W + gap:] = cv2.resize(color_viz, (W, H))
    composite[H + gap:, :W] = depth_viz

    # Right side: per-landmark zoomed ROI patches
    # Show up to 6 landmarks (wrist + 5 MCPs) in a grid
    key_lms = [0, 1, 5, 9, 13, 17]
    patch_size = 96  # px
    pad = 4
    cols = 3
    rows = 2
    grid_w = cols * (patch_size + pad) + pad
    grid_h = rows * (patch_size + pad) + pad
    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    for i, lm_idx in enumerate(key_lms):
        # Find the projection data for this landmark
        pd_item = next((d for d in projection_data if d["idx"] == lm_idx), None)
        if pd_item is None:
            continue
        ru, rv = pd_item["ru"], pd_item["rv"]

        # Extract zoomed ROI from the raw depth
        v_start = max(0, rv - r)
        v_end = min(dh, rv + r + 1)
        u_start = max(0, ru - r)
        u_end = min(dw, ru + r + 1)

        roi_raw = depth_col[v_start:v_end, u_start:u_end]
        if roi_raw.size == 0:
            continue

        # Normalize ROI for display
        roi_norm = cv2.normalize(roi_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        roi_bgr = cv2.cvtColor(roi_norm, cv2.COLOR_GRAY2BGR)
        # Mark center pixel
        cc = (patch_size // 2, patch_size // 2)
        cv2.circle(roi_bgr, cc, 3, (0, 0, 255), -1)
        # Resize if needed
        roi_bgr = cv2.resize(roi_bgr, (patch_size, patch_size))

        col = i % cols
        row = i // cols
        x0 = pad + col * (patch_size + pad)
        y0 = pad + row * (patch_size + pad)
        grid[y0:y0 + patch_size, x0:x0 + patch_size] = roi_bgr

        # Label
        label = f"{LM(lm_idx).name}"
        cv2.putText(grid, label, (x0, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

    # Place the grid in the right half
    grid_x = W + gap + (W - grid_w) // 2
    grid_y = H + gap + (composite_h - H - gap - grid_h) // 2
    if grid_y > H + gap and grid_x > W + gap:
        composite[grid_y:grid_y + grid_h, grid_x:grid_x + grid_w] = grid

    # Info line at the very bottom
    if projection_data:
        wrist = next((d for d in projection_data if d["idx"] == 0), None)
        if wrist:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            info = f"[{ts}] WRIST: color=({wrist['u']:.0f}, {wrist['v']:.0f}) -> depth=({wrist['ud']:.0f}, {wrist['vd']:.0f})"
            cv2.putText(composite, info, (4, composite.shape[0] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return composite


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Camera device ID (e.g. kinect_0)")
    args = parser.parse_args()

    print("Scanning devices...")
    manager = CameraManager()
    devices = manager.list_devices()
    if not devices:
        print("ERROR: no cameras found")
        return

    if args.device:
        dev_id = args.device
        if dev_id not in devices:
            print(f"ERROR: device '{dev_id}' not found. Available: {list(devices.keys())}")
            return
    else:
        print("Available devices:")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d}")
        idx = int(input("Select device index: "))
        dev_id = list(devices.keys())[idx]

    print(f"Using camera: {dev_id}")

    print("Starting camera...")
    manager.start(dev_id)

    # Wait a moment for the camera to settle
    time.sleep(1)

    # Get backend for depth projection
    backend = manager.get_backend(dev_id)
    if backend is None:
        print("ERROR: no backend for", dev_id)
        return

    print("Initializing MediaPipe hand detector...")
    detector = MediaPipeHand(hand="right", mode="image")

    print(f"\nDebug images saved to {_DEBUG_DIR}/")
    print("Press SPACE to capture, ESC/q to quit.\n")

    frame_count = 0
    running = True
    while running:
        # Capture a frame
        tick = time.time_ns() // 1000
        frame: Frame | None = manager.nearest_frame(dev_id, tick)

        if frame is None:
            time.sleep(0.1)
            continue

        # Prepare frame
        prepared = prepare_frame(frame)

        # Run MediaPipe detection
        det = detector.detect(prepared)
        if det is None:
            print("No hand detected")
            time.sleep(0.1)
            continue

        # Build landmarks dict: {idx: [u, v]}
        landmarks_2d = {}
        for k, idx in enumerate(det.indices):
            uv = det.px[k]
            if not np.isnan(uv).any():
                landmarks_2d[int(idx)] = uv

        # Pseudo-color depth (clean NaN before normalize)
        depth_vis = np.nan_to_num(prepared.depth_m, nan=0.0, posinf=2.0)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_col = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Build debug composite
        composite = make_debug_image(
            frame.color,
            depth_col,
            landmarks_2d,
            prepared.depth_K,
            backend,
        )

        # Show window
        scale = 800 / composite.shape[1]
        display = cv2.resize(composite, None, fx=scale, fy=scale)
        cv2.imshow("Depth Projection Debug (SPACE=save, ESC=quit)", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord("q"):
            running = False
        elif key == ord(" "):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(_DEBUG_DIR, f"depth_proj_{ts}.png")
            cv2.imwrite(path, composite)
            print(f"Saved {path}")
            frame_count += 1

    cv2.destroyAllWindows()
    detector.close()
    manager.stop(dev_id)
    print(f"\nDone. {frame_count} frames saved.")


if __name__ == "__main__":
    main()
