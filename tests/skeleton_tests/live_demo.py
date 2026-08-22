"""
scripts/live_demo.py
--------------------
Live skeleton overlay from a 2D webcam using the new modular detector stack
in LIVE_STREAM mode.

Default configuration runs MediaPipeArm + MediaPipeHand together via
CompositeLandmarkDetector in FusionMode.ANY — arm chain (slots 0, 21, 22)
plus full 21 hand keypoints (slots 0..20). Slot 0 (WRIST) is resolved by
priority: arm (priority=0) wins over hand (priority=10).

Usage:
    python scripts/live_demo.py [--camera 0] [--hand right] [--width 640]

Controls:
    q / Esc — quit
"""

import argparse
import time

import cv2
import numpy as np

from viki.capture.base import Frame, CameraIntrinsics
from viki.skeleton.camera_prep import prepare_frame
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    MediaPipeArm,
    MediaPipeHand,
)
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import LM
from viki.skeleton.stats import SkeletonStats, pretty_print

# Default landmarks for post-analysis: wrist (arm) + all fingertips (hand).
_DEFAULT_ANALYSIS_LM = [
    LM.WRIST,
    LM.THUMB_TIP,
    LM.INDEX_TIP,
    LM.MIDDLE_TIP,
    LM.RING_TIP,
    LM.PINKY_TIP,
]


# Finger chains for drawing skeleton overlay
CHAINS = [
    [LM.WRIST, LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP],
    [LM.WRIST, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP],
    [LM.WRIST, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP],
    [LM.WRIST, LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP],
    [LM.WRIST, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP],
    [LM.SHOULDER, LM.ELBOW, LM.WRIST],
]

CHAIN_COLORS = [
    (0, 165, 255),  # thumb   — orange
    (0, 255, 0),  # index   — green
    (255, 255, 0),  # middle  — yellow
    (255, 0, 255),  # ring    — magenta
    (0, 255, 255),  # pinky   — cyan
    (255, 100, 100),  # arm     — light blue
]


def draw_skeleton(frame_bgr: np.ndarray, detection) -> np.ndarray:
    """Draw 2D skeleton overlay from HandDetection pixel coords."""
    img = frame_bgr.copy()
    px = np.full((LM.N, 2), np.nan, dtype=np.float32)
    for landmark, point in detection.points.items():
        px[int(landmark)] = point

    for chain, color in zip(CHAINS, CHAIN_COLORS):
        pts = px[chain]
        if np.isnan(pts).any():
            continue
        for i in range(len(pts) - 1):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[i + 1].astype(int))
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

    # Draw landmark dots
    for i, (u, v) in enumerate(px):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.circle(img, (int(u), int(v)), 4, (255, 255, 255), -1)
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 0), 1)

    return img


def _post_analysis(
    stats: SkeletonStats,
    landmarks: list[int],
    save_anim: str | None,
    plots_dir: str | None,
) -> None:
    """Display position/speed/acceleration plots and 3-D skeleton viz after recording."""
    import os
    import matplotlib.pyplot as plt

    pos, t, _ = stats.position_over_time(landmarks)
    if pos.shape[0] < 3:
        print("Not enough detected frames for post-analysis (need ≥ 3).")
        return

    print(
        f"\nPost-analysis: {pos.shape[0]} detected frames over {t[-1]:.1f}s "
        f"— {len(landmarks)} landmarks selected."
    )

    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Saving plots to {plots_dir}/\n")
    else:
        print("Close each plot window to advance to the next one.\n")

    for fig, title, filename in [
        (
            stats.plot_position(landmarks, axes="xyz"),
            "Position over time",
            "position.png",
        ),
        (stats.plot_speed(landmarks), "Speed over time", "speed.png"),
        (
            stats.plot_acceleration(landmarks),
            "Acceleration over time",
            "acceleration.png",
        ),
        (stats.plot_3d_trace(), "3-D landmark traces", "trace_3d.png"),
    ]:
        if plots_dir is not None:
            path = os.path.join(plots_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved {filename}")
            plt.close(fig)
        else:
            plt.show()

    print("3-D animation — close the window to exit.")
    anim = stats.animate_3d(fps=30.0, save_path=save_anim)
    if anim is not None:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--landmarks",
        default=",".join(str(i) for i in _DEFAULT_ANALYSIS_LM),
        help="Comma-separated landmark indices for post-analysis plots "
        f"(default: wrist + fingertips = {_DEFAULT_ANALYSIS_LM})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip post-analysis plots after recording ends.",
    )
    parser.add_argument(
        "--save-anim",
        metavar="PATH",
        default=None,
        help="Save the 3-D skeleton animation to this MP4 path.",
    )
    parser.add_argument(
        "--save-plots",
        metavar="DIR",
        default=None,
        help="Save all post-analysis plots as PNGs to this folder "
        "(created if it does not exist). Skips interactive display.",
    )
    args = parser.parse_args()

    try:
        analysis_landmarks = [int(x) for x in args.landmarks.split(",") if x.strip()]
    except ValueError:
        parser.error("--landmarks must be comma-separated integers, e.g. '0,8,12'")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    # Arm + hand together. ANY mode keeps frames when either detector succeeds,
    # so missing-hand or missing-arm frames are still useful.
    detector = CompositeLandmarkDetector(
        detectors=[
            MediaPipeArm(hand=args.hand, mode="live"),
            MediaPipeHand(hand=args.hand, mode="live"),
        ],
        mode=FusionMode.ANY,
    )
    stats = SkeletonStats(window=150)

    ret, bgr = cap.read()
    if not ret:
        print("Cannot read from camera")
        return

    h, w = bgr.shape[:2]
    # Downscale if needed
    if w > args.width:
        scale = args.width / w
        proc_w = args.width
        proc_h = int(h * scale)
    else:
        scale = 1.0
        proc_w, proc_h = w, h

    K = np.array(
        [[proc_w * 0.8, 0, proc_w / 2], [0, proc_w * 0.8, proc_h / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    dist = np.zeros(5, dtype=np.float32)
    depth_fake = np.full((proc_h, proc_w), 700, dtype=np.uint16)

    frame_idx = 0
    t0 = time.perf_counter()
    fps_display = 0.0

    print("Running — press q or Esc to quit")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        if scale != 1.0:
            proc = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc = bgr

        frame = Frame(
            color=proc,
            depth=depth_fake,
            timestamp_us=frame_idx * 33333,
            device_id="webcam",
            color_intrinsics=CameraIntrinsics(
                fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                width=proc_w, height=proc_h,
            ),
            depth_intrinsics=CameraIntrinsics(
                fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                width=proc_w, height=proc_h,
            ),
        )
        prepared = prepare_frame(frame)
        detection = detector.detect(prepared)

        if detection is not None:
            stats.update(None, confidence=detection.confidence)
        else:
            stats.update(None)

        # FPS calculation (rolling over last second)
        now = time.perf_counter()
        if now - t0 >= 1.0:
            fps_display = frame_idx / (now - t0)

        display = proc.copy()
        if detection is not None:
            display = draw_skeleton(display, detection)
            wrist = detection.points[LM.WRIST]
            cv2.putText(
                display,
                f"conf={detection.confidence:.2f}",
                (10, proc_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display,
                "no hand",
                (10, proc_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            f"FPS {fps_display:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("ViKi live demo", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    pretty_print(stats.summary())

    if not args.no_plots:
        _post_analysis(stats, analysis_landmarks, args.save_anim, args.save_plots)


if __name__ == "__main__":
    main()
