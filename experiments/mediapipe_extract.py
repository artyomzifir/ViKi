"""
mediapipe_extract.py — 3D skeleton extraction via MediaPipe Tasks API
mediapipe >= 0.10

Usage:
    python mediapipe_extract.py video.mp4 [--preview]
    python mediapipe_extract.py ./recordings/ [--preview]

При первом запуске скачивает модели (~30 MB) в папку скрипта.

Output .npz:
    body              (T, 33, 3)  метры, origin = центр бёдер
    right_hand        (T, 21, 3)  метры, origin = запястье — после интерполяции
    left_hand         (T, 21, 3)
    right_hand_raw    (T, 21, 3)  до интерполяции (нули = нет детекции)
    left_hand_raw     (T, 21, 3)
    right_interpolated (T,)       bool — какие кадры были интерполированы
    left_interpolated  (T,)       bool
    visibility        (T, 33)
    fps               float
"""

import sys, os, time, argparse, urllib.request
from pathlib import Path
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    sys.exit("pip install mediapipe")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("WARN: matplotlib not found — confidence plots disabled")

SCRIPT_DIR = Path(__file__).parent

MODELS = {
    "pose": (
        SCRIPT_DIR / "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "hand": (
        SCRIPT_DIR / "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}

# ── Landmark connections for drawing ─────────────────────────────────────────

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),   # arms
    (11,12),(23,24),(11,23),(12,24),   # torso
    (23,25),(25,27),(24,26),(26,28),   # legs
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COLOR_BODY        = (0,   220, 100)
COLOR_HAND_DET    = (255, 140,  0 )   # orange — detected
COLOR_HAND_INTERP = (80,  80,  255)   # blue   — interpolated
COLOR_MISSING     = (60,  60,  60 )   # dark grey — no data at all


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_models():
    for name, (path, url) in MODELS.items():
        if path.exists():
            continue
        print(f"Downloading {name} model → {path.name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  OK ({path.stat().st_size // 1024} KB)")


def interpolate_missing(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill missing frames in a hand landmark array, guaranteeing 100% coverage.

    Strategy:
        - Between two detected frames: cubic spline (smooth, no linear kinks)
        - Before first detection / after last detection: hold (constant extrapolation)
        - If fewer than 2 detected frames: hold the only detected frame or return zeros

    Args:
        arr: (T, 21, 3)  raw — zeros where hand not detected

    Returns:
        filled:       (T, 21, 3)  100% filled, no zero rows
        interpolated: (T,) bool   True where a frame was synthesised
    """
    try:
        from scipy.interpolate import CubicSpline
        have_scipy = True
    except ImportError:
        have_scipy = False

    T            = len(arr)
    detected     = np.any(arr != 0, axis=(1, 2))   # (T,) bool
    interpolated = ~detected                         # everything not detected = filled
    filled       = arr.copy()

    det_idx = np.where(detected)[0]

    if det_idx.size == 0:
        # nothing detected at all — return zeros, mark all as interpolated
        return filled, interpolated

    if det_idx.size == 1:
        # only one frame — hold it everywhere
        filled[:] = arr[det_idx[0]]
        return filled, interpolated

    all_idx = np.arange(T)

    for i in range(21):
        for j in range(3):
            col_det = arr[det_idx, i, j]   # values at detected frames

            if have_scipy and det_idx.size >= 4:
                # cubic spline through all detected points
                cs  = CubicSpline(det_idx, col_det, extrapolate=False)
                col = cs(all_idx)
                # NaN outside the knot range (extrapolate=False) → hold edge values
                first, last = det_idx[0], det_idx[-1]
                col[:first] = col_det[0]
                col[last+1:] = col_det[-1]
            else:
                # fallback: linear interpolation + hold edges
                col = np.interp(all_idx, det_idx, col_det)

            filled[:, i, j] = col.astype(np.float32)

    return filled, interpolated


def draw_hand(frame, lm_img, connections, color, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in lm_img]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, color, -1)


def save_confidence_plot(
    stem: str,
    out_dir: Path,
    fps: float,
    body_conf: np.ndarray,    # (T,) float  mean landmark presence
    r_conf: np.ndarray,       # (T,) float  0 when not detected
    l_conf: np.ndarray,       # (T,) float
    r_interp: np.ndarray,     # (T,) bool
    l_interp: np.ndarray,
):
    if not MPL_OK:
        return

    T    = len(body_conf)
    t    = np.arange(T) / fps

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Landmark presence confidence & interpolation — {stem}",
                 fontsize=12, fontweight="bold")

    pairs = [
        (axes[0], body_conf, None,     "Body (mean landmark presence)",       "#00b894"),
        (axes[1], r_conf,    r_interp, "Right hand (mean landmark presence)", "#e17055"),
        (axes[2], l_conf,    l_interp, "Left hand (mean landmark presence)",  "#6c5ce7"),
    ]

    for ax, conf, interp, label, color in pairs:
        # smooth slightly for readability
        kernel = np.ones(5) / 5
        conf_s = np.convolve(conf, kernel, mode="same")

        ax.plot(t, conf,   color=color,   lw=0.6, alpha=0.4)           # raw
        ax.plot(t, conf_s, color=color,   lw=1.5, label=label)         # smoothed
        ax.fill_between(t, conf_s, 0, alpha=0.12, color=color)
        ax.axhline(0.3, color="#b2bec3", lw=0.9, ls="--", alpha=0.8,
                   label="threshold 0.3")

        # shade interpolated zones
        if interp is not None and interp.any():
            in_zone, z_start = False, 0.0
            for i, v in enumerate(interp):
                if v and not in_zone:
                    z_start = t[i]; in_zone = True
                elif not v and in_zone:
                    ax.axvspan(z_start, t[i], alpha=0.30,
                               color="#4a90d9", label="_nolegend_")
                    in_zone = False
            if in_zone:
                ax.axvspan(z_start, t[-1], alpha=0.30, color="#4a90d9")

        ax.set_ylabel("Presence score")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.2)

        handles = [
            mpatches.Patch(color=color,     alpha=0.7, label=label),
            mpatches.Patch(color="#4a90d9", alpha=0.4, label="interpolated zone"),
            plt.Line2D([0],[0], color="#b2bec3", ls="--", lw=0.9,
                       label="threshold 0.3"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    out = out_dir / f"{stem}_confidence.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  plot → {out.name}")


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(video_path: str, preview: bool = False) -> str:
    video_path = Path(video_path).resolve()
    out_npz    = video_path.with_suffix(".npz")
    out_dir    = video_path.parent

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}"); return None

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n{video_path.name}  {W}x{H}  {fps:.0f}fps  ~{total} frames")

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MODELS["pose"][0])),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MODELS["hand"][0])),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,   # ↓ снижено
        min_hand_presence_confidence=0.3,    # ↓ ключевой порог для трекинга
        min_tracking_confidence=0.3,         # ↓ снижено
    )

    body_buf  = []
    rhand_buf = []
    lhand_buf = []
    vis_buf   = []

    # per-frame mean landmark presence (0.0 when not detected)
    body_conf_buf  = []
    rhand_conf_buf = []
    lhand_conf_buf = []

    # image-space landmarks for preview (norm coords)
    rhand_img_buf = []
    lhand_img_buf = []

    t0 = time.monotonic()

    with (mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_det,
          mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det):

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(frame_idx * 1000 / fps)

            p_res = pose_det.detect_for_video(mp_img, ts_ms)
            h_res = hand_det.detect_for_video(mp_img, ts_ms)

            # ── Body ─────────────────────────────────────────────────────
            body_row = np.zeros((33, 3), dtype=np.float32)
            vis_row  = np.zeros(33,      dtype=np.float32)
            body_conf = 0.0
            if p_res.pose_world_landmarks:
                presences = []
                for i, lm in enumerate(p_res.pose_world_landmarks[0]):
                    body_row[i] = [lm.x, lm.y, lm.z]
                    vis_row[i]  = getattr(lm, "visibility", 1.0)
                    p = getattr(lm, "presence", None)
                    presences.append(p if p is not None else vis_row[i])
                presences = [p for p in presences if p is not None]
                body_conf = float(np.mean(presences)) if presences else 0.0

            # ── Hands (world) ─────────────────────────────────────────────
            rhand_row     = np.zeros((21, 3), dtype=np.float32)
            lhand_row     = np.zeros((21, 3), dtype=np.float32)
            rhand_img_row = None
            lhand_img_row = None
            rhand_conf    = 0.0
            lhand_conf    = 0.0

            if h_res.hand_world_landmarks:
                for hand_world, hand_img, handedness in zip(
                    h_res.hand_world_landmarks,
                    h_res.hand_landmarks,
                    h_res.handedness,
                ):
                    label = handedness[0].category_name
                    arr   = np.array([[lm.x, lm.y, lm.z]
                                      for lm in hand_world], dtype=np.float32)
                    pvals = [getattr(lm, "presence", None) for lm in hand_world]
                    pvals = [p for p in pvals if p is not None]
                    conf  = float(np.mean(pvals)) if pvals else 1.0
                    if label == "Right":
                        rhand_row     = arr
                        rhand_img_row = hand_img
                        rhand_conf    = conf
                    else:
                        lhand_row     = arr
                        lhand_img_row = hand_img
                        lhand_conf    = conf

            body_buf.append(body_row)
            rhand_buf.append(rhand_row)
            lhand_buf.append(lhand_row)
            vis_buf.append(vis_row)
            body_conf_buf.append(body_conf)
            rhand_conf_buf.append(rhand_conf)
            lhand_conf_buf.append(lhand_conf)
            rhand_img_buf.append(rhand_img_row)
            lhand_img_buf.append(lhand_img_row)

            # ── Preview ───────────────────────────────────────────────────
            if preview:
                ann = frame.copy()

                # body skeleton
                if p_res.pose_landmarks:
                    lms = p_res.pose_landmarks[0]
                    pts = [(int(lm.x * W), int(lm.y * H)) for lm in lms]
                    for a, b in POSE_CONNECTIONS:
                        cv2.line(ann, pts[a], pts[b], COLOR_BODY, 2)
                    for pt in pts:
                        cv2.circle(ann, pt, 3, COLOR_BODY, -1)

                # right hand — color depends on detection status (real-time,
                # interpolation status unknown here, so just show det/missing)
                if rhand_img_row is not None:
                    draw_hand(ann, rhand_img_row, HAND_CONNECTIONS,
                              COLOR_HAND_DET, W, H)
                if lhand_img_row is not None:
                    draw_hand(ann, lhand_img_row, HAND_CONNECTIONS,
                              COLOR_HAND_DET, W, H)

                r_ok = "DET" if rhand_img_row  else "---"
                l_ok = "DET" if lhand_img_row  else "---"
                b_ok = "DET" if p_res.pose_world_landmarks else "---"
                cv2.putText(ann,
                    f"body:{b_ok}  R:{r_ok}  L:{l_ok}  [{frame_idx}/{total}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.imshow(video_path.name, ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.monotonic() - t0
                speed   = frame_idx / elapsed
                print(f"  {frame_idx/total*100:5.1f}%  {speed:.1f} fps  "
                      f"ETA {(total-frame_idx)/speed:.0f}s", end="\r", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print()

    T          = len(body_buf)
    body       = np.stack(body_buf)
    rhand_raw  = np.stack(rhand_buf)
    lhand_raw  = np.stack(lhand_buf)
    visibility = np.stack(vis_buf)
    body_conf  = np.array(body_conf_buf,  dtype=np.float32)
    rhand_conf = np.array(rhand_conf_buf, dtype=np.float32)
    lhand_conf = np.array(lhand_conf_buf, dtype=np.float32)

    # ── Detection masks (before interpolation) ────────────────────────────
    body_det = np.any(body      != 0, axis=(1,2))
    r_det    = np.any(rhand_raw != 0, axis=(1,2))
    l_det    = np.any(lhand_raw != 0, axis=(1,2))

    # ── Interpolation ─────────────────────────────────────────────────────
    right_hand, r_interp = interpolate_missing(rhand_raw)
    left_hand,  l_interp = interpolate_missing(lhand_raw)

    # ── Stats ─────────────────────────────────────────────────────────────
    b_pct  = body_det.mean() * 100
    r_pct  = r_det.mean()    * 100
    l_pct  = l_det.mean()    * 100
    ri_pct = r_interp.mean() * 100
    li_pct = l_interp.mean() * 100

    print(f"  {T} frames")
    print(f"  body       detected: {b_pct:.0f}%  mean_conf: {body_conf[body_det].mean():.2f}" if body_det.any() else f"  body       detected: 0%")
    print(f"  right_hand detected: {r_pct:.0f}%  interpolated: {ri_pct:.0f}%"
          f"  → effective: {min(r_pct+ri_pct,100):.0f}%"
          + (f"  mean_conf: {rhand_conf[r_det].mean():.2f}" if r_det.any() else ""))
    print(f"  left_hand  detected: {l_pct:.0f}%  interpolated: {li_pct:.0f}%"
          f"  → effective: {min(l_pct+li_pct,100):.0f}%"
          + (f"  mean_conf: {lhand_conf[l_det].mean():.2f}" if l_det.any() else ""))

    # ── Confidence plot ───────────────────────────────────────────────────
    save_confidence_plot(
        stem      = video_path.stem,
        out_dir   = out_dir,
        fps       = fps,
        body_conf = body_conf,
        r_conf    = rhand_conf,
        l_conf    = lhand_conf,
        r_interp  = r_interp,
        l_interp  = l_interp,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(
        out_npz,
        body               = body,
        right_hand         = right_hand,
        left_hand          = left_hand,
        right_hand_raw     = rhand_raw,
        left_hand_raw      = lhand_raw,
        right_interpolated = r_interp,
        left_interpolated  = l_interp,
        body_conf          = body_conf,
        right_conf         = rhand_conf,
        left_conf          = lhand_conf,
        visibility         = visibility,
        fps                = fps,
        frame_count        = T,
        source             = str(video_path),
    )
    print(f"  → {out_npz.name}")
    return str(out_npz)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help=".mp4 или папка")
    parser.add_argument("--preview", action="store_true",
                        help="показывать окно с оверлеем (Q — пропустить)")
    args = parser.parse_args()

    download_models()

    inp = Path(args.input)
    videos = sorted(inp.glob("*.mp4")) if inp.is_dir() else [inp]
    for v in videos:
        extract(str(v), args.preview)


if __name__ == "__main__":
    main()