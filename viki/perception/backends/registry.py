"""
viki.perception.backends.registry
---------------------------------
The flat list of hand-pose models the Extract tab can pick from. Every model is
a 21-keypoint hand estimator (MediaPipe topology) under an Apache-2.0 / MIT
licence — no MANO, no mesh.

``impl`` selects the runtime:
  * ``mediapipe``      — MediaPipe Tasks HandLandmarker (weights auto-download)
  * ``rtmpose``        — RTMPose-Hand SimCC via rtmlib on ONNX Runtime
                         (detector + pose ONNX auto-download to ~/.cache/rtmlib)
  * ``mmpose-heatmap`` — a top-down MMPose heatmap model as a plain ONNX in
                         ``MODELS_DIR``. mmpose does **not** publish these as
                         ONNX, so the file is user-supplied: convert with
                         ``mmdeploy tools/deploy.py`` or grab it from the
                         OpenMMLab Deploee, then drop it in ``models/``.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from viki import config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mediapipe"

# rtmlib downloads its own ONNX; RTMPose-Hand shares this RTMDet-nano hand
# detector across every rtmpose / mmpose-heatmap model (they are all top-down).
RTM_DET_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
)

# Numbers are COCO-WholeBody-Hand val (PCK@0.2 / AUC / EPE px) from the MMPose
# model zoo; RTMPose-m is hand5. GFLOPs are approximate.
MODELS: list[dict] = [
    {
        "id": "mediapipe", "label": "MediaPipe HandLandmarker", "impl": "mediapipe",
        "license": "Apache-2.0", "pck": 0.80, "gflops": None,
        "note": "21 pts + relative z + handedness; weights auto-download",
    },
    {
        "id": "rtmpose-m-hand5", "label": "RTMPose-m Hand5", "impl": "rtmpose",
        "license": "Apache-2.0", "pck": 0.815, "auc": 0.839, "gflops": 2.58,
        "pose_url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        ),
    },
    {
        "id": "hrnetv2-w18-hand", "label": "HRNetv2-w18", "impl": "mmpose-heatmap",
        "license": "Apache-2.0", "pck": 0.813, "auc": 0.840, "epe": 4.39, "gflops": 4.3,
        "filename": "hrnetv2_w18_coco_wholebody_hand_256x256.onnx",
        "input_size": [256, 256], "heatmap_size": [64, 64],
    },
    {
        "id": "hourglass52-hand", "label": "Hourglass-52", "impl": "mmpose-heatmap",
        "license": "Apache-2.0", "pck": 0.804, "auc": 0.835, "epe": 4.54, "gflops": 28.0,
        "filename": "hourglass52_coco_wholebody_hand_256x256.onnx",
        "input_size": [256, 256], "heatmap_size": [64, 64],
    },
    {
        "id": "scnet50-hand", "label": "SCNet-50", "impl": "mmpose-heatmap",
        "license": "Apache-2.0", "pck": 0.803, "auc": 0.834, "epe": 4.55, "gflops": 5.6,
        "filename": "scnet50_coco_wholebody_hand_256x256.onnx",
        "input_size": [256, 256], "heatmap_size": [64, 64],
    },
    {
        "id": "res50-hand", "label": "ResNet-50", "impl": "mmpose-heatmap",
        "license": "Apache-2.0", "pck": 0.800, "auc": 0.833, "epe": 4.64, "gflops": 5.5,
        "filename": "res50_coco_wholebody_hand_256x256.onnx",
        "input_size": [256, 256], "heatmap_size": [64, 64],
    },
]

_PUBLIC_KEYS = ("id", "label", "impl", "license", "pck", "auc", "epe", "gflops", "note")


def _models_dir() -> Path:
    return Path(getattr(config, "MODELS_DIR", "models"))


def get(model_id: str | None) -> dict | None:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None


def _rtmlib_cache_dir() -> Path:
    return Path.home() / ".cache" / "rtmlib"


def _rtm_onnx_name(url: str) -> str:
    """rtmlib names the extracted model after the zip: <basename>.onnx."""
    return Path(url).name[:-4] + ".onnx" if url.endswith(".zip") else Path(url).name


def _rtmlib_has(url: str) -> bool:
    d = _rtmlib_cache_dir()
    if not d.is_dir():
        return False
    want = _rtm_onnx_name(url)
    return any(p.name == want for p in d.rglob("*.onnx"))


def is_present(model_id: str) -> bool:
    m = get(model_id)
    if m is None:
        return False
    if m["impl"] == "mediapipe":
        return (_models_dir() / "hand_landmarker.task").exists()
    if m["impl"] == "rtmpose":
        return _rtmlib_has(m["pose_url"]) and _rtmlib_has(RTM_DET_URL)
    if m["impl"] == "mmpose-heatmap":
        return (_models_dir() / m["filename"]).exists() and _rtmlib_has(RTM_DET_URL)
    return False


def list_models() -> list[dict]:
    """Flat list for the Extract-tab model picker."""
    out = []
    for m in MODELS:
        row = {k: m.get(k) for k in _PUBLIC_KEYS}
        row["present"] = is_present(m["id"])
        # mmpose-heatmap ONNX is user-supplied (mmpose publishes no ONNX for it)
        row["downloadable"] = m["impl"] != "mmpose-heatmap"
        out.append(row)
    return out


def model_path(model_id: str) -> Path | None:
    m = get(model_id)
    if m and m["impl"] == "mmpose-heatmap":
        return _models_dir() / m["filename"]
    return None


def download(model_id: str, report=None, log=None) -> str:
    """Fetch a model's weights. Returns a path/dir. Raises for the
    mmpose-heatmap models (no published ONNX)."""
    report = report or (lambda **k: None)
    log = log or (lambda m: None)
    m = get(model_id)
    if m is None:
        raise ValueError(f"unknown model {model_id!r}")
    report(stage="download", model=model_id)

    if m["impl"] == "mediapipe":
        from viki.perception.backends.mediapipe import _ensure_model

        log("fetching MediaPipe hand_landmarker.task")
        return _ensure_model(str(_models_dir()))

    if m["impl"] == "rtmpose":
        # constructing Hand with explicit det/pose URLs makes rtmlib fetch +
        # unzip both into ~/.cache/rtmlib.
        log(f"fetching RTMPose-Hand weights for {model_id}")
        from rtmlib import Hand

        Hand(
            mode="lightweight",
            det=RTM_DET_URL, det_input_size=(320, 320),
            pose=m["pose_url"], pose_input_size=(256, 256),
            backend="onnxruntime", device="cpu",
        )
        log(f"rtmlib weights ready in {_rtmlib_cache_dir()}")
        return str(_rtmlib_cache_dir())

    if m["impl"] == "mmpose-heatmap":
        raise RuntimeError(
            f"{m['label']}: MMPose publishes no ONNX for this checkpoint. Convert "
            f"it with `mmdeploy tools/deploy.py` (pose-detection_onnxruntime_static "
            f"config) or download the SDK model from the OpenMMLab Deploee, then "
            f"put {m['filename']} in {_models_dir()}/."
        )

    raise ValueError(f"no downloader for impl {m['impl']!r}")
