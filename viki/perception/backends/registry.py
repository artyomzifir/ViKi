"""
viki.perception.backends.registry
---------------------------------
Which model files each pose backend can use, and how to fetch them. Each backend
lists up to three tiers — ``quality`` / ``balance`` / ``speed`` — so the Extract
tab can offer a choice and a Download button for missing files.

Only ``mediapipe`` is wired to a working backend; the ``rtmpose`` / ``hamer`` /
``yolo`` entries are real download targets but their backends still raise
``NotImplementedError`` until implemented.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from viki import config

logger = logging.getLogger(__name__)


def _models_dir() -> Path:
    return Path(getattr(config, "MODELS_DIR", "models"))


# backend -> [ {id, tier, filename, url} ]  (tier order = quality, balance, speed)
MODELS: dict[str, list[dict]] = {
    "mediapipe": [
        {
            "id": "hand_landmarker",
            "tier": "balance",
            "filename": "hand_landmarker.task",
            "url": (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            ),
        },
    ],
    "rtmpose": [
        {
            "id": "rtmpose-l-hand",
            "tier": "quality",
            "filename": "rtmpose-l_simcc-hand5_pt-aic-coco_270e-256x256.onnx",
            "url": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "rtmpose-l_simcc-hand5_pt-aic-coco_270e-256x256-92f5a029_20230314.zip"
            ),
        },
        {
            "id": "rtmpose-m-hand",
            "tier": "balance",
            "filename": "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256.onnx",
            "url": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
            ),
        },
        {
            "id": "rtmpose-t-hand",
            "tier": "speed",
            "filename": "rtmpose-t_simcc-hand5_pt-aic-coco_210e-256x256.onnx",
            "url": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "rtmpose-t_simcc-hand5_pt-aic-coco_210e-256x256-4b21e6c8_20230320.zip"
            ),
        },
    ],
    "hamer": [
        {"id": "hamer", "tier": "quality", "filename": "hamer.ckpt",
         "url": "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"},
    ],
    "yolo": [
        {"id": "yolo11x-pose", "tier": "quality", "filename": "yolo11x-pose.pt",
         "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"},
        {"id": "yolo11m-pose", "tier": "balance", "filename": "yolo11m-pose.pt",
         "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"},
        {"id": "yolo11n-pose", "tier": "speed", "filename": "yolo11n-pose.pt",
         "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"},
    ],
}


def _entry(backend: str, model_id: str) -> dict | None:
    for m in MODELS.get(backend, []):
        if m["id"] == model_id:
            return m
    return None


def model_path(backend: str, model_id: str) -> Path | None:
    m = _entry(backend, model_id)
    return _models_dir() / m["filename"] if m else None


def is_present(backend: str, model_id: str) -> bool:
    p = model_path(backend, model_id)
    return bool(p and p.exists())


def list_models() -> dict[str, list[dict]]:
    """{backend: [{id, tier, present}]} for the UI."""
    return {
        b: [{"id": m["id"], "tier": m["tier"], "present": is_present(b, m["id"])}
            for m in ms]
        for b, ms in MODELS.items()
    }


def download(backend: str, model_id: str, report=None, log=None) -> str:
    """Fetch a model file into ``MODELS_DIR``. Returns the local path."""
    report = report or (lambda **k: None)
    log = log or (lambda m: None)
    m = _entry(backend, model_id)
    if m is None:
        raise ValueError(f"unknown model {backend}/{model_id}")
    dst = _models_dir() / m["filename"]
    if dst.exists():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"downloading {m['url']}")
    report(stage="download", model=model_id)

    def _hook(blocks, bs, total):
        if total > 0:
            report(stage="download", model=model_id,
                   frame=min(blocks * bs, total), total=total)

    urllib.request.urlretrieve(m["url"], tmp, _hook)
    tmp.rename(dst)
    log(f"saved {dst}")
    return str(dst)
