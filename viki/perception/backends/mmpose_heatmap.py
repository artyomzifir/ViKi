"""
viki.perception.backends.mmpose_heatmap
---------------------------------------
Top-down MMPose hand models (HRNetv2 / Hourglass / SCNet / ResNet on
COCO-WholeBody-Hand) run as a plain ONNX on ONNX Runtime:

    RTMDet-nano hand bbox  ->  square crop -> 256x256  ->  <model>.onnx
    ->  (1, 21, Hh, Ww) heatmaps  ->  argmax + quarter-offset refine  ->  21 kpts

MMPose publishes no ONNX for these checkpoints, so ``models/<file>.onnx`` is
user-supplied — convert with ``mmdeploy tools/deploy.py`` (a
``pose-detection_onnxruntime_static`` config) or grab the SDK model from the
OpenMMLab Deploee. 21 keypoints, MediaPipe topology, 1:1 to :class:`LM`;
``lm_z_rel`` is zeros (the depth lift uses measured depth).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from viki.contracts import HAND_LM_COUNT, Hand, HandDetection, LM, PreparedFrame
from viki.perception.backends.base import HandPoseBackend
from viki.perception.backends.registry import RTM_DET_URL, model_path

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _providers(device: str | None):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if (device or "").lower() == "cpu":
        return ["CPUExecutionProvider"]
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]


def _square_bbox(x0, y0, x1, y1, w, h, pad=1.25):
    """bbox -> a padded square inside the image; returns (ix0, iy0, side)."""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * pad
    ix0 = int(round(cx - side / 2.0))
    iy0 = int(round(cy - side / 2.0))
    side = int(round(side))
    # clip origin so the crop stays in-frame (shrink side if needed)
    ix0 = max(0, min(ix0, w - 1))
    iy0 = max(0, min(iy0, h - 1))
    side = max(1, min(side, w - ix0, h - iy0))
    return ix0, iy0, side


def _decode_heatmaps(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(K, Hh, Ww) -> (K, 2) xy in heatmap px + (K,) peak value."""
    k, hh, ww = hm.shape
    flat = hm.reshape(k, -1)
    idx = np.argmax(flat, axis=1)
    conf = flat[np.arange(k), idx]
    ys, xs = np.divmod(idx, ww)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    # quarter-offset toward the brighter neighbour (DARK-pose lite)
    for i in range(k):
        x, y = int(xs[i]), int(ys[i])
        if 1 < x < ww - 1:
            coords[i, 0] += 0.25 * np.sign(hm[i, y, x + 1] - hm[i, y, x - 1])
        if 1 < y < hh - 1:
            coords[i, 1] += 0.25 * np.sign(hm[i, y + 1, x] - hm[i, y - 1, x])
    return coords, conf


class MMPoseHeatmapBackend(HandPoseBackend):
    """A top-down MMPose heatmap hand model as ONNX. One instance per camera."""

    name = "mmpose-heatmap"

    def __init__(
        self,
        *,
        model_entry: dict,
        mode: str = "video",  # parity; unused
        min_confidence: float = 0.5,
        device: str | None = None,
        **_ignored,
    ) -> None:
        try:
            import onnxruntime as ort
            from rtmlib import RTMDet
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mmpose-heatmap backend needs `onnxruntime` + `rtmlib`"
            ) from exc

        onnx = model_path(model_entry["id"])
        if onnx is None or not onnx.exists():
            raise RuntimeError(
                f"{model_entry['label']}: MMPose publishes no ONNX for this "
                f"checkpoint. Convert it with `mmdeploy tools/deploy.py` or grab "
                f"the SDK model from the OpenMMLab Deploee, then put "
                f"{model_entry['filename']} in models/."
            )

        self._min_conf = float(min_confidence)
        self._in_w, self._in_h = model_entry.get("input_size", [256, 256])
        self._warned_multi = False
        provs = _providers(device)
        self._sess = ort.InferenceSession(str(onnx), providers=provs)
        self._inp = self._sess.get_inputs()[0].name
        self._det = RTMDet(
            RTM_DET_URL, model_input_size=(320, 320),
            backend="onnxruntime", device="cpu" if "CUDA" not in provs[0] else "cuda",
        )
        logger.info(
            "mmpose-heatmap %s: %s on %s", model_entry["id"], onnx.name, provs[0]
        )

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        bgr = np.ascontiguousarray(frame.rgb[:, :, ::-1])
        h, w = bgr.shape[:2]

        boxes = self._det(bgr)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, boxes.shape[-1]) if len(boxes) else np.empty((0, 4), np.float32)
        if len(boxes) == 0:
            return None
        if boxes.shape[1] >= 5:
            boxes = boxes[np.argsort(-boxes[:, 4])]
        elif len(boxes) > 1 and not self._warned_multi:
            logger.warning("mmpose-heatmap: %d hand boxes on %s; using the first "
                           "as %r", len(boxes), frame.device_id, hand)
            self._warned_multi = True
        x0, y0, x1, y1 = boxes[0, :4]

        ix0, iy0, side = _square_bbox(x0, y0, x1, y1, w, h)
        crop = bgr[iy0:iy0 + side, ix0:ix0 + side]
        if crop.size == 0:
            return None
        inp = cv2.resize(crop, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0          # BGR->RGB, 0..1
        inp = (inp - _MEAN) / _STD
        inp = np.transpose(inp, (2, 0, 1))[None]                  # NCHW

        hm = self._sess.run(None, {self._inp: inp.astype(np.float32)})[0][0]  # (21,Hh,Ww)
        coords, conf = _decode_heatmaps(hm)
        hh, ww = hm.shape[1:]
        # heatmap px -> crop px -> image px
        coords[:, 0] = ix0 + coords[:, 0] / ww * side
        coords[:, 1] = iy0 + coords[:, 1] / hh * side

        mean_c = float(conf.mean())
        if mean_c < self._min_conf:
            return None
        points = {LM(i): coords[i].astype(np.float32) for i in range(HAND_LM_COUNT)}
        return HandDetection(
            points=points,
            lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
            confidence=mean_c,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

    def close(self) -> None:
        self._sess = None
        self._det = None
