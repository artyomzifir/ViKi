"""
viki.perception.backends
------------------------
Hand-pose model backends behind one abstraction (:class:`HandPoseBackend`).

    backend = load_backend(model_id)          # id from registry.MODELS
    det = backend.detect(prepared_frame, hand="right")

Every model is a 21-keypoint hand estimator (MediaPipe topology). ``impl`` in
the registry entry selects the runtime class below.
"""

from __future__ import annotations

from viki.perception.backends.base import HandPoseBackend

__all__ = ["HandPoseBackend", "load_backend"]

# impl -> "module:ClassName", imported lazily so a missing heavy dependency
# (mediapipe, rtmlib, onnxruntime) only errors when that model is actually used.
_IMPL: dict[str, str] = {
    "mediapipe": "viki.perception.backends.mediapipe:MediaPipeHandBackend",
    "rtmpose": "viki.perception.backends.rtmpose:RTMPoseHandBackend",
    "mmpose-heatmap": "viki.perception.backends.mmpose_heatmap:MMPoseHeatmapBackend",
    "hamer": "viki.perception.backends.hamer:HaMeRHandBackend",  # stub, not in MODELS
}


def load_backend(model: str = "mediapipe", **kwargs) -> HandPoseBackend:
    """Instantiate the backend for a registry model id."""
    from viki.perception.backends import registry

    entry = registry.get(model)
    if entry is None:
        raise ValueError(
            f"unknown model {model!r}; known: "
            + ", ".join(m["id"] for m in registry.MODELS)
        )
    spec = _IMPL.get(entry["impl"])
    if spec is None:
        raise ValueError(f"no backend for impl {entry['impl']!r}")

    import importlib

    module_name, cls_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(model_entry=entry, **kwargs)
