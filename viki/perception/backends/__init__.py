"""
viki.perception.backends
------------------------
Hand-pose model backends behind one abstraction (:class:`HandPoseBackend`).
``mediapipe`` works; ``rtmpose`` / ``hamer`` / ``yolo`` are stubs.

    backend = load_backend(cfg.POSE_BACKEND)
    det = backend.detect(prepared_frame, hand="right")
"""

from __future__ import annotations

from viki.perception.backends.base import HandPoseBackend

__all__ = ["HandPoseBackend", "BACKENDS", "load_backend"]

# name -> "module:ClassName", imported lazily so a missing heavy dependency
# (mediapipe, rtmlib, hamer, ultralytics) only errors when that backend is used.
BACKENDS: dict[str, str] = {
    "mediapipe": "viki.perception.backends.mediapipe:MediaPipeHandBackend",
    "rtmpose": "viki.perception.backends.rtmpose:RTMPoseHandBackend",
    "hamer": "viki.perception.backends.hamer:HaMeRHandBackend",
    "yolo": "viki.perception.backends.yolo:YoloHandBackend",
}


def load_backend(name: str = "mediapipe", **kwargs) -> HandPoseBackend:
    """Instantiate a backend by name (``cfg.POSE_BACKEND``)."""
    try:
        spec = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown pose backend {name!r}; known: {', '.join(sorted(BACKENDS))}"
        ) from None
    import importlib

    module_name, cls_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
