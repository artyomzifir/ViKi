# viki.perception.backends — hand-pose models

One abstraction, `HandPoseBackend`, behind which any 21-keypoint hand model
plugs in. ViKi tracks a **single hand**; the caller picks which one.

## The contract

```python
class HandPoseBackend(ABC):
    name: str
    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None: ...
    def close(self) -> None: ...
```

`HandDetection.points` is `{LM(0..20): (u, v)}` in pixels, `lm_z_rel` is a
`(21,)` model-relative z (not metric — depth lifting ignores it when measured
depth is present), `confidence` is a scalar in `[0, 1]`. Return `None` when the
requested hand is absent. Stateful implementations create **one instance per
camera**.

## Files

| file | status | notes |
|---|---|---|
| `base.py` | — | the ABC |
| `mediapipe.py` | **working** | MediaPipe Tasks `HandLandmarker`, image/video modes. Self-contained (the old `mediapipe_base.py` plumbing is inlined). |
| `rtmpose.py` | stub | RTMPose-Hand (MMPose / rtmlib). 21 kpts already in MediaPipe topology → 1:1 `LM` map. |
| `hamer.py` | stub | HaMeR (MANO mesh). Project the 3-D joints to pixels; `lm_z_rel` can carry real root-relative z. |
| `yolo.py` | stub | YOLO-Pose (Ultralytics). Use the visibility channel as the `v` factor of the fusion weight. |

## Selection

`load_backend(cfg.POSE_BACKEND, **kw)` — registry in `__init__.py::BACKENDS`,
imported lazily so a missing heavy dependency only errors when that backend is
actually used.

## Adding one

Add a `HandPoseBackend` subclass in a new file, map its keypoint order to `LM`
in `detect`, and add `"name": "module:Class"` to `BACKENDS`.
