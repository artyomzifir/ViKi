# ViKi — Mini UML (core pipeline)

A **compact** class diagram of the `viki` pipeline: capture → 3D skeleton →
recorded → prepared → IK/retarget → robot dataset. Only the main classes and
how they connect — full detail lives in the code.

```mermaid
classDiagram
    direction BT

    CameraManager "1" *-- "many" CameraBackend : backends + workers
    SkeletonPipeline --> CameraManager : reads frames
    SkeletonPipeline --> CalibrationManager : uses extrinsics
    SkeletonPipeline --> SkeletonRecorder : feeds results
    SkeletonRecorder --> "rec-*.npz" : writes (raw recordings)
    PreparationPipeline --> "rec-*.npz" : reads raw
    PreparationPipeline --> "cln-*.npz" : prepared
    retarget --> "cln-*.npz" : consumes
    retarget --> "robot .h5" : dataset
    App --> SkeletonPipeline : runs
    App --> retarget : calls
    App ..> CameraManager
    App ..> CalibrationManager
    App ..> SkeletonRecorder
    App ..> PreparationPipeline

    class CameraManager {
        +latest_frame()
        +start/stop_camera()
    }
    class CalibrationManager {
        +extrinsics / intrinsics
    }
    class SkeletonPipeline {
        +process(group) -> 3D skeleton
    }
    class SkeletonRecorder {
        +record() -> rec-*.npz
    }
    class PreparationPipeline {
        +smooth_recording()
        +list_recordings()
    }
    class retarget {
        <<IK / PINK>>
        +retarget_from_poses()
    }
    class App {
        +wires managers + recorder
    }
```

Graphviz source: `classes_viki.dot` → `classes_viki.png` (`dot -Tpng
classes_viki.dot -o classes_viki.png`).
