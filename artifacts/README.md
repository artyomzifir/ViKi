# ViKi — Mini UML (core pipeline)

A **compact** class diagram of the `viki` pipeline: capture → 3D skeleton →
prepared data → IK/retarget → robot dataset. Only the main classes and how they
connect — full detail lives in the code.

## Files

| File | Format | Viewable |
|---|---|---|
| `uml_viki.md` | Mermaid class diagram | Yes — GitHub / mermaid.live (no Graphviz) |
| `classes_viki.dot` | Graphviz `dot` source | `dot -Tpng classes_viki.dot -o classes_viki.png` |
| `classes_viki.png` | Rendered PNG | image viewer |
| `README.md` | this file | — |

The older `vikisim/` mirror package and its `classes_vikisim.dot` /
`packages_vikisim.dot` / `*.png` are **superseded** — they described a
pre-restructure simplified mirror. The new files describe the *actual* `viki/`
package.

## Main classes (core pipeline only)

- **`CameraManager`** (`viki.capture`) — owns `CameraBackend`s + workers, serves latest frames.
- **`CalibrationManager`** (`viki.calibration`) — per-camera intrinsics/extrinsics.
- **`SkeletonPipeline`** (`viki.skeleton`) — synced frames → 3D skeleton (uses `CameraManager` + `CalibrationManager`).
- **`SkeletonRecorder`** (`viki.skeleton.recorder`) — writes the 3D results to `rec-*.npz` on disk.
- **`PreparationPipeline`** (`viki.optimization.preparation`) — raw `rec-*.npz` → prepared `cln-*.npz` (smooth/interp/fuse).
- **`retarget`** (`viki.optimization.retarget`, IK/PINK) — prepared `cln-*.npz` → robot `.h5` dataset; invoked by `App`.
- **`App`** (`viki.server`) — FastAPI assembly; wires the managers + recorder and **calls** `retarget`.

## Key relationships

- `CameraManager` **owns** `CameraBackend`s/workers.
- `SkeletonPipeline` **reads** `CameraManager` frames and **uses** `CalibrationManager` extrinsics, then **feeds** `SkeletonRecorder`, which **writes** `rec-*.npz`.
- `PreparationPipeline` reads `rec-*.npz` and writes `cln-*.npz`; `retarget` (IK) turns `cln-*.npz` into robot `.h5`.
- `App` wires `CameraManager`, `CalibrationManager`, `SkeletonRecorder`, `PreparationPipeline`, and **calls** `retarget`.
