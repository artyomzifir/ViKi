# viki — package map

One directory per arrow of the pipeline. Stages talk to each other only through
`viki.contracts` (DTOs, enums, Protocols, artifact schemas) and through file
artifacts inside an **episode directory**.

```
record        cameras/      live RGB-D  ->  episodes/<id>/raw/
extract       perception/   raw/        ->  rec.npz      per-camera hand landmark trajectories
prepare       prepare/      rec.npz     ->  cln.npz      fused + smoothed wrist trajectory + EE pose + gripper
retarget      retarget/     cln.npz     ->  plan.h5      synthesised robot joint trajectory
replay        replay/       plan.h5     ->  replay.h5    proprioception attained on hardware   [stub]
label         labeling.py   -> meta.json["labels"]       task string / phase segments / outcome
export        export/       episodes/*  ->  datasets/<name>/   LeRobot dataset                 [stub]

calibration/  intrinsics + extrinsics (board -> world), a side input to perception
render/       depth colourise, MJPEG, 3-D matplotlib views — no FastAPI, no hardware
server/       transport only: FastAPI over the offline stages + camera preview
cli.py        `viki record|extract|prepare|retarget|replay|label|export|run`
```

## Cross-cutting

| module | purpose |
|---|---|
| `contracts.py` | every cross-stage DTO, `LM` enum, Protocols, `Episode`/`EpisodeLabels`, `*_KEYS` schema tuples |
| `config.py` | `Config` (frozen) + `load()`; UPPER_SNAKE keys mirror `data/*_configuration.json` |
| `episode.py` | episode-directory helpers: `new_episode`, meta/status r/w, `mark_stage` |
| `dsp.py` | Savitzky-Golay + NaN interpolation, shared by `prepare` and `retarget` |
| `gripper.py` | `Gripper` ABC + `BinaryGripper` (`cfg.GRIPPER`) |

## Rules

- A stage imports `viki.contracts` and the public `__init__` of a neighbour — never a neighbour's internal module.
- `render/` depends only on `contracts` + numpy/cv2/matplotlib.
- `server/` depends on everything; nothing depends on `server/`.
- There is **no live pipeline**: record scenes, then extract/prepare/retarget offline.

Modules tagged `[stub]` are contract-complete but not implemented; each cites the
thesis section it must satisfy (`paper/src/ch3_methodology.tex`).
