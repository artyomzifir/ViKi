# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

It has two parts:

1. **[Using the project](#part-1--using-the-project)** — how to run ViKi and get a
   dataset out of it.
2. **[Developing the project](#part-2--developing-the-project)** — how the code is
   laid out and the rules that keep it that way.

The user-facing overview lives in [`README.md`](README.md); hardware bring-up in
[`SETUP_GUIDE.md`](SETUP_GUIDE.md). Don't duplicate either here.

---

## What this is

ViKi turns multi-view RGB-D video of a human doing a manipulation task into a
robot-ready demonstration dataset. One directory per arrow of the pipeline; each
stage writes a durable artifact into an **episode directory**:

```
record     cameras/      live RGB-D  ->  episodes/<id>/raw/
extract    perception/   raw/        ->  rec.npz     per-camera hand landmark trajectories
prepare    prepare/      rec.npz     ->  cln.npz     fused + smoothed + palm pose + gripper
retarget   retarget/     cln.npz     ->  plan.h5     targets -> adapter -> TCP -> robot joints
replay     replay/       plan.h5     ->  replay.h5   proprioception on hardware        [stub]
label      labeling.py   ->  meta.json["labels"]     task / phase segments / outcome
export     export/       episodes/*  ->  datasets/<name>/
```

**There is no live pipeline.** Record scenes first, then extract/prepare/retarget
offline. `data/` and `models/` are gitignored — no recordings, calibration, URDFs
or weights are in the repository.

---

# Part 1 — Using the project

## Everything runs in Docker

The app talks to hardware SDKs (`pyrealsense2`, `libk4a.so`) installed in the
image, not on the host. Run and test through Docker Compose.

```bash
sudo ./scripts/host_setup.sh              # once per host: Docker, udev rules, groups
docker compose up --build                 # web UI + API on :8000 (then just: docker compose up)
docker compose run --rm terminal          # debug shell inside the container
docker compose run --rm test              # full test suite
docker compose run --rm cli <verb> ...    # one pipeline stage (viki <verb> ...)
```

One `docker-compose.yml`, one image (Dockerfile `test` target). Services: `web`
(default, `up`), and `test` / `cli` / `terminal` behind the `tools` profile,
meant for `run`. `test` and `cli` append their args to `pytest` / `viki`.

The server serves UI + API at `http://localhost:8000` (`network_mode: host`).
`viki/` is bind-mounted, so code edits apply on container restart — no rebuild
unless `pyproject.toml` changes.

Kinect has host-level prerequisites (GRUB `usbcore.usbfs_memory_mb=1000`,
`xhost +local:`, a separate 10 Gbps USB hub per Kinect, sync cable). Read
`SETUP_GUIDE.md` before touching camera bring-up.

## The CLI

```bash
viki record   ...                 # capture a synced RGB-D scene into a new episode
viki extract  <episode>           # raw/ -> rec.npz
viki prepare  <episode>           # rec.npz -> cln.npz
viki retarget <episode>           # cln.npz -> plan.h5
viki replay   <episode>           # plan.h5 -> replay.h5            [stub]
viki label    <episode> ...       # get/set episode labels
viki export   <episode>... --out <dir> [--format trajectory|lerobot]
viki run      <episode>           # extract -> prepare -> retarget -> replay
viki cloud    <episode>           # raw/ -> cloud/ (viewer artifact only)
viki hand-fit <episode>           # batch capsule-hand fit, appends hand_fit_* to cln.npz
viki viz      <episode>           # headless 3-D figure (rec|cln)
```

`export` defaults to `--format trajectory`: a self-contained `.npz` bundle plus a
JSON manifest, **numpy only**. `--format lerobot` needs `viki[export]` (lerobot,
torch) and stricter eligibility. See [`viki/export/README.md`](viki/export/README.md).

## The web UI

Tabs: Cameras, Calibration, Record, Extract, Viewer, Retarget, Export. The Extract
tab drives perception over one / several / a whole dataset of episodes through a
background FIFO job queue (`viki/server/jobs.py`, one worker).

## Configuration

All tunables live in `data/user_configuration.json`, copied from
`data/default_configuration.json` on first run. `viki/config.py` reads that file
**once at import** and injects every key into its module globals, so code does
`from viki.config import RETARGET_IK_SOLVER`.

Changing config at runtime means editing the JSON and restarting — the
`/api/config` routes plus `/api/restart` do exactly that.

## What you must supply yourself

Robot and gripper URDFs (fetched by `robot_descriptions` into `models/` on the
first retarget), hand-pose weights (MediaPipe/RTMPose auto-download; the
mmpose-heatmap ONNX files are user-converted), camera calibration, recordings,
and the config file. The README's
[What you must supply yourself](README.md#what-you-must-supply-yourself) table is
the authoritative list.

---

# Part 2 — Developing the project

## Package map

Nine packages under `viki/`, **each with its own README — read those**, they carry
hardware quirks and API tables not repeated here.

| package | role |
|---|---|
| `cameras/` | the only package that touches camera SDKs |
| `calibration/` | intrinsics + extrinsics (chessboard / ChArUco), a side input to perception |
| `perception/` | raw/ → rec.npz: detection, multi-view triangulation, backends, hand fit |
| `prepare/` | rec.npz → cln.npz: interpolate, fuse, Savitzky–Golay, EE pose, gripper |
| `retarget/` | cln.npz → plan.h5: PINK / Pinocchio whole-trajectory IK |
| `replay/` | plan.h5 → replay.h5 — **stub**, contract-complete only |
| `export/` | episodes → dataset: trajectory bundle (works) and LeRobot (partial) |
| `render/` | pure pixel work — depth colourise, MJPEG, matplotlib. No FastAPI, no hardware |
| `server/` | transport only: FastAPI over the offline stages + camera preview |

Cross-cutting modules: `contracts.py` (every cross-stage DTO, the `LM` enum,
Protocols, `Episode`, `*_KEYS` schema tuples), `config.py`, `episode.py`
(episode-directory helpers, `mark_stage`), `dsp.py`, `gripper.py`, `labeling.py`,
`datasets.py`, `cli.py`.

`cameras/` owns one daemon worker thread per active camera, each writing into a
bounded ring buffer under a lock. All consumers **pull** via `latest_frame()` /
`nearest_frame()` — there is no push pub/sub. The Kinect backend is raw ctypes
over `libk4a.so`; its README lists constraints that must not change without
hardware in hand (`align_depth_to_color` is bugged and disabled,
`WFOV_UNBINNED` capped at 15 fps, `stop()` sleeps 2 s).

## Rules that must not be broken

- A stage imports `viki.contracts` and the **public `__init__`** of a neighbour —
  never a neighbour's internal module.
- `render/` depends only on `contracts` + numpy/cv2/matplotlib, so it stays
  testable without hardware.
- `server/` depends on everything; **nothing depends on `server/`**.
- **Server layering, do not skip levels:** `routes/` (thin handlers) → `deps.py`
  (DI) → `streams.py` (poll + timing) → `render/` (pure pixel work) →
  `config.py` (constants). Handlers delegate; they don't compute pixels or camera
  frames.
- `app.py` builds every long-lived object in `lifespan` and stores it on
  `app.state`; handlers get them via `Depends(...)` from `deps.py`.
- Modules tagged `[stub]` are contract-complete but unimplemented, and each cites
  the thesis section it must satisfy (`paper/thesis/ch3_methodology.tex`). Don't
  make a stub silently pretend to work.

## API routers

All mounted under `/api`. Router ↔ stage mapping, since it is easy to guess wrong:

| prefix | what |
|---|---|
| `/api/cameras`, `/api/calibration`, `/api/record` | live hardware: preview, calibration, capture |
| `/api/skeleton` | live hand estimation (preview only — not a pipeline stage) |
| `/api/pipeline` | the offline stages: extract → prepare, job queue, episode artifacts, cloud |
| `/api/episodes`, `/api/datasets` | episode + dataset listing, retarget jobs and viz |
| `/api/export`, `/api/label`, `/api/replay`, `/api/models` | export, labels, replay stub, model registry |

## Adding a tunable

Add the key to **both** `data/default_configuration.json` and
`data/user_configuration.json`, then declare its type annotation in
`viki/config.py`. All three, or it silently won't exist.

## Frontend

`viki/server/static/` — plain HTML/CSS/JS, **no build step**. `index.html` plus one
JS module per panel (`cameras.js`, `calibration.js`, `record.js`, `perception.js`,
`retarget.js`, `export.js`, …), served directly by FastAPI.

The one exception: the **Viewer** and **Extract** tabs share `scene3d.js`, a
three.js (WebGL) scene controller rendering the ChArUco world, the per-frame
coloured point cloud, per-camera and fused hand skeletons, wrist trajectory, palm
triad, gripper marker, the robot reach envelope and camera frusta, with layer
toggles and a transport. three.js is the **only** vendored dependency
(`static/js/vendor/`, resolved by an importmap in `index.html`); a dense cloud
(~10^5 points/frame) is not viable on a hand-rolled 2-D canvas. Everything else
stays plain, no-build JS.

## Tests

`tests/unit_tests/**` — pytest, hardware-independent, one directory per package.

```bash
docker compose run --rm test                                            # all
docker compose run --rm test tests/unit_tests/perception/test_x.py::t   # one
```

`tests/unit_tests/e2e/test_pipeline_smoke.py` walks a synthetic episode
rec.npz → prepare → plan.h5 → replay(dryrun) → label → export. PINK/Pinocchio
ships in the image, so the retarget leg really runs in-container.

Run the **full suite** after every change; it is fast and it is the only thing
standing between a refactor and a silently broken stage. Check any touched JS with
`node --check`.

## Working conventions

- **Commits carry no `Co-Authored-By` / agent trailers** in this repository.
- Measure before claiming. There is no ground truth in this setup, so a
  millimetre figure is a self-consistency or tracking-error measure — say which.
  Protocols and results go in `docs/`, and a retracted result gets retracted in
  writing, not deleted.
- Protected baselines under `intermediates/baselines/<profile>/` are SHA-256
  guarded on purpose; don't route around the guard to make a comparison pass.
- Perception profiles in `viki/perception/profiles.py` are **code-owned and
  immutable**. Add a new named profile rather than editing a frozen one.
- A stopped container still holds its RAM: `docker rm -f viki-terminal-run-*`
  after killing a long job, or the next batch gets OOM-killed.

## Documentation

| | |
|---|---|
| [`docs/math.md`](docs/math.md) | the full mathematical description of the pipeline |
| [`docs/robust_retarget_experiments.md`](docs/robust_retarget_experiments.md) | measurement protocols and results (E6–E14) |
| [`docs/object_centric_decision.md`](docs/object_centric_decision.md) | why object-relative representation is not built yet |
| [`docs/paper_code_divergences.md`](docs/paper_code_divergences.md) | where the thesis text and the code disagree |
| [`docs/licensing_todo.md`](docs/licensing_todo.md) | open licence audit items |
