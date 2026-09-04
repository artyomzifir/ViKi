# Stage 0 — rig time-sync measurement (multi-view triangulation task)

Status: **RESOLVED.** The Kinect pair was wired for hardware sync; a fresh take
measures inter-camera P95 = **2.5 ms** (was 20–31 ms software-only). The
triangulation task is unblocked.

## After wiring the hardware sync (2026-09-03, take `2026-09-03_16-25-03`)

| | median | P95 | max | groups > 16.7 ms |
|---|---|---|---|---|
| software-only (5 takes) | 7–16 ms | 20–31 ms | 22–39 ms | 8–44 % |
| **wired sync** | **1.6 ms** | **2.5 ms** | **5.0 ms** | **0 %** |

At 1 m/s hand speed the spatial error from desync drops from ~30 mm to ~2.5 mm —
well inside the budget for multi-view triangulation.

Cable direction: **kinect_1 SYNC OUT → kinect_0 SYNC IN**, i.e. master =
`kinect_1`, subordinate = `kinect_0` (the SDK's cable-presence check fails on
*both* devices when the roles are assigned the wrong way round — that is how the
direction was determined). `KINECT_SYNC` in the config is set accordingly;
`record.py` stamps `meta["kinect_sync"]` so a take is self-identifying.

`std_offset_us` (offset-from-tick jitter, a separate quantity) also roughly
halved, 9 ms → 4.8 ms. The `drift_ms_per_min` in `sync_stats.json` for a 10 s
take is a meaningless fit (too short); ignore `sync_bounded: false` on short
clips.

---

## Original measurement (software-only) — kept for the record

Measured from the existing `raw/timestamps.json` of every recorded episode
(`data/_sync_stage0.py`, since removed). No pipeline changes.

## What was measured

Each synced frame-group carries `sync_us` (the shared host-monotonic tick) and
`offsets_us[dev]` (how far the frame `nearest_to()` picked for that camera sits
from the tick). The number that matters for triangulation is the **inter-camera
gap per group**, `|offset_kinect_0 − offset_kinect_1|` — the real time
difference between the two frames a triangulator would pair. Spatial error
`Δx ≈ v · Δt`; at 1 m/s hand speed, 30 ms ⇒ 30 mm.

## Results (5 episodes, incl. the newest clean `pick_up` take)

| episode | groups | inter-cam median | **P95** | max | groups > 16.7 ms | Kinect sync |
|---|---|---|---|---|---|---|
| 2026-09-01_10-51-08 | 599 | 16.4 ms | 19.9 ms | 24.3 ms | 44 % | software-only |
| 2026-09-02_10-51-10 | 610 | 11.7 ms | 21.0 ms | 25.1 ms | 26 % | software-only |
| 2026-09-02_11-46-51 | 898 | 7.1 ms | 28.6 ms | 38.7 ms | 13 % | software-only |
| 2026-09-03_10-10-07 | 899 | 1.3 ms | 30.6 ms | 34.8 ms | 8.5 % | software-only |
| **2026-09-03_15-52-12** (`pick_up`, clean) | 898 | **14.9 ms** | **21.5 ms** | 22.2 ms | **44 %** | software-only |

Per-device offset-from-tick: mean ≈ −16 ms (a nearest-frame phase artefact,
harmless), P95 |offset| ≈ 30–32 ms. **Clock drift is not a factor** — linear fits
give `R² ≈ 0.03–0.04`; the offset is frame-phase jitter, not drift.

## Findings

1. **The Kinect pair is not hardware-synced in any recording.**
   `KINECT_SYNC = {}` in both config files ⇒ `routes/cameras.py::_wired_sync_for`
   returns `(0, 0)` (standalone) for every device. The wired master/subordinate
   path exists (`viki/cameras/sync.py` docstring, `_wired_sync_for`) but was
   never configured. So the ~15–30 ms inter-camera gap is a **real exposure-time
   skew**, not just host-arrival jitter — two free-running 30 fps cameras have a
   uniform phase offset in ±16.7 ms, and arrival jitter stacks on top.

2. **`MultiCameraSync.max_offset_us` default is 150000 (150 ms)**, while the
   class docstring says "Default: half a 30 fps frame (≈ 16.7 ms)".
   `record.py` constructs `MultiCameraSync(self._mgr, sync_fps=fps)` with no
   override, so 150 ms is in force. It is only a *rejection/warn* threshold
   (`nearest_to` still returns the closest frame), so groups are still formed
   correctly in normal operation — but a stalled camera's stale frame would be
   accepted silently for up to 150 ms.

3. **Device timestamps are not recorded** anywhere.
   `k4a_image_get_device_timestamp_usec` and the RealSense equivalent do not
   appear in `viki/cameras` or `viki/perception`. The numbers above are
   host-arrival time; without device timestamps we cannot measure the true
   exposure skew directly. (Not fixing now — logged as a known gap.)

## Decision

Per the task's own rule ("если десятки [мс] — синхронизация становится
приоритетом выше триангуляции"): **P95 is 20–31 ms across every episode, so the
multi-view triangulation task is paused.** Triangulating two views that are
15–30 ms apart on a moving hand would inject 15–30 mm of error — larger than the
geometry gain triangulation is meant to deliver over the current mono lift.

### Blocking work, in order

1. **Enable Azure Kinect wired sync.** Connect the 3.5 mm sync cable between the
   two Kinects (subordinate ← master), set
   `KINECT_SYNC = {"master": "kinect_0", "subordinates": ["kinect_1"],
   "subordinate_delay_us": 160}` (values per SETUP_GUIDE), start the subordinate
   before the master. This drops the Kinect-pair exposure skew to the hardware
   trigger precision (sub-millisecond).
2. **Re-record** the reference episode and **re-run this measurement.** Expect
   inter-camera P95 to fall to single-digit µs–low-ms.
3. **Fix `max_offset_us`.** Bring the default and the docstring into agreement at
   a value set from the re-measured P95 with margin (≈ 5 ms is a real
   stalled-camera guard once wired sync is on; keep it well under a frame).
4. Only then resume **Stage 1** (persist 2-D observations) of the triangulation
   task.

### RealSense

Was already out of the geometry scope until Stage 0 cleared it. It stays a
policy-only observation: its inter-camera gap vs the Kinects ran to a P95 of
~55 ms / max ~150 ms (it has no wired-sync path to the Kinects at all).
