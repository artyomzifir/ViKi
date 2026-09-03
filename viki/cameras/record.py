"""
viki.cameras.record
-------------------
Record synchronised RGB-D scenes into an episode directory.

Pipeline stage 1. Writes ``<dataset>/<id>/raw/`` — one colour ``.mp4`` and one
folder of raw ``uint16`` depth ``.npy`` per camera, plus ``timestamps.json``, the
SDK-reported intrinsics + active extrinsics in force at capture time, and the
offline colour↔depth calibration (Kinect: ``<dev>_k4a_calib.bin``; RealSense:
``<dev>_rs_calib.json``) — then marks ``status.json``. ``raw/`` is written once and
never touched again; every later stage writes new artifacts alongside it, so
re-processing can never corrupt the recording.

There is no live skeleton path: you record as many scenes as you want, then run
``viki extract`` / ``prepare`` / ``retarget`` offline.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from viki.cameras.manager import CameraManager
from viki.cameras.sync import MultiCameraSync
from viki.contracts import Episode
from viki.episode import load_meta, mark_stage, new_episode, save_meta

logger = logging.getLogger(__name__)


class SceneRecorder:
    """One recording session → one episode directory."""

    def __init__(
        self,
        manager: CameraManager,
        *,
        dataset: str | None = None,
        episodes_dir: str | Path | None = None,
        meta: dict | None = None,
    ) -> None:
        self._mgr = manager
        if dataset is not None:
            from viki import datasets as _datasets

            target: str | Path = _datasets.dataset_dir(dataset)
            Path(target).mkdir(parents=True, exist_ok=True)
        elif episodes_dir is not None:
            target = episodes_dir
        else:
            from viki import config

            target = getattr(config, "EPISODES_DIR", "data/episodes")
        self.episode: Episode = new_episode(target, {**(meta or {}), "dataset": dataset})
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._depth_dirs: dict[str, Path] = {}
        self._timestamps: list[dict] = []
        self._n = 0
        self._dropped = 0

    def record(self, seconds: float, fps: int = 15, stop_event=None) -> Episode:
        """Write every synced group until ``seconds`` elapse or ``stop_event``
        is set (the Stop button). ``seconds`` is the safety cap. Returns the episode.

        The capture loop only samples the synced group and hands it to a writer
        thread — the mp4 encode + depth ``.npy`` writes never block the loop, so
        it holds the ``1/fps`` cadence and the constant-fps mp4 stays true to
        real time. If the writer can't keep up the queue fills and groups are
        dropped (a shorter clean take beats a juddering full one)."""
        sync = MultiCameraSync(self._mgr, sync_fps=fps)
        self._write_sensor_meta()

        q: queue.Queue = queue.Queue(maxsize=max(4, fps * 3))
        self._dropped = 0
        writer = threading.Thread(
            target=self._writer_loop, args=(q, fps), name="scene-writer", daemon=True
        )
        writer.start()

        period = 1.0 / fps
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                if stop_event is not None and stop_event.is_set():
                    break
                t0 = time.monotonic()
                group = sync.get_synced_frame()
                if group is not None:
                    try:
                        q.put_nowait(group)
                    except queue.Full:
                        self._dropped += 1
                slp = period - (time.monotonic() - t0)
                if slp > 0:
                    time.sleep(slp)
        finally:
            q.put(None)  # sentinel — drain and stop the writer
            writer.join(timeout=30.0)

        if self._dropped:
            logger.warning(
                "recorder: dropped %d group(s) — writer/disk could not keep up at %d fps",
                self._dropped, fps,
            )
        self._finish(fps)
        return self.episode

    # ------------------------------------------------------------------

    def _raw(self) -> Path:
        return self.episode.raw_dir

    def _writer_loop(self, q: "queue.Queue", fps: int) -> None:
        """Drain synced groups from the capture loop and persist them. Runs on
        its own thread; ``cv2.VideoWriter.write`` and ``np.save`` release the GIL
        during the encode/IO so the capture loop keeps its cadence."""
        while True:
            group = q.get()
            if group is None:
                return
            try:
                self._save(group, fps)
            except Exception:  # noqa: BLE001 — never let one bad group kill the take
                logger.exception("recorder: failed to write a group")

    def _save(self, group, fps: int) -> None:
        for dev_id, frame in group.frames.items():
            if dev_id not in self._writers:
                h, w = frame.color.shape[:2]
                self._writers[dev_id] = cv2.VideoWriter(
                    str(self._raw() / f"{dev_id}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                self._depth_dirs[dev_id] = self._raw() / f"{dev_id}_depth"
                self._depth_dirs[dev_id].mkdir(exist_ok=True)
            self._writers[dev_id].write(frame.color)
            # only write real depth — a colour-only capture leaves frame.depth a
            # zeros placeholder, which downstream must not treat as measured
            if frame.has_depth() and frame.depth.any():
                np.save(self._depth_dirs[dev_id] / f"{self._n:06d}.npy", frame.depth)
        self._timestamps.append(
            {"sync_us": group.sync_timestamp_us, "offsets_us": group.offsets_us}
        )
        self._n += 1

    @staticmethod
    def _intr_dict(intr) -> dict | None:
        if intr is None:
            return None
        return {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "dist_coeffs": np.asarray(intr.dist_coeffs).tolist(),
        }

    def _stamp_depth_calibration(self, dev_id: str, backend, cam_entry: dict) -> None:
        """Persist whatever the offline colour↔depth reprojection needs for this
        camera: the Kinect's raw k4a blob (+ enum ints), or the RealSense's
        stream intrinsics + depth→colour extrinsic as ``<dev>_rs_calib.json``.
        No-op for a backend that exposes neither."""
        if backend is None:
            return

        blob = getattr(backend, "get_raw_calibration", lambda: None)()
        if blob:
            (self._raw() / f"{dev_id}_k4a_calib.bin").write_bytes(blob)
            cam_entry["k4a_calib"] = f"{dev_id}_k4a_calib.bin"
            try:
                from viki.cameras.kinect import _COLOR_RES_MAP, _DEPTH_MODE_MAP

                cfg = backend.config or {}
                cam_entry["k4a_depth_mode_int"] = _DEPTH_MODE_MAP.get(cfg.get("depth_mode"))
                cam_entry["k4a_color_res_int"] = _COLOR_RES_MAP.get(
                    (int(cfg.get("color_width", 0)), int(cfg.get("color_height", 0)))
                )
            except Exception:  # noqa: BLE001
                pass
            return

        rs_cal = getattr(backend, "get_rs_calibration", lambda: None)()
        if rs_cal:
            (self._raw() / f"{dev_id}_rs_calib.json").write_text(json.dumps(rs_cal, indent=2))
            cam_entry["rs_calib"] = f"{dev_id}_rs_calib.json"

    def _write_sensor_meta(self) -> None:
        """Snapshot everything an offline run needs: SDK intrinsics, the active
        extrinsics, and each camera's capture config. Written once, before the
        first frame."""
        from viki.calibration.file import read_device_extrinsics

        intr: dict = {}
        extr: dict = {}
        cams: dict = {}
        for dev_id in self._mgr.active_device_ids():
            frame = self._mgr.latest_frame(dev_id)
            ci = frame.color_intrinsics if frame else None
            di = frame.depth_intrinsics if frame else None
            # SDK-reported intrinsics only — no stored-file fallback.
            depth_intr = self._intr_dict(di)
            if depth_intr and frame is not None and frame.has_depth():
                # Stamp the *actual* frame size, not the nominal-from-mode value.
                dh, dw = frame.depth.shape[:2]
                depth_intr["width"], depth_intr["height"] = int(dw), int(dh)
            intr[dev_id] = {
                "color": self._intr_dict(ci),
                "depth": depth_intr,
                "source": "sdk" if ci is not None else "none",
            }
            e = read_device_extrinsics(dev_id)
            if e is not None:
                extr[dev_id] = {
                    "rvec": np.asarray(e.rvec).tolist(),
                    "tvec": np.asarray(e.tvec).tolist(),
                }
            backend = self._mgr.get_backend(dev_id)
            cams[dev_id] = {
                "type": type(backend).__name__.replace("Backend", "").lower() if backend else None,
                "requested": backend.config if backend else None,
                "color_shape": list(frame.color.shape) if frame is not None else None,
                "depth_shape": list(frame.depth.shape) if frame is not None and frame.has_depth() else None,
            }
            self._stamp_depth_calibration(dev_id, backend, cams[dev_id])

        (self._raw() / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        (self._raw() / "extrinsics.json").write_text(json.dumps(extr, indent=2))

        # World anchor — T_world_display for the viewer / AABB / export only.
        # extrinsics.json above is the RIG (reference-camera) frame; this is the
        # separate presentation transform. Absent ⇒ downstream treats it as
        # identity (rig frame == display frame).
        try:
            from viki import config as _cfg

            _ap = getattr(_cfg, "WORLD_ANCHOR_FILENAME", "data/world_anchor.json")
            if os.path.exists(_ap):
                shutil.copyfile(_ap, self._raw() / "world_anchor.json")
            _vp = getattr(_cfg, "VALIDATION_FILENAME", "data/validation_report.json")
            if os.path.exists(_vp):
                shutil.copyfile(_vp, self._raw() / "validation_report.json")
        except Exception:  # noqa: BLE001
            logger.warning("record: setup artifacts not snapshotted", exc_info=True)

        meta = load_meta(self.episode)
        meta["cameras"] = cams
        try:
            from viki.calibration.presets import current_active

            meta["calibration_preset"] = current_active()
        except Exception:  # noqa: BLE001
            pass
        save_meta(self.episode, meta)

    def _finish(self, fps: int) -> None:
        for w in self._writers.values():
            w.release()
        (self._raw() / "timestamps.json").write_text(json.dumps(self._timestamps, indent=2))
        stats = _sync_stats(self._timestamps)
        (self._raw() / "sync_stats.json").write_text(json.dumps(stats, indent=2))
        logger.info(
            "sync: %s (%s)",
            "bounded drift" if stats["bounded"] else "DRIFT EXCEEDS BOUND",
            ", ".join(
                f"{d}: |off|<={s['max_abs_offset_us'] / 1000:.1f}ms "
                f"drift={s['drift_ms_per_min']:+.2f}ms/min"
                for d, s in stats["per_device"].items()
            ) or "single camera",
        )
        mark_stage(
            self.episode, "record",
            frames=self._n, fps=fps, cameras=sorted(self._writers),
            sync_bounded=stats["bounded"], dropped=self._dropped,
        )
        logger.info("recorded %d frames → %s", self._n, self.episode.root)


# Residual alignment jitter is measured, not assumed (paper §3.3): every frame
# carries its offset from the shared host monotonic tick, so a linear fit of that
# offset over the recording gives each camera's clock drift relative to the host,
# and the peak |offset| is the worst-case grouping error.
_DRIFT_BOUND_MS_PER_MIN = 1.0  # ridgerun: a healthy Kinect pair drifts < 1 ms/min


def _sync_stats(timestamps: list[dict]) -> dict:
    """Per-camera offset jitter + linear clock drift from the recorded ticks."""
    per_device: dict[str, dict] = {}
    t0 = timestamps[0]["sync_us"] if timestamps else 0
    series: dict[str, list[tuple[float, float]]] = {}
    for row in timestamps:
        t_rel = float(row["sync_us"] - t0)
        for dev, off in (row.get("offsets_us") or {}).items():
            series.setdefault(dev, []).append((t_rel, float(off)))

    worst_drift = 0.0
    for dev, pairs in series.items():
        ts = np.asarray([p[0] for p in pairs], dtype=np.float64)
        off = np.asarray([p[1] for p in pairs], dtype=np.float64)
        # slope [µs offset / µs elapsed] → ms per minute
        drift = 0.0
        if ts.size >= 2 and np.ptp(ts) > 0:
            slope = float(np.polyfit(ts, off, 1)[0])
            drift = slope * 60_000_000.0 / 1000.0
        per_device[dev] = {
            "samples": int(off.size),
            "mean_offset_us": float(off.mean()) if off.size else 0.0,
            "std_offset_us": float(off.std()) if off.size else 0.0,
            "max_abs_offset_us": float(np.abs(off).max()) if off.size else 0.0,
            "drift_ms_per_min": drift,
        }
        worst_drift = max(worst_drift, abs(drift))

    return {
        "per_device": per_device,
        "worst_drift_ms_per_min": worst_drift,
        "drift_bound_ms_per_min": _DRIFT_BOUND_MS_PER_MIN,
        "bounded": worst_drift <= _DRIFT_BOUND_MS_PER_MIN,
    }
