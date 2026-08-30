"""
viki.cameras.record
-------------------
Record synchronised RGB-D scenes into an episode directory.

Pipeline stage 1. Writes ``<dataset>/<id>/raw/`` — one colour ``.mp4`` and one
folder of raw ``uint16`` depth ``.npy`` per camera, plus ``timestamps.json`` and
the SDK-reported intrinsics + active extrinsics in force at capture time — then
marks ``status.json``. ``raw/`` is written once and never touched again; every
later stage writes new artifacts alongside it, so re-processing can never
corrupt the recording.

There is no live skeleton path: you record as many scenes as you want, then run
``viki extract`` / ``prepare`` / ``retarget`` offline.
"""

from __future__ import annotations

import json
import logging
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

    def record(self, seconds: float, fps: int = 15) -> Episode:
        """Block for ``seconds``, writing every synced group. Returns the episode."""
        sync = MultiCameraSync(self._mgr, sync_fps=fps)
        self._write_sensor_meta()
        period = 1.0 / fps
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            group = sync.get_synced_frame()
            if group is not None:
                self._save(group, fps)
            slp = period - (time.monotonic() - t0)
            if slp > 0:
                time.sleep(slp)
        self._finish(fps)
        return self.episode

    # ------------------------------------------------------------------

    def _raw(self) -> Path:
        return self.episode.raw_dir

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
            if frame.has_depth():
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

    def _write_sensor_meta(self) -> None:
        """Snapshot everything an offline run needs: SDK intrinsics, the active
        extrinsics, and each camera's capture config. Written once, before the
        first frame."""
        from viki.calibration.file import read_device_extrinsics, read_device_intrinsics

        intr: dict = {}
        extr: dict = {}
        cams: dict = {}
        for dev_id in self._mgr.active_device_ids():
            frame = self._mgr.latest_frame(dev_id)
            ci = frame.color_intrinsics if frame else None
            di = frame.depth_intrinsics if frame else None
            # Prefer the live SDK-reported intrinsics; fall back to a stored file.
            file_i = read_device_intrinsics(dev_id)
            depth_intr = self._intr_dict(di)
            if depth_intr and frame is not None and frame.has_depth():
                # Stamp the *actual* frame size, not the nominal-from-mode value.
                dh, dw = frame.depth.shape[:2]
                depth_intr["width"], depth_intr["height"] = int(dw), int(dh)
            intr[dev_id] = {
                "color": self._intr_dict(ci) or self._intr_dict(file_i),
                "depth": depth_intr,
                "source": "sdk" if ci is not None else ("file" if file_i else "none"),
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

        (self._raw() / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        (self._raw() / "extrinsics.json").write_text(json.dumps(extr, indent=2))

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
        mark_stage(
            self.episode, "record",
            frames=self._n, fps=fps, cameras=sorted(self._writers),
        )
        logger.info("recorded %d frames → %s", self._n, self.episode.root)
