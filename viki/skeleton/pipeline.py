"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.
"""

from __future__ import annotations

from time import sleep

from pathlib import Path
from typing import Any, Dict, Optional, Literal
import logging

from viki.calibration.models import CalibrationExtrinsics

logger = logging.getLogger(__name__)

import numpy as np
from viki.capture.base import SyncedFrameGroup

from viki.capture.manager import CameraManager
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import prepare_frame
from viki.skeleton.geometry import lift_to_3d, camera_landmarks_to_world
from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    MediaPipeHand,
)
from viki.skeleton.models import (
    Landmarks3D,
    SkeletonFrame,
    PipelineResult,
    HandDetection,
    PreparedFrame,
    DepthDebug,
    LM,
)
from viki.calibration.models import CalibrationExtrinsics
import viki.config

# Palm/knuckle landmarks used to pick a representative hand position.
_PALM_LM_POS = (
    LM.WRIST,
    LM.THUMB_CMC,
    LM.INDEX_MCP,
    LM.MIDDLE_MCP,
    LM.RING_MCP,
    LM.PINKY_MCP,
)


def _depth_stats(frame_depth: np.ndarray) -> tuple[float, float, float]:
    """
    Summarise a raw uint16‑mm depth image.

    Returns (valid_fraction, median_m, mean_m). ``valid_fraction`` is the share
    of pixels whose depth is in the plausible 0.01–10 m range; medians/means
    are taken over those pixels only. Mirrors the range filter used by
    ``lift_to_3d``.
    """
    d = frame_depth.astype(np.float32) / 1000.0
    valid = (d > 0.01) & (d <= 10.0)
    frac = float(valid.mean()) if d.size else 0.0
    if not valid.any():
        return frac, float("nan"), float("nan")
    dv = d[valid]
    return frac, float(np.median(dv)), float(dv.mean())


def _wrist_depth_m(
    det: HandDetection,
    prepared: PreparedFrame,
    backend: Any,
) -> float:
    """
    Depth (metres) measured at the detected wrist landmark, or NaN.

    Projects the wrist pixel into depth space and samples the measured depth —
    the exact value that feeds hand‑position estimation for the lifted camera.
    """
    uv = det.points.get(LM.WRIST)
    if uv is None or np.isnan(uv[0]) or np.isnan(uv[1]):
        return float("nan")
    res = backend.project_color_to_depth(uv[0], uv[1], 1.0)
    if res is None:
        res = (uv[0], uv[1])
    ud, vd = res
    depth_m = prepared.depth_m
    if depth_m is None:
        return float("nan")
    h, w = depth_m.shape[:2]
    ui, vi = int(round(ud)), int(round(vd))
    if 0 <= vi < h and 0 <= ui < w:
        z = depth_m[vi, ui]
        if np.isfinite(z) and 0.01 < z <= 10.0:
            return float(z)
    return float("nan")


class SkeletonPipeline:
    """
    End‑to‑end skeleton detection from SyncedFrameGroup to per‑camera frames.

    This pipeline:
        1. Prepares each camera frame (undistort, depth clean).
        2. Runs hand detection (MediaPipe) on each camera in parallel.
        3. Lifts 2D detections to 3D using depth maps.
        4. Transforms each camera's 3D landmarks to world frame and emits one
           ``SkeletonFrame`` per camera (tagged with its ``device_id``).

    Capture‑time fusion is intentionally NOT performed here: the per‑camera
    trajectories are recorded as‑is and fused later at the smooth/optimisation
    stage.

    Parameters
    ----------
    calibrator : CalibrationManager
        Provides per‑device intrinsics and extrinsics.
    manager : CameraManager
        Provides access to camera backends (for depth projection).
    hand : Literal["right", "left"]
        Which hand to track. Default from config.
    """

    def __init__(
        self,
        calibrator: CalibrationManager,
        manager: CameraManager,
        hand: Literal["right", "left"] = viki.config.HAND_TO_DETECT,
        discard_outliers: bool = viki.config.DISCARD_OUTLIERS,
        discard_outliers_max_portion: float = viki.config.DISCARD_OUTLIERS_MAX_PORTION,
        position_from_wrist: bool = viki.config.POSITION_FROM_WRIST,
        depth_debug: bool = viki.config.DEPTH_DEBUG,
    ) -> None:
        self._hand = hand
        self._calibrator = calibrator
        self._manager = manager
        self._detectors: dict[str, CompositeLandmarkDetector] = {}
        self._hand_type = hand

        self._discard_outliers = discard_outliers
        self._discard_outliers_max_portion = discard_outliers_max_portion
        self._position_from_wrist = position_from_wrist
        self._depth_debug = depth_debug

        # Previous hand position per camera (camera frame) used for outlier
        # rejection of the depth estimate.  None until the first valid hand.
        self._prev_hand_pos: dict[str, np.ndarray] = {}

        self._ext_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def process(self, group: SyncedFrameGroup) -> PipelineResult:
        """
        Run the full pipeline on one SyncedFrameGroup.

        Parameters
        ----------
        group : SyncedFrameGroup
            Output of MultiCameraSync.get_synced_frame().

        Returns
        -------
        PipelineResult
            Contains one ``SkeletonFrame`` per camera (world‑frame, tagged with
            ``device_id``) plus per‑camera detections and optional debug marks.
        """
        detections: dict[str, HandDetection | None] = {}
        lms_3d: dict[str, Landmarks3D | None] = {}
        prepared_by_dev: dict[str, PreparedFrame | None] = {}
        extrinsics: dict[str, CalibrationExtrinsics] = {}

        # # 1. Run detections in parallel across all cameras
        # futures = {
        #     self._executor.submit(self._detect_camera, dev_id, group): dev_id
        #     for dev_id in group.frames.keys()
        # }

        # for future in futures:
        #     dev_id, det, prepared = future.result()
        #     detections[dev_id] = det
        #     prepared_by_dev[dev_id] = prepared
        #     # 2. Lift to 3D (sequential, but fast)
        #     lms_3d[dev_id] = self._lift_camera(dev_id, group, det, prepared)
        #     extr = self._calibrator.get_extrinsics(dev_id)
        #     extrinsics[dev_id] = extr if extr else CalibrationExtrinsics()
        # 1. Detect + lift each camera sequentially (no concurrent MediaPipe
        #    inference — running two live models at once contends on the GPU and
        #    produces stale/out-of-order detections).
        dev_ids = list(group.frames.keys())
        if not dev_ids:
            return PipelineResult(frames=[], detections={})

        for dev_id in dev_ids:
            dev_id, det, prepared = self._detect_camera(dev_id, group)
            detections[dev_id] = det
            prepared_by_dev[dev_id] = prepared
            # 2. Lift to 3D (sequential, but fast)
            lms_3d[dev_id] = self._lift_camera(dev_id, group, det, prepared)
            extr = self._calibrator.get_extrinsics(dev_id)
            extrinsics[dev_id] = extr if extr else CalibrationExtrinsics()

        # 3. Build one world‑frame SkeletonFrame per camera (no fusion here).
        frames: list[SkeletonFrame] = []
        for dev_id in dev_ids:
            lm = lms_3d.get(dev_id)
            if lm is None:
                continue
            world_points = camera_landmarks_to_world(lm, extrinsics[dev_id])
            if not world_points:
                continue
            pose = compute_end_effector_pose(world_points, int(group.sync_timestamp_us))
            frames.append(
                SkeletonFrame(
                    device_id=dev_id,
                    points=world_points,
                    timestamp_us=int(group.sync_timestamp_us),
                    end_effector=pose,
                )
            )

        debug_depth_marks = None
        if self._depth_debug:
            debug_depth_marks = self._compute_debug_depth_marks(
                detections, prepared_by_dev, extrinsics
            )

        # 4. Depth diagnostics for every camera in the group (regardless of
        #    whether it was lifted). Surfaces what the depth cameras were doing
        #    this frame so recordings can be inspected for depth corruption.
        depth_debug: dict[str, DepthDebug] = {}
        for dev_id, frame in group.frames.items():
            frac, median, mean = _depth_stats(frame.depth)
            det = detections.get(dev_id)
            prepared = prepared_by_dev.get(dev_id) if det is not None else None
            backend = self._manager.get_backend(dev_id) if det is not None else None
            wrist = float("nan")
            if det is not None and prepared is not None and backend is not None:
                wrist = _wrist_depth_m(det, prepared, backend)
            depth_debug[dev_id] = DepthDebug(
                device_id=dev_id,
                depth_valid_fraction=frac,
                depth_median_m=median,
                depth_mean_m=mean,
                hand_detected=det is not None,
                wrist_depth_m=wrist,
            )

        return PipelineResult(
            frames=frames,
            detections=detections,
            debug_depth_marks=debug_depth_marks,
            depth_debug=depth_debug,
        )

    def _detect_camera(
        self, dev_id: str, group: SyncedFrameGroup
    ) -> tuple[str, Optional[HandDetection], Optional[PreparedFrame]]:
        """
        Helper for parallel detection.
        Returns a tuple (device_id, detection, prepared_frame).
        """
        prepared = self._prepare_camera(dev_id, group)
        if prepared is None:
            return dev_id, None, None

        if dev_id not in self._detectors:
            self._detectors[dev_id] = CompositeLandmarkDetector(
                detectors=[
                    # MediaPipeArm(hand=self._hand, mode="video"),
                    MediaPipeHand(hand=self._hand, mode="video"),
                ],
                mode=FusionMode.ANY,
            )

        det = self._detectors[dev_id].detect(prepared)
        return dev_id, det, prepared

    def close(self) -> None:
        """Release MediaPipe resources."""
        for detector in self._detectors.values():
            detector.close()

    def __enter__(self) -> "SkeletonPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _prepare_camera(
        self, device_id: str, group: SyncedFrameGroup
    ) -> Optional[PreparedFrame]:
        """
        Stage 1: prepare frame for detection.

        Parameters
        ----------
        device_id : str
            Camera ID.
        group : SyncedFrameGroup
            The sync group containing the frame.

        Returns
        -------
        PreparedFrame or None
            Prepared frame, or None if the frame is missing.
        """
        frame = group.frames.get(device_id)
        if frame is None:
            logger.debug("SkeletonPipeline: no synced frames from SyncFrameGroup")
            return None

        prepared = prepare_frame(frame)
        self._attach_base_depth(device_id, prepared)
        return prepared

    def _attach_base_depth(self, device_id: str, prepared: PreparedFrame) -> None:
        """
        Load the captured static-background depth (if present) onto ``prepared``.

        ``lift_to_3d`` uses ``prepared.base_depth_m`` to subtract the scene so the
        tracked hand stands out from the background. Missing/corrupt/mismatched
        bases are silently ignored so estimation never breaks.
        """
        base_dir = getattr(viki.config, "SKELETON_DEPTH_BASE_DIR", "data/depth_bases/")
        base_path = Path(base_dir) / f"{device_id}.npy"
        if not base_path.exists():
            return
        try:
            base = np.load(base_path)
        except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
            logger.error("Failed to load base depth for %s: %s", device_id, exc)
            return
        if prepared.depth_m is None or base.shape != prepared.depth_m.shape:
            logger.warning(
                "Base depth shape %s does not match frame %s for %s; ignoring",
                getattr(base, "shape", None),
                getattr(prepared.depth_m, "shape", None),
                device_id,
            )
            return
        prepared.base_depth_m = base

    def _lift_camera(
        self,
        device_id: str,
        group: SyncedFrameGroup,
        detection: Optional[HandDetection],
        prepared: Optional[PreparedFrame] = None,
    ) -> Optional[Landmarks3D]:
        """
        Stage 3: lift 2D detection to 3D.

        Parameters
        ----------
        device_id : str
            Camera ID.
        group : SyncedFrameGroup
            The sync group (used to re‑prepare if needed).
        detection : Optional[HandDetection]
            2D detection (None if no hand).
        prepared : Optional[PreparedFrame]
            Prepared frame (if already available).

        Returns
        -------
        Landmarks3D or None
            3D landmarks in camera coordinates, or None if detection absent or backend not Kinect.
        """
        if detection is None:
            # No hand this frame — forget the previous position so the next
            # detection is not compared against a stale one.
            self._prev_hand_pos.pop(device_id, None)
            return None

        # Use the provided prepared frame, or re-prepare if missing
        if prepared is None:
            prepared = self._prepare_camera(device_id, group)

        if prepared is None:
            return None

        backend = self._manager.get_backend(device_id)
        if backend is None:
            return None

        landmarks = lift_to_3d(
            detection,
            prepared,
            backend,
            prev_position=self._prev_hand_pos.get(device_id),
            discard_outliers=self._discard_outliers,
            discard_outliers_max_portion=self._discard_outliers_max_portion,
            position_from_wrist=self._position_from_wrist,
        )
        self._update_prev_hand_pos(device_id, landmarks)
        return landmarks

    @staticmethod
    def _hand_position(landmarks: Optional[Landmarks3D]) -> np.ndarray | None:
        """
        Extract a single representative hand position (camera frame) from a set
        of 3D landmarks: the wrist if finite, else the centroid of the finite
        palm/knuckle landmarks.
        """
        if landmarks is None:
            return None
        wrist = landmarks.points.get(LM.WRIST)
        if wrist is not None and np.all(np.isfinite(wrist)):
            return wrist.astype(np.float64)
        pts = [
            p
            for lm, p in landmarks.points.items()
            if lm in _PALM_LM_POS and p is not None and np.all(np.isfinite(p))
        ]
        if not pts:
            return None
        return np.mean(pts, axis=0).astype(np.float64)

    def _update_prev_hand_pos(
        self, device_id: str, landmarks: Optional[Landmarks3D]
    ) -> None:
        pos = self._hand_position(landmarks)
        if pos is None:
            self._prev_hand_pos.pop(device_id, None)
        else:
            self._prev_hand_pos[device_id] = pos

    def set_depth_debug(self, enabled: bool) -> None:
        """Enable/disable emission of raw depth-projection debug marks."""
        self._depth_debug = enabled

    def _compute_debug_depth_marks(
        self,
        detections: dict[str, HandDetection | None],
        prepared_by_dev: dict[str, PreparedFrame | None],
        extrinsics: dict[str, CalibrationExtrinsics],
    ) -> dict[str, dict[LM, np.ndarray]]:
        """
        Build per-camera, per-landmark 3D points obtained *purely* from the
        depth camera: each detected landmark is projected into depth space and
        deprojected at its own measured depth. These are the raw depth estimates
        that feed hand-position estimation, emitted for frontend visualisation.

        The points are transformed into world frame with the same
        ``transform_matrix`` used by fusion so they align with the skeleton.
        """
        out: dict[str, dict[LM, np.ndarray]] = {}
        for dev_id, det in detections.items():
            prepared = prepared_by_dev.get(dev_id)
            if det is None or prepared is None:
                continue

            backend = self._manager.get_backend(dev_id)
            if backend is None:
                continue

            depth_m = prepared.depth_m
            h, w = depth_m.shape[:2]
            K = prepared.depth_K
            if K is None or K[0, 0] <= 0 or K[1, 1] <= 0 or h == 0 or w == 0:
                continue
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            cam_marks: dict[LM, np.ndarray] = {}
            for lm, uv in det.points.items():
                if np.isnan(uv[0]) or np.isnan(uv[1]):
                    continue
                res = backend.project_color_to_depth(uv[0], uv[1], 1.0)
                if res is None:
                    res = (uv[0], uv[1])
                ud, vd = res
                ui, vi = int(round(ud)), int(round(vd))
                if 0 <= vi < h and 0 <= ui < w:
                    z = depth_m[vi, ui]
                    if not np.isnan(z) and 0.01 < z <= 10.0:
                        X = (ud - cx) * z / fx
                        Y = (vd - cy) * z / fy
                        cam_marks[lm] = np.array([X, Y, z], dtype=np.float32)

            if not cam_marks:
                continue

            extr = extrinsics.get(dev_id)
            T = extr.transform_matrix if extr else np.eye(4)
            world_marks: dict[LM, np.ndarray] = {}
            for lm, vec in cam_marks.items():
                pos_mtx = np.eye(4)
                pos_mtx[:3, 3] = vec
                world_marks[lm] = (T @ pos_mtx)[:3, 3].flatten().astype(np.float32)
            out[dev_id] = world_marks

        return out
