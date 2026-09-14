"""Experimental rigid object models from SAM-supported semantic clouds.

This module deliberately sits *after* :mod:`viki.perception.segmentation`.
It never rewrites the semantic cloud: the output contains a compact canonical
surface, an SE(3) track, and decisions referring back to point indices in each
source frame.  Retarget does not consume this artifact yet.

The model has two parts.  A conservative ``core`` is bootstrapped from a short
stationary prefix using a spatial component and temporal voxel consensus.  A
``shell`` can then add repeatedly observed, model-adjacent surface voxels, but
only away from operator contact.  This prevents one contact frame from growing
the hand into the object while still allowing a previously hidden face to be
learned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)

OBJECT_MODEL_SCHEMA = "viki_object_models_v1"

POINT_REJECTED = np.uint8(0)
POINT_ACCEPTED = np.uint8(1)
POINT_REASSIGNED = np.uint8(2)
POINT_AMBIGUOUS = np.uint8(3)

MODEL_CORE = np.uint8(1)
MODEL_SHELL = np.uint8(2)


@dataclass(frozen=True)
class ObjectModelConfig:
    """Parameters for the first non-production object-model experiment."""

    voxel_m: float = 0.004
    component_radius_m: float = 0.012
    bootstrap_frames: int = 60
    core_support_fraction: float = 0.25
    min_component_points: int = 20
    icp_iterations: int = 7
    icp_trim_fraction: float = 0.80
    max_correspondence_m: float = 0.030
    inlier_distance_m: float = 0.012
    shell_support_frames: int = 5
    shell_support_fraction: float = 0.03
    shell_max_core_ratio: float = 0.50
    shell_distance_m: float = 0.018
    shell_extent_margin_m: float = 0.015
    centroid_reacquire_m: float = 0.080
    reacquire_confidence: float = 0.25
    max_prediction_correction_m: float = 0.025
    max_component_step_m: float = 0.035
    max_rotation_step_deg: float = 12.0
    contact_distance_m: float = 0.012
    contact_fraction: float = 0.05
    assignment_margin_m: float = 0.003


@dataclass
class _Frame:
    xyz: np.ndarray
    initial_id: np.ndarray
    point_index: np.ndarray
    source_camera: np.ndarray
    operator_xyz: np.ndarray


@dataclass
class _Track:
    rotation: np.ndarray
    translation: np.ndarray
    confidence: np.ndarray
    residual_median: np.ndarray
    residual_p95: np.ndarray
    coverage: np.ndarray
    retained_fraction: np.ndarray
    rotation_information: np.ndarray


@dataclass
class _Model:
    object_id: int
    label: str
    xyz: np.ndarray
    support: np.ndarray
    kind: np.ndarray
    track: _Track
    contact: np.ndarray


def _radius_components(points: np.ndarray, radius_m: float) -> list[np.ndarray]:
    """Return point-index components, largest first."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return []
    pairs = cKDTree(points).query_pairs(radius_m, output_type="ndarray")
    if len(pairs):
        row = np.concatenate((pairs[:, 0], pairs[:, 1]))
        column = np.concatenate((pairs[:, 1], pairs[:, 0]))
        graph = coo_matrix(
            (np.ones(len(row), np.uint8), (row, column)),
            shape=(len(points), len(points)),
        ).tocsr()
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(points), dtype=np.int32)
    sizes = np.bincount(labels)
    return [
        np.flatnonzero(labels == value)
        for value in np.argsort(sizes)[::-1]
    ]


def _load_frames(
    semantic_cloud: Path,
    object_ids: np.ndarray,
    operator_ids: set[int],
) -> list[_Frame]:
    wanted = set(int(value) for value in object_ids) | operator_ids
    frames: list[_Frame] = []
    for path in sorted(semantic_cloud.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["instance_id"], dtype=np.int32)
            keep = np.asarray([int(value) in wanted for value in ids], bool)
            indices = np.flatnonzero(keep).astype(np.int32)
            xyz = np.asarray(archive["xyz"][keep], dtype=np.float32)
            kept_ids = ids[keep]
            cameras = np.asarray(archive["source_camera"][keep], dtype=np.uint8)
        is_operator = np.asarray(
            [int(value) in operator_ids for value in kept_ids], bool
        )
        is_object = ~is_operator
        frames.append(_Frame(
            xyz=xyz[is_object],
            initial_id=kept_ids[is_object],
            point_index=indices[is_object],
            source_camera=cameras[is_object],
            operator_xyz=xyz[is_operator],
        ))
    return frames


def _object_rows(frame: _Frame, object_id: int) -> np.ndarray:
    return np.flatnonzero(frame.initial_id == object_id)


def _seed_component(
    frame: _Frame,
    object_id: int,
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _object_rows(frame, object_id)
    points = frame.xyz[rows]
    cameras = frame.source_camera[rows]
    components = _radius_components(points, cfg.component_radius_m)
    candidates = [part for part in components if len(part) >= cfg.min_component_points]
    if not candidates:
        raise ValueError(f"object {object_id} has no bootstrap component")
    # Multi-view support wins before size.  The prompt-derived instance id is
    # already a strong prior; this only removes detached mask leakage.
    selected = max(
        candidates,
        key=lambda part: (len(np.unique(cameras[part])), len(part)),
    )
    return points[selected], cameras[selected]


def _bootstrap_observations(
    frames: list[_Frame],
    object_id: int,
    cfg: ObjectModelConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    count = min(cfg.bootstrap_frames, len(frames))
    seed, seed_cameras = _seed_component(frames[0], object_id, cfg)
    observations = [seed]
    camera_observations = [seed_cameras]
    anchor = np.median(seed, axis=0)
    expected_size = len(seed)

    for frame in frames[1:count]:
        rows = _object_rows(frame, object_id)
        points = frame.xyz[rows]
        cameras = frame.source_camera[rows]
        choices = [
            part for part in _radius_components(points, cfg.component_radius_m)
            if len(part) >= cfg.min_component_points
        ]
        if not choices:
            continue

        def cost(part: np.ndarray) -> float:
            center = np.median(points[part], axis=0)
            distance = float(np.linalg.norm(center - anchor))
            size_cost = 0.005 * abs(np.log(max(len(part), 1) / expected_size))
            view_cost = 0.020 if len(np.unique(cameras[part])) < 2 else 0.0
            return distance + size_cost + view_cost

        selected = min(choices, key=cost)
        cloud = points[selected]
        observations.append(cloud)
        camera_observations.append(cameras[selected])
        anchor = 0.9 * anchor + 0.1 * np.median(cloud, axis=0)
        expected_size = max(1, int(round(0.9 * expected_size + 0.1 * len(cloud))))
    return observations, camera_observations


def _voxel_consensus(
    observations: list[np.ndarray],
    camera_observations: list[np.ndarray],
    *,
    voxel_m: float,
    minimum_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # key -> [frame count, point sum, point count, camera bit mask]
    records: dict[tuple[int, int, int], list[object]] = {}
    for points, cameras in zip(observations, camera_observations):
        keys = np.floor(np.asarray(points) / voxel_m).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        sums = np.zeros((len(unique), 3), np.float64)
        counts = np.zeros(len(unique), np.int32)
        np.add.at(sums, inverse, points)
        np.add.at(counts, inverse, 1)
        for index, key_array in enumerate(unique):
            key = tuple(int(value) for value in key_array)
            local = inverse == index
            bits = 0
            for camera in np.unique(cameras[local]):
                bits |= 1 << int(camera)
            mean = sums[index] / counts[index]
            if key not in records:
                records[key] = [1, mean.copy(), 1, bits]
            else:
                item = records[key]
                item[0] = int(item[0]) + 1
                item[1] = np.asarray(item[1]) + mean
                item[2] = int(item[2]) + 1
                item[3] = int(item[3]) | bits

    points, support, camera_bits = [], [], []
    denominator = max(len(observations), 1)
    for item in records.values():
        frame_count = int(item[0])
        if frame_count < minimum_frames:
            continue
        points.append(np.asarray(item[1]) / int(item[2]))
        support.append(frame_count / denominator)
        camera_bits.append(int(item[3]))
    if not points:
        raise ValueError("temporal consensus removed every object voxel")
    return (
        np.asarray(points, np.float64),
        np.asarray(support, np.float32),
        np.asarray(camera_bits, np.uint16),
    )


def _kabsch(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    covariance = ((source - source_center) * weights[:, None]).T @ (
        target - target_center
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _fit_pose(
    model: np.ndarray,
    tree: cKDTree,
    observed: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(observed) < cfg.min_component_points:
        return rotation, translation, np.full(len(observed), np.inf), np.zeros(
            len(observed), np.int64
        )
    current_rotation = rotation.copy()
    current_translation = translation.copy()
    for _ in range(cfg.icp_iterations):
        canonical = (observed - current_translation) @ current_rotation
        distance, match = tree.query(canonical, workers=1)
        quantile = float(np.quantile(distance, cfg.icp_trim_fraction))
        gate = min(
            cfg.max_correspondence_m,
            max(cfg.inlier_distance_m, quantile),
        )
        keep = distance <= gate
        if int(keep.sum()) < cfg.min_component_points:
            break
        residual = distance[keep]
        scale = max(0.004, 1.4826 * float(np.median(residual)))
        weights = 1.0 / (1.0 + (residual / scale) ** 2)
        next_rotation, next_translation = _kabsch(
            model[match[keep]], observed[keep], weights
        )
        delta_t = float(np.linalg.norm(next_translation - current_translation))
        delta_r = _rotation_angle(next_rotation @ current_rotation.T)
        current_rotation, current_translation = next_rotation, next_translation
        if delta_t < 1e-5 and delta_r < np.deg2rad(0.01):
            break
    canonical = (observed - current_translation) @ current_rotation
    distance, match = tree.query(canonical, workers=1)
    return current_rotation, current_translation, distance, match


def _translation_seed(
    observed: np.ndarray,
    previous: np.ndarray,
    cfg: ObjectModelConfig,
) -> np.ndarray:
    """Use the nearest substantial component to keep ICP inside its basin."""

    choices = [
        part for part in _radius_components(observed, cfg.component_radius_m)
        if len(part) >= cfg.min_component_points
    ]
    if not choices:
        return previous
    centers = np.asarray([np.median(observed[part], axis=0) for part in choices])
    distance = np.linalg.norm(centers - previous, axis=1)
    best = int(np.argmin(distance))
    if distance[best] > cfg.centroid_reacquire_m:
        return previous
    return centers[best]


def _camera_motion_seed(
    observed: np.ndarray,
    cameras: np.ndarray,
    previous_translation: np.ndarray,
    previous_centers: dict[int, np.ndarray],
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Predict translation from same-camera visible-centroid increments.

    Absolute visible centroids differ between viewpoints.  Their increments
    are much less biased, including when a fused component temporarily splits
    into two camera-specific surfaces.
    """

    centers: dict[int, np.ndarray] = {}
    deltas = []
    for camera in np.unique(cameras):
        camera_id = int(camera)
        points = observed[cameras == camera]
        choices = [
            part for part in _radius_components(points, cfg.component_radius_m)
            if len(part) >= cfg.min_component_points
        ]
        if not choices:
            continue
        candidates = np.asarray([np.median(points[part], axis=0) for part in choices])
        old = previous_centers.get(camera_id)
        selected = 0 if old is None else int(np.argmin(
            np.linalg.norm(candidates - old, axis=1)
        ))
        center = candidates[selected]
        centers[camera_id] = center
        if old is not None:
            delta = center - old
            if float(np.linalg.norm(delta)) <= cfg.max_component_step_m:
                deltas.append(delta)
    if not deltas:
        return previous_translation, centers
    return previous_translation + np.median(np.asarray(deltas), axis=0), centers


def _contact_track(
    frames: list[_Frame], object_id: int, cfg: ObjectModelConfig
) -> np.ndarray:
    contact = np.zeros(len(frames), bool)
    for index, frame in enumerate(frames):
        rows = _object_rows(frame, object_id)
        if not len(rows) or not len(frame.operator_xyz):
            continue
        distance = cKDTree(frame.operator_xyz).query(frame.xyz[rows], workers=1)[0]
        contact[index] = float(np.mean(distance <= cfg.contact_distance_m)) >= (
            cfg.contact_fraction
        )
    return contact


def _rotation_information(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(3, np.float32)
    centered = points - np.mean(points, axis=0)
    information = np.zeros((3, 3), np.float64)
    for point in centered:
        information += np.dot(point, point) * np.eye(3) - np.outer(point, point)
    values = np.linalg.eigvalsh(information)
    maximum = max(float(values[-1]), 1e-12)
    return np.asarray(values / maximum, np.float32)


def _pose_metrics(
    distance: np.ndarray,
    match: np.ndarray,
    model_size: int,
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, float, float, float, float, float]:
    inlier = distance <= cfg.inlier_distance_m
    if not inlier.any():
        return inlier, np.inf, np.inf, 0.0, 0.0, 0.0
    median = float(np.median(distance[inlier]))
    p95 = float(np.percentile(distance[inlier], 95))
    coverage = len(np.unique(match[inlier])) / model_size
    retained = float(np.mean(inlier))
    confidence = (
        np.exp(-((median / 0.006) ** 2))
        * min(1.0, coverage / 0.25)
        * min(1.0, retained / 0.50)
    )
    return inlier, median, p95, coverage, retained, float(confidence)


def _track_model(
    frames: list[_Frame],
    object_id: int,
    model: np.ndarray,
    initial_translation: np.ndarray,
    cfg: ObjectModelConfig,
    report: Callable[..., None] | None,
    stage: str,
) -> _Track:
    total = len(frames)
    rotation = np.zeros((total, 3, 3), np.float32)
    translation = np.zeros((total, 3), np.float32)
    confidence = np.zeros(total, np.float32)
    median = np.full(total, np.inf, np.float32)
    p95 = np.full(total, np.inf, np.float32)
    coverage = np.zeros(total, np.float32)
    retained = np.zeros(total, np.float32)
    information = np.zeros((total, 3), np.float32)
    current_rotation = np.eye(3, dtype=np.float64)
    current_translation = np.asarray(initial_translation, np.float64).copy()
    previous_camera_centers: dict[int, np.ndarray] = {}
    tree = cKDTree(model)

    for frame_index, frame in enumerate(frames):
        rows = _object_rows(frame, object_id)
        observed = np.asarray(frame.xyz[rows], np.float64)
        observed_cameras = frame.source_camera[rows]
        previous_rotation = current_rotation.copy()
        previous_translation = current_translation.copy()
        predicted_translation, camera_centers = _camera_motion_seed(
            observed,
            observed_cameras,
            current_translation,
            previous_camera_centers,
            cfg,
        )
        previous_camera_centers = camera_centers
        fitted_rotation, fitted_translation, distance, match = _fit_pose(
            model,
            tree,
            observed,
            current_rotation,
            predicted_translation,
            cfg,
        )
        inlier, med, high, cov, keep_fraction, score = _pose_metrics(
            distance, match, len(model), cfg
        )
        translation_step = float(np.linalg.norm(
            fitted_translation - predicted_translation
        ))
        rotation_step = _rotation_angle(fitted_rotation @ current_rotation.T)
        implausible = (
            translation_step > cfg.max_prediction_correction_m
            or rotation_step > np.deg2rad(cfg.max_rotation_step_deg)
        )

        # Normal tracking keeps the previous SE(3) seed.  A component-centroid
        # seed is tried only after confidence falls, otherwise a one-camera
        # dropout can inject its visible-centroid jump into an otherwise stable
        # pose track.
        if (score < cfg.reacquire_confidence or implausible) and len(observed):
            seeded_translation = _translation_seed(
                observed, current_translation, cfg
            )
            if not np.allclose(seeded_translation, current_translation):
                alt_rotation, alt_translation, alt_distance, alt_match = _fit_pose(
                    model,
                    tree,
                    observed,
                    current_rotation,
                    seeded_translation,
                    cfg,
                )
                alt_metrics = _pose_metrics(alt_distance, alt_match, len(model), cfg)
                alt_implausible = (
                    float(np.linalg.norm(alt_translation - predicted_translation))
                    > cfg.max_prediction_correction_m
                    or _rotation_angle(alt_rotation @ current_rotation.T)
                    > np.deg2rad(cfg.max_rotation_step_deg)
                )
                if (
                    (implausible and not alt_implausible and alt_metrics[-1] >= 0.10)
                    or (not alt_implausible and alt_metrics[-1] > score)
                ):
                    fitted_rotation, fitted_translation = alt_rotation, alt_translation
                    distance, match = alt_distance, alt_match
                    inlier, med, high, cov, keep_fraction, score = alt_metrics
                    implausible = False

        if implausible:
            score = 0.0

        # A failed fit must not teleport the prediction used by the next frame.
        if score < 0.10 or int(inlier.sum()) < cfg.min_component_points:
            current_rotation, current_translation = (
                previous_rotation, predicted_translation
            )
            canonical = (observed - current_translation) @ current_rotation
            distance, match = tree.query(canonical, workers=1) if len(observed) else (
                np.empty(0), np.empty(0, np.int64)
            )
            inlier, med, high, cov, keep_fraction, score = _pose_metrics(
                distance, match, len(model), cfg
            )
        else:
            current_rotation, current_translation = fitted_rotation, fitted_translation

        rotation[frame_index] = current_rotation
        translation[frame_index] = current_translation
        confidence[frame_index] = score
        median[frame_index] = med
        p95[frame_index] = high
        coverage[frame_index] = cov
        retained[frame_index] = keep_fraction
        if inlier.any():
            information[frame_index] = _rotation_information(model[match[inlier]])
        if report is not None and (frame_index + 1) % 30 == 0:
            report(stage=stage, frame=frame_index + 1, total=total, object_id=object_id)
    return _Track(
        rotation=rotation,
        translation=translation,
        confidence=confidence,
        residual_median=median,
        residual_p95=p95,
        coverage=coverage,
        retained_fraction=retained,
        rotation_information=information,
    )


def _component_supported_points(
    observed: np.ndarray,
    distance: np.ndarray,
    cfg: ObjectModelConfig,
) -> np.ndarray:
    keep = np.zeros(len(observed), bool)
    for part in _radius_components(observed, cfg.component_radius_m):
        if len(part) < cfg.min_component_points:
            continue
        supported = distance[part] <= cfg.inlier_distance_m
        if int(supported.sum()) >= max(5, int(np.ceil(0.02 * len(part)))):
            keep[part] = True
    return keep


def _grow_shell(
    frames: list[_Frame],
    object_id: int,
    core: np.ndarray,
    track: _Track,
    contact: np.ndarray,
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(core)
    lower, upper = np.percentile(core, [1, 99], axis=0)
    lower -= cfg.shell_extent_margin_m
    upper += cfg.shell_extent_margin_m
    records: dict[tuple[int, int, int], list[object]] = {}
    eligible_frames = 0
    for frame_index, frame in enumerate(frames):
        if contact[frame_index] or track.confidence[frame_index] < 0.35:
            continue
        rows = _object_rows(frame, object_id)
        observed = np.asarray(frame.xyz[rows], np.float64)
        if not len(observed):
            continue
        rotation = np.asarray(track.rotation[frame_index], np.float64)
        translation = np.asarray(track.translation[frame_index], np.float64)
        canonical = (observed - translation) @ rotation
        distance = tree.query(canonical, workers=1)[0]
        component_keep = _component_supported_points(observed, distance, cfg)
        bounded = np.all((canonical >= lower) & (canonical <= upper), axis=1)
        keep = component_keep & bounded & (distance <= cfg.shell_distance_m)
        if not keep.any():
            continue
        eligible_frames += 1
        points = canonical[keep]
        keys = np.floor(points / cfg.voxel_m).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        sums = np.zeros((len(unique), 3), np.float64)
        counts = np.zeros(len(unique), np.int32)
        np.add.at(sums, inverse, points)
        np.add.at(counts, inverse, 1)
        for index, key_array in enumerate(unique):
            key = tuple(int(value) for value in key_array)
            mean = sums[index] / counts[index]
            if key not in records:
                records[key] = [1, mean.copy(), 1]
            else:
                item = records[key]
                item[0] = int(item[0]) + 1
                item[1] = np.asarray(item[1]) + mean
                item[2] = int(item[2]) + 1

    shell, support = [], []
    denominator = max(eligible_frames, 1)
    minimum_support = max(
        cfg.shell_support_frames,
        int(np.ceil(cfg.shell_support_fraction * eligible_frames)),
    )
    for item in records.values():
        count = int(item[0])
        if count < minimum_support:
            continue
        point = np.asarray(item[1]) / int(item[2])
        if float(tree.query(point, workers=1)[0]) <= 0.75 * cfg.voxel_m:
            continue
        shell.append(point)
        support.append(count / denominator)
    shell_array = np.asarray(shell, np.float64).reshape(-1, 3)
    support_array = np.asarray(support, np.float32)
    budget = max(0, int(np.ceil(cfg.shell_max_core_ratio * len(core))))
    if budget == 0:
        return np.empty((0, 3), np.float64), np.empty(0, np.float32)
    if len(shell_array) > budget:
        selected = np.argsort(support_array, kind="stable")[-budget:]
        shell_array = shell_array[selected]
        support_array = support_array[selected]
    return shell_array, support_array


def _build_model(
    frames: list[_Frame],
    object_id: int,
    label: str,
    cfg: ObjectModelConfig,
    report: Callable[..., None] | None,
) -> _Model:
    observations, camera_observations = _bootstrap_observations(
        frames, object_id, cfg
    )
    minimum = max(2, int(np.ceil(cfg.core_support_fraction * len(observations))))
    core_world, core_support, _ = _voxel_consensus(
        observations,
        camera_observations,
        voxel_m=cfg.voxel_m,
        minimum_frames=minimum,
    )
    center = np.median(core_world, axis=0)
    core = core_world - center
    contact = _contact_track(frames, object_id, cfg)
    provisional = _track_model(
        frames, object_id, core, center, cfg, report, "object_model_core_track"
    )
    shell, shell_support = _grow_shell(
        frames, object_id, core, provisional, contact, cfg
    )
    model = np.vstack((core, shell)) if len(shell) else core
    support = np.concatenate((core_support, shell_support))
    kind = np.concatenate((
        np.full(len(core), MODEL_CORE, np.uint8),
        np.full(len(shell), MODEL_SHELL, np.uint8),
    ))
    final = _track_model(
        frames, object_id, model, center, cfg, report, "object_model_final_track"
    )
    return _Model(object_id, label, model, support, kind, final, contact)


def _classify_points(
    frames: list[_Frame],
    models: list[_Model],
    cfg: ObjectModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    point_indices, initial_ids, assigned_ids, states, residuals = [], [], [], [], []
    trees = [cKDTree(model.xyz) for model in models]
    model_ids = np.asarray([model.object_id for model in models], np.int32)
    for frame_index, frame in enumerate(frames):
        count = len(frame.xyz)
        if not count:
            offsets.append(offsets[-1])
            continue
        distances = np.full((count, len(models)), np.inf, np.float64)
        for column, (model, tree) in enumerate(zip(models, trees)):
            rotation = np.asarray(model.track.rotation[frame_index], np.float64)
            translation = np.asarray(model.track.translation[frame_index], np.float64)
            canonical = (frame.xyz - translation) @ rotation
            distances[:, column] = tree.query(canonical, workers=1)[0]
        order = np.argsort(distances, axis=1)
        best_column = order[:, 0]
        best = distances[np.arange(count), best_column]
        second = (
            distances[np.arange(count), order[:, 1]]
            if len(models) > 1 else np.full(count, np.inf)
        )
        valid = best <= cfg.inlier_distance_m
        ambiguous = valid & (second <= cfg.inlier_distance_m) & (
            second - best < cfg.assignment_margin_m
        )
        assigned = np.where(valid & ~ambiguous, model_ids[best_column], 0).astype(
            np.int32
        )
        state = np.full(count, POINT_REJECTED, np.uint8)
        state[valid & ~ambiguous & (assigned == frame.initial_id)] = POINT_ACCEPTED
        state[valid & ~ambiguous & (assigned != frame.initial_id)] = POINT_REASSIGNED
        state[ambiguous] = POINT_AMBIGUOUS
        assigned[ambiguous] = 0
        point_indices.append(frame.point_index)
        initial_ids.append(frame.initial_id)
        assigned_ids.append(assigned)
        states.append(state)
        residuals.append(best.astype(np.float32))
        offsets.append(offsets[-1] + count)
    concatenate = lambda values, dtype: (
        np.concatenate(values).astype(dtype) if values else np.empty(0, dtype)
    )
    return (
        np.asarray(offsets, np.int64),
        concatenate(point_indices, np.int32),
        concatenate(initial_ids, np.int32),
        concatenate(assigned_ids, np.int32),
        concatenate(states, np.uint8),
        concatenate(residuals, np.float32),
    )


def build_object_models(
    segmentation_dir: str | Path,
    *,
    config: ObjectModelConfig | None = None,
    output: str | Path | None = None,
    report: Callable[..., None] | None = None,
) -> Path:
    """Build a compact, non-destructive object-model artifact.

    Parameters
    ----------
    segmentation_dir:
        Directory produced by ``viki segment`` / ``viki segment-lift``.
    output:
        Defaults to ``<segmentation_dir>/object_models.npz``.
    """

    started = time.perf_counter()
    cfg = config or ObjectModelConfig()
    root = Path(segmentation_dir)
    meta_path = root / "meta.json"
    semantic_cloud = root / "semantic_cloud"
    if not meta_path.is_file() or not semantic_cloud.is_dir():
        raise FileNotFoundError("object model requires SAM meta.json and semantic_cloud/")
    meta = json.loads(meta_path.read_text())
    objects = [item for item in meta.get("objects", []) if item["label"] != "operator"]
    if not objects:
        raise ValueError("segmentation declares no non-operator objects")
    object_ids = np.asarray([int(item["id"]) for item in objects], np.int32)
    operator_ids = {
        int(item["id"]) for item in meta.get("objects", [])
        if item["label"] == "operator"
    }

    load_started = time.perf_counter()
    frames = _load_frames(semantic_cloud, object_ids, operator_ids)
    load_elapsed = time.perf_counter() - load_started
    if not frames:
        raise ValueError("semantic cloud has no frames")

    model_started = time.perf_counter()
    models = [
        _build_model(frames, int(item["id"]), str(item["label"]), cfg, report)
        for item in objects
    ]
    model_elapsed = time.perf_counter() - model_started
    classify_started = time.perf_counter()
    offsets, indices, initial, assigned, state, residual = _classify_points(
        frames, models, cfg
    )
    classify_elapsed = time.perf_counter() - classify_started

    model_offsets = [0]
    for model in models:
        model_offsets.append(model_offsets[-1] + len(model.xyz))
    out = Path(output) if output is not None else root / "object_models.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "load_elapsed_sec": load_elapsed,
        "model_elapsed_sec": model_elapsed,
        "classify_elapsed_sec": classify_elapsed,
        "elapsed_sec": time.perf_counter() - started,
        "source_frames": len(frames),
        "candidate_points": int(len(indices)),
    }
    with out.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(OBJECT_MODEL_SCHEMA),
            source_schema=np.asarray(meta.get("schema", "")),
            frame_count=np.int32(len(frames)),
            object_ids=object_ids,
            object_labels=np.asarray([model.label for model in models]),
            model_offsets=np.asarray(model_offsets, np.int64),
            model_xyz_object=np.concatenate([model.xyz for model in models]).astype(
                np.float32
            ),
            model_support=np.concatenate([model.support for model in models]).astype(
                np.float32
            ),
            model_kind=np.concatenate([model.kind for model in models]).astype(np.uint8),
            rotation_world_object=np.stack([
                model.track.rotation for model in models
            ], axis=1),
            translation_world=np.stack([
                model.track.translation for model in models
            ], axis=1),
            track_confidence=np.stack([
                model.track.confidence for model in models
            ], axis=1),
            residual_median_m=np.stack([
                model.track.residual_median for model in models
            ], axis=1),
            residual_p95_m=np.stack([
                model.track.residual_p95 for model in models
            ], axis=1),
            coverage=np.stack([model.track.coverage for model in models], axis=1),
            retained_fraction=np.stack([
                model.track.retained_fraction for model in models
            ], axis=1),
            rotation_information=np.stack([
                model.track.rotation_information for model in models
            ], axis=1),
            contact=np.stack([model.contact for model in models], axis=1),
            candidate_frame_offsets=offsets,
            candidate_point_indices=indices,
            candidate_initial_id=initial,
            candidate_assigned_id=assigned,
            candidate_state=state,
            candidate_residual_m=residual,
            config_json=np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
            metrics_json=np.asarray(json.dumps(metrics, sort_keys=True)),
        )
    logger.info("object models %s: %s", root, out)
    return out


__all__ = [
    "MODEL_CORE",
    "MODEL_SHELL",
    "OBJECT_MODEL_SCHEMA",
    "POINT_ACCEPTED",
    "POINT_AMBIGUOUS",
    "POINT_REASSIGNED",
    "POINT_REJECTED",
    "ObjectModelConfig",
    "build_object_models",
]
