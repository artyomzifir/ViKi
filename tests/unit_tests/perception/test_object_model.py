import hashlib
import json

import numpy as np

from viki.perception.object_model import (
    OBJECT_MODEL_SCHEMA,
    POINT_ACCEPTED,
    POINT_AMBIGUOUS,
    POINT_REASSIGNED,
    POINT_REJECTED,
    ObjectModelConfig,
    _Frame,
    _Model,
    _Track,
    _classify_points,
    _radius_components,
    build_object_models,
)


def _surface(center):
    axis = np.linspace(-0.02, 0.02, 9)
    z_axis = np.linspace(-0.01, 0.01, 5)
    x, y, z = np.meshgrid(axis, axis, z_axis)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())) + np.asarray(center)


def _write_semantic_sequence(root, frames=20):
    cloud = root / "semantic_cloud"
    cloud.mkdir(parents=True)
    (root / "meta.json").write_text(json.dumps({
        "schema": "viki_sam2_masks_v1",
        "objects": [
            {"id": 1, "label": "operator"},
            {"id": 2, "label": "manipulated_object"},
        ],
    }))
    for frame in range(frames):
        shift = np.array([max(frame - 7, 0) * 0.0025, 0.0, 0.0])
        obj = _surface([0.0, 0.0, 0.50]) + shift
        # A persistent SAM leak must not become part of the rigid model merely
        # because it repeats over the whole bootstrap interval.
        distractor = _surface([0.18, 0.0, 0.50])[:40]
        operator = _surface([-0.15, 0.0, 0.55])[:50]
        xyz = np.vstack((obj, distractor, operator)).astype(np.float32)
        instance = np.concatenate((
            np.full(len(obj), 2, np.int32),
            np.full(len(distractor), 2, np.int32),
            np.full(len(operator), 1, np.int32),
        ))
        camera = np.arange(len(xyz), dtype=np.uint8) % 2
        with (cloud / f"{frame:06d}.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                xyz=xyz,
                rgb=np.zeros((len(xyz), 3), np.uint8),
                instance_id=instance,
                label_code=np.zeros(len(xyz), np.uint8),
                source_camera=camera,
            )
    return cloud


def test_radius_components_separate_detached_clusters():
    points = np.vstack((
        np.arange(5)[:, None] * np.array([[0.005, 0.0, 0.0]]),
        np.array([1.0, 0.0, 0.0]) + np.arange(3)[:, None] * np.array([[0.005, 0.0, 0.0]]),
    ))

    components = _radius_components(points, 0.011)

    assert [len(part) for part in components] == [5, 3]


def test_object_model_rejects_persistent_detached_leak_and_tracks_motion(tmp_path):
    root = tmp_path / "sam2.1_hiera_small"
    cloud = _write_semantic_sequence(root)
    source = cloud / "000000.npz"
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    output = build_object_models(
        root,
        config=ObjectModelConfig(
            bootstrap_frames=8,
            core_support_fraction=0.25,
            min_component_points=10,
            icp_iterations=5,
            shell_support_frames=3,
        ),
    )

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema"]) == OBJECT_MODEL_SCHEMA
        assert int(archive["frame_count"]) == 20
        model = archive["model_xyz_object"]
        dimensions = np.percentile(model, 98, axis=0) - np.percentile(model, 2, axis=0)
        assert dimensions[0] < 0.08
        translation = archive["translation_world"][:, 0]
        assert translation[-1, 0] - translation[0, 0] > 0.020
        state = archive["candidate_state"]
        assert np.mean(state == POINT_ACCEPTED) > 0.70
        assert np.sum(state == POINT_REJECTED) >= 20 * 30
        kinds, counts = np.unique(archive["model_kind"], return_counts=True)
        by_kind = dict(zip(kinds.tolist(), counts.tolist()))
        assert by_kind.get(2, 0) <= np.ceil(0.5 * by_kind[1])
        assert len(archive["model_support"]) == len(model)
        assert len(archive["model_kind"]) == len(model)
        assert archive["model_offsets"].tolist()[0] == 0
        assert archive["model_offsets"].tolist()[-1] == len(model)


def test_mutual_assignment_marks_overlap_ambiguous_and_can_reassign():
    def track(translation):
        return _Track(
            rotation=np.eye(3, dtype=np.float32)[None],
            translation=np.asarray(translation, np.float32)[None],
            confidence=np.ones(1, np.float32),
            residual_median=np.zeros(1, np.float32),
            residual_p95=np.zeros(1, np.float32),
            coverage=np.ones(1, np.float32),
            retained_fraction=np.ones(1, np.float32),
            rotation_information=np.ones((1, 3), np.float32),
        )

    models = [
        _Model(
            2, "object-a", np.zeros((1, 3)), np.ones(1), np.ones(1),
            track([0, 0, 0]), np.zeros(1, bool),
        ),
        _Model(
            3, "object-b", np.zeros((1, 3)), np.ones(1), np.ones(1),
            track([0.02, 0, 0]), np.zeros(1, bool),
        ),
    ]
    frame = _Frame(
        xyz=np.asarray([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], np.float32),
        initial_id=np.asarray([2, 2, 2], np.int32),
        point_index=np.arange(3, dtype=np.int32),
        source_camera=np.zeros(3, np.uint8),
        operator_xyz=np.empty((0, 3), np.float32),
    )

    _, _, _, assigned, state, _ = _classify_points(
        [frame], models, ObjectModelConfig()
    )

    np.testing.assert_array_equal(
        state, [POINT_ACCEPTED, POINT_AMBIGUOUS, POINT_REASSIGNED]
    )
    np.testing.assert_array_equal(assigned, [2, 0, 3])
