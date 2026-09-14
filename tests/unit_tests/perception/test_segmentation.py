import json

import numpy as np
import pytest

from viki.perception.auto_prompts import (
    AutoPromptConfig,
    _box_from_uv,
    _interior_point,
    _select_prompt_components,
)
from viki.perception.segmentation import (
    MASK_SCHEMA,
    PROMPT_SCHEMA,
    MaskArchive,
    _save_mask_chunk,
    load_prompt_spec,
    unpack_mask_chunk,
)


def _grid(center, shape, step):
    axes = [
        (np.arange(size) - (size - 1) / 2.0) * step
        for size in shape
    ]
    values = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return values + np.asarray(center)


def test_auto_prompt_component_selection_separates_large_operator_and_objects():
    operator = _grid([0.0, 0.0, 0.3], (8, 8, 8), 0.012)
    object_centers = [[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    object_clouds = [_grid(center, (5, 5, 5), 0.01) for center in object_centers]
    xyz = np.concatenate([operator, *object_clouds]).astype(np.float32)
    rgb = np.concatenate([
        np.tile([90, 92, 91], (len(operator), 1)),
        np.tile([220, 30, 30], (len(object_clouds[0]), 1)),
        np.tile([100, 30, 180], (len(object_clouds[1]), 1)),
        np.tile([20, 50, 210], (len(object_clouds[2]), 1)),
    ]).astype(np.uint8)
    camera = np.arange(len(xyz), dtype=np.int32) % 2
    cfg = AutoPromptConfig(
        min_component_points=50,
        component_radius_m=0.018,
        max_object_extent_m=0.15,
    )

    selected_operator, objects, components = _select_prompt_components(
        xyz, rgb, camera, 2, cfg,
    )

    assert selected_operator.point_count == len(operator)
    assert len(objects) == 3
    assert len(components) == 4
    np.testing.assert_allclose(
        [item.centroid[0] for item in objects], [-0.25, 0.0, 0.25], atol=1e-6,
    )


def test_auto_prompt_projection_makes_bounded_box_and_interior_click():
    uv = np.array([[x, y] for y in range(20, 41, 2) for x in range(10, 31, 2)])
    cfg = AutoPromptConfig(box_margin_px=4, box_percentile=0)

    box = _box_from_uv(uv, 50, 60, cfg)
    point = _interior_point(uv, 50, 60)

    assert box == [6, 16, 34, 44]
    assert box[0] <= point[0] <= box[2]
    assert box[1] <= point[1] <= box[3]


def _prompt_manifest():
    return {
        "schema": PROMPT_SCHEMA,
        "objects": [
            {"id": 1, "label": "operator"},
            {"id": 2, "label": "manipulated_object"},
        ],
        "cameras": {
            "cam0": [
                {"frame": 0, "object_id": 1, "box": [1, 2, 10, 12]},
                {
                    "frame": 0,
                    "object_id": 2,
                    "points": [[20, 21], [2, 3]],
                    "point_labels": [1, 0],
                },
            ]
        },
    }


def test_prompt_manifest_is_strict_and_typed(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(_prompt_manifest()))

    spec = load_prompt_spec(path)

    assert spec.objects == {1: "operator", 2: "manipulated_object"}
    assert spec.cameras["cam0"][0].box == (1.0, 2.0, 10.0, 12.0)
    np.testing.assert_array_equal(spec.cameras["cam0"][1].point_labels, [1, 0])


def test_prompt_manifest_rejects_unknown_object(tmp_path):
    manifest = _prompt_manifest()
    manifest["cameras"]["cam0"][0]["object_id"] = 99
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unknown object"):
        load_prompt_spec(path)


def test_bitpacked_mask_chunk_and_random_access_round_trip(tmp_path):
    root = tmp_path / "sam2.1_hiera_small"
    camera = root / "cam0"
    camera.mkdir(parents=True)
    masks = np.zeros((2, 2, 4, 11), bool)
    masks[0, 0, 1:3, 2:7] = True
    masks[1, 1, :, 9:] = True
    path = camera / "masks_000000_000001.npz"
    _save_mask_chunk(
        path,
        frame_start=0,
        masks=masks,
        object_ids=[1, 2],
        object_labels=["operator", "manipulated_object"],
        area_px=masks.sum(axis=(2, 3)),
        mean_positive_logit=np.ones((2, 2), np.float32),
        overlap_px=np.zeros(2, np.int32),
    )
    (root / "meta.json").write_text(json.dumps({
        "schema": MASK_SCHEMA,
        "objects": [
            {"id": 1, "label": "operator"},
            {"id": 2, "label": "manipulated_object"},
        ],
        "cameras": {
            "cam0": {
                "frames": 2,
                "chunks": [{"file": path.name, "frame_start": 0, "frame_count": 2}],
            }
        },
    }))

    chunk = unpack_mask_chunk(path)
    np.testing.assert_array_equal(chunk["masks"], masks)
    ids, labels, frame = MaskArchive(root).frame("cam0", 1)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(labels, ["operator", "manipulated_object"])
    np.testing.assert_array_equal(frame, masks[1])
