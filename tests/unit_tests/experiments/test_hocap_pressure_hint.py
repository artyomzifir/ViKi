"""Geometry and view-fusion contracts for the opt-in pressure diagnostic."""
import numpy as np
import pytest

from experiments.hocap_pressure_hint import hand_crop, digit_regions, fuse_views, PRESSURE_VALUES


def test_crop_matches_upstream_scale_and_clamps_to_actual_image():
    points = np.tile([100., 100.], (21, 1))
    points[0], points[-1] = [80, 90], [120, 110]
    assert hand_crop(points, 640, 480) == (70, 70, 130, 130)
    points -= [90, 90]
    assert hand_crop(points, 640, 480) == (0, 0, 40, 40)


def test_regions_keep_finger_identity_with_overlapping_joint_endpoints():
    points = np.zeros((21, 2))
    for finger in range(5):
        points[1 + finger * 4:5 + finger * 4] = [[10 + finger * 20, y] for y in (10, 20, 30, 40)]
    labels = digit_regions(points, np.full(15, 3.), shape=(60, 110))
    for finger in range(5):
        assert labels[25, 10 + finger * 20] == finger
    assert labels[55, 55] == -1


def test_two_view_fusion_averages_strongest_views_and_is_permutation_invariant():
    values = np.zeros((8, 2, 5))
    values[0, :, 3] = 10
    values[6, :, 3] = 6
    fused = fuse_views(values)
    np.testing.assert_array_equal(fused[:, 3], [8, 8])
    np.testing.assert_array_equal(fused, fuse_views(values[::-1]))
    assert fused[:, :3].sum() == 0
    with pytest.raises(ValueError):
        fuse_views(values[:1])


def test_pressure_values_preserve_published_nine_class_midpoints():
    np.testing.assert_array_equal(PRESSURE_VALUES, [0, .75, 1.5, 3, 6, 12, 24, 48, 96])
