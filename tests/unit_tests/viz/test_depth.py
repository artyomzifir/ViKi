"""
Tests for depth visualization utilities.
Verifies colorization, undistortion, and temporal stabilization.
"""

import pytest

# ...
import numpy as np
import cv2
from viki.viz.depth import DepthColorizer, Undistorter, DepthStabilizer


def test_depth_colorizer_initialization():
    """Verify correct default values for DepthColorizer."""
    colorizer = DepthColorizer(alpha=0.1, min_valid_fraction=0.1)
    assert colorizer.alpha == 0.1
    assert colorizer.min_valid_fraction == 0.1
    assert colorizer.d_min == 0.0
    assert colorizer.d_max == 1.0
    assert colorizer._ema_initialised is False


def test_depth_colorizer_valid_frame():
    """Verify that a valid depth frame is correctly colorized into a BGR image."""
    colorizer = DepthColorizer()
    # Create a depth frame with values between 100 and 1000
    depth = np.random.randint(100, 1000, (480, 640), dtype=np.uint16)

    img = colorizer.colorize(depth)

    assert img is not None
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8
    assert colorizer._ema_initialised is True
    assert 0 <= colorizer.d_min <= 1000
    assert 0 <= colorizer.d_max <= 1000


def test_depth_colorizer_empty_frame():
    """Verify that empty frames are handled by returning None or the last good frame."""
    colorizer = DepthColorizer(min_valid_fraction=0.1)

    # 1. First frame is empty -> should return None
    depth_empty = np.zeros((480, 640), dtype=np.uint16)
    assert colorizer.colorize(depth_empty) is None

    # 2. Provide a good frame
    depth_good = np.random.randint(100, 1000, (480, 640), dtype=np.uint16)
    img_good = colorizer.colorize(depth_good)
    assert img_good is not None

    # 3. Next frame is empty -> should return the last good frame
    assert np.array_equal(colorizer.colorize(depth_empty), img_good)


def test_depth_colorizer_ema_update():
    """Verify that the colorizer's depth range (min/max) updates via EMA."""
    colorizer = DepthColorizer(alpha=0.5)

    # Frame 1: range [100, 200]
    depth1 = np.zeros((10, 10), dtype=np.uint16)
    depth1[0, 0] = 100
    depth1[0, 1] = 200
    depth1[1:] = 150
    colorizer.colorize(depth1)

    d_min_1 = colorizer.d_min
    d_max_1 = colorizer.d_max

    # Frame 2: range [300, 400]
    depth2 = np.zeros((10, 10), dtype=np.uint16)
    depth2[0, 0] = 300
    depth2[0, 1] = 400
    depth2[1:] = 350
    colorizer.colorize(depth2)

    # EMA: new = 0.5 * p + 0.5 * old
    # Note: np.percentile might differ slightly depending on interpolation
    assert colorizer.d_min > d_min_1
    assert colorizer.d_max > d_max_1


def test_undistorter_caching():
    """Verify that undistortion maps are computed once and then cached."""
    mtx = np.eye(3)
    dist = np.zeros(5)
    undistorter = Undistorter(mtx, dist)

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    assert undistorter._map1 is None
    undistorter.apply(img)
    assert undistorter._map1 is not None

    map1_before = undistorter._map1.copy()
    undistorter.apply(img)
    assert np.array_equal(undistorter._map1, map1_before)


def test_depth_stabilizer():
    """Verify that temporal median filtering effectively removes depth spikes."""
    stabilizer = DepthStabilizer(window_size=3)

    # Create 3 frames with a spike in one
    f1 = np.full((10, 10), 100, dtype=np.uint16)
    f2 = np.full((10, 10), 100, dtype=np.uint16)
    f2[0, 0] = 1000  # Spike
    f3 = np.full((10, 10), 100, dtype=np.uint16)

    # First frame: returns as is
    assert np.array_equal(stabilizer.stabilize(f1), f1)

    # Second frame: median of f1, f2
    res2 = stabilizer.stabilize(f2)
    assert res2[0, 0] == 550  # Median of 100 and 1000 is 550.0
    # Actually np.median([100, 1000]) = 550.0

    # Third frame: median of f1, f2, f3
    res3 = stabilizer.stabilize(f3)
    assert res3[0, 0] == 100  # Median of [100, 1000, 100] is 100
