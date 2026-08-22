"""
Tests for the landmark interpolation system.
Verifies filling of missing data (NaNs) using linear interpolation across time.
"""

import pytest

# ...
import math
from viki.optimization.interpolation.interpolation import Interpolator


def test_interpolator_no_nans():
    """Verify that data without gaps is returned unchanged."""
    interpolator = Interpolator()
    data = [
        {"ts": 1000, "landmarks": {0: [1.0, 2.0, 3.0]}},
        {"ts": 2000, "landmarks": {0: [2.0, 3.0, 4.0]}},
    ]
    result = interpolator.process(data)
    assert result == data


def test_interpolator_basic_gap():
    """Verify simple linear interpolation between two known points."""
    interpolator = Interpolator()
    # Landmark 0 has a gap at ts=2000
    data = [
        {"ts": 1000, "landmarks": {0: [1.0, 1.0, 1.0]}},
        {"ts": 2000, "landmarks": {0: [float("nan"), float("nan"), float("nan")]}},
        {"ts": 3000, "landmarks": {0: [3.0, 3.0, 3.0]}},
    ]

    result = interpolator.process(data)

    # Expected interpolation at ts=2000: (1.0 + 3.0) / 2 = 2.0
    interp_vec = result[1]["landmarks"][0]
    assert interp_vec == [2.0, 2.0, 2.0]


def test_interpolator_multiple_gaps():
    """Verify interpolation across multiple consecutive missing frames."""
    interpolator = Interpolator()
    data = [
        {"ts": 1000, "landmarks": {0: [0.0, 0.0, 0.0]}},
        {"ts": 2000, "landmarks": {0: [float("nan"), float("nan"), float("nan")]}},
        {"ts": 3000, "landmarks": {0: [float("nan"), float("nan"), float("nan")]}},
        {"ts": 4000, "landmarks": {0: [30.0, 30.0, 30.0]}},
    ]

    result = interpolator.process(data)

    # ts=2000: weight = (2000-1000)/(4000-1000) = 1/3. value = 0 + 1/3 * 30 = 10
    # ts=3000: weight = (3000-1000)/(4000-1000) = 2/3. value = 0 + 2/3 * 30 = 20
    assert result[1]["landmarks"][0] == [10.0, 10.0, 10.0]
    assert result[2]["landmarks"][0] == [20.0, 20.0, 20.0]


def test_interpolator_no_start_value():
    """Verify that gaps at the start of a sequence cannot be interpolated."""
    interpolator = Interpolator()
    # Gap at the beginning -> cannot interpolate
    data = [
        {"ts": 1000, "landmarks": {0: [float("nan"), float("nan"), float("nan")]}},
        {"ts": 2000, "landmarks": {0: [1.0, 1.0, 1.0]}},
    ]
    result = interpolator.process(data)
    assert math.isnan(result[0]["landmarks"][0][0])


def test_interpolator_no_end_value():
    """Verify that gaps at the end of a sequence remain pending."""
    interpolator = Interpolator()
    # Gap at the end -> cannot interpolate (pending frames stay pending)
    data = [
        {"ts": 1000, "landmarks": {0: [1.0, 1.0, 1.0]}},
        {"ts": 2000, "landmarks": {0: [float("nan"), float("nan"), float("nan")]}},
    ]
    result = interpolator.process(data)
    assert math.isnan(result[1]["landmarks"][0][0])


def test_interpolator_mixed_nans():
    """Verify handling of frames where only some dimensions are NaN."""
    interpolator = Interpolator()
    data = [
        {"ts": 1000, "landmarks": {0: [1.0, float("nan"), 3.0]}},  # Frame 1
        {"ts": 2000, "landmarks": {0: [float("nan"), 2.0, float("nan")]}},  # Frame 2
        {"ts": 3000, "landmarks": {0: [3.0, 3.0, 5.0]}},  # Frame 3
    ]

    # Process:
    # Frame 1: contains NaN -> cannot be 'prev_known' for any idx (due to any(isnan))
    # Frame 2: contains NaN -> cannot be 'prev_known'
    # Frame 3: known.

    # Wait, the implementation says:
    # if any(math.isnan(v) for v in vec):
    #     if idx in prev_known_vecs:
    #         pending.setdefault(idx, []).append(frame)
    #     continue

    # Since Frame 1 and 2 both have NaNs, and prev_known_vecs is empty,
    # they are simply skipped and NOT added to pending.

    result = interpolator.process(data)
    assert math.isnan(result[0]["landmarks"][0][1])
    assert math.isnan(result[1]["landmarks"][0][0])
