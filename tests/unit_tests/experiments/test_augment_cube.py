"""Object-relative offsets preserve pickup and placement anchors."""

import numpy as np
import pytest

from experiments.augment_cube import path_offsets


def test_path_offsets_follow_red_then_blend_to_green():
    red = np.array([0.02, -0.01])
    green = np.array([-0.015, 0.025])
    result = path_offsets(5, 20, [3, 7, 14, 18], red, green)
    np.testing.assert_allclose(result[0], [0, 0, 0])
    np.testing.assert_allclose(result[4], [*red, 0])
    np.testing.assert_allclose(result[5 + 7], [*red, 0])
    np.testing.assert_allclose(result[5 + 14], [*green, 0])
    np.testing.assert_allclose(result[-1], [*green, 0])
    fraction = 3 / 7
    np.testing.assert_allclose(result[5 + 10],
                               [*((1 - fraction) * red + fraction * green), 0])


def test_path_offsets_reject_invalid_phase_bounds():
    with pytest.raises(ValueError, match="phase bounds"):
        path_offsets(5, 20, [3, 14, 7, 18], np.zeros(2), np.zeros(2))
