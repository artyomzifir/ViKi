"""Placement adjustments start after pickup; closure timing stays local."""

import numpy as np

from experiments.quality_search import shifted_closing, transfer_fraction


def test_transfer_fraction_preserves_approach_and_pickup():
    fraction = transfer_fraction(3, 8, [1, 2, 6, 7])
    np.testing.assert_allclose(fraction[:3 + 3], 0.0, atol=1e-12)
    assert fraction[3 + 4] == 0.5
    np.testing.assert_allclose(fraction[3 + 6:], 1.0)


def test_shifted_closing_preserves_release_command():
    drive = np.array([0., 0., 0.2, 0.4, 0.6, 0.6, 0.1, 0.])
    early = shifted_closing(drive, 1, 4, 1)
    late = shifted_closing(drive, 1, 4, -1)
    assert early[2] > drive[2]
    assert late[2] < drive[2]
    np.testing.assert_allclose(early[5:], drive[5:])
    np.testing.assert_allclose(late[5:], drive[5:])
