"""Pure guards for K4A offline helpers that do not require libk4a."""

from __future__ import annotations

import numpy as np

from viki.perception.k4a_offline import K4ACalibration


def test_mask_warp_preserves_four_channels_and_chunks_a_fifth():
    calibration = K4ACalibration.__new__(K4ACalibration)
    calls = []

    def identity_warp(packed, depth):
        calls.append(packed.copy())
        return packed

    calibration._align_bgra_to_depth = identity_warp
    masks = np.zeros((5, 3, 4), bool)
    for index in range(5):
        masks[index, index % 3, index % 4] = True

    aligned = calibration.align_masks_to_depth(
        masks,
        np.full((3, 4), 1000, np.uint16),
    )

    np.testing.assert_array_equal(aligned, masks)
    assert len(calls) == 2
    assert calls[0].shape == (3, 4, 4)
    assert calls[1].shape == (3, 4, 4)
