"""
Tests for skeleton point fusion.
Verifies combining landmarks from multiple cameras into a single world-space frame.
"""

import pytest

# ...
import numpy as np
from viki.skeleton.fusion import fuse
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame
from viki.calibration.models import CalibrationExtrinsics


def test_fuse_no_observations():
    """Verify that fusing with no valid observations returns a SkeletonFrame with NaNs."""
    # No landmarks provided
    result = fuse(
        dev_ids=["cam1"], lms={"cam1": None}, extrinsics={}, timestamp_us=1000
    )
    assert isinstance(result, SkeletonFrame)
    assert np.isnan(result.points[LM(0)]).any()


def test_fuse_single_camera():
    """Verify fusion with a single camera (should be a simple transform)."""
    # Identity transform
    extr = CalibrationExtrinsics(rvec=np.zeros(3), tvec=np.zeros(3))
    lm = Landmarks3D(
        points={LM(i): np.array([1.0, 2.0, 3.0]) for i in range(LM.N)},
        device_id="cam1",
        timestamp_us=1000,
    )

    result = fuse(
        dev_ids=["cam1"], lms={"cam1": lm}, extrinsics={"cam1": extr}, timestamp_us=1000
    )

    np.testing.assert_allclose(result.points[LM(0)], [1.0, 2.0, 3.0])


def test_fuse_multi_camera_mean():
    """Verify that fusion of multiple cameras results in the average world position."""
    # Two cameras, both seeing the same point in world space
    # Cam 1: Identity
    extr1 = CalibrationExtrinsics(rvec=np.zeros(3), tvec=np.zeros(3))
    lm1 = Landmarks3D(
        points={LM(i): np.array([1.0, 1.0, 1.0]) for i in range(LM.N)},
        device_id="cam1",
        timestamp_us=1000,
    )

    # Cam 2: Translated by 1m in X
    tvec_2 = np.array([1.0, 0.0, 0.0])
    extr2 = CalibrationExtrinsics(rvec=np.zeros(3), tvec=tvec_2)
    # In Cam 2's local space, the point [1,1,1] world is [0,1,1]
    lm2 = Landmarks3D(
        points={LM(i): np.array([0.0, 1.0, 1.0]) for i in range(LM.N)},
        device_id="cam2",
        timestamp_us=1000,
    )

    result = fuse(
        dev_ids=["cam1", "cam2"],
        lms={"cam1": lm1, "cam2": lm2},
        extrinsics={"cam1": extr1, "cam2": extr2},
        timestamp_us=1000,
    )

    # Both should project to [1, 1, 1] in world space
    np.testing.assert_allclose(result.points[LM(0)], [1.0, 1.0, 1.0])


def test_fuse_weighted_confidence():
    """Verify that fusion uses confidence weights to average positions."""
    extr1 = CalibrationExtrinsics(rvec=np.zeros(3), tvec=np.zeros(3))
    lm1 = Landmarks3D(
        points={LM(i): np.array([0.0, 0.0, 0.0]) for i in range(LM.N)},
        device_id="cam1",
        timestamp_us=1000,
    )

    extr2 = CalibrationExtrinsics(rvec=np.zeros(3), tvec=np.zeros(3))
    lm2 = Landmarks3D(
        points={LM(i): np.array([1.0, 1.0, 1.0]) for i in range(LM.N)},
        device_id="cam2",
        timestamp_us=1000,
    )

    # Cam 2 has much higher confidence
    confidences = {
        "cam1": {LM(i): 0.1 for i in range(LM.N)},
        "cam2": {LM(i): 0.9 for i in range(LM.N)},
    }

    result = fuse(
        dev_ids=["cam1", "cam2"],
        lms={"cam1": lm1, "cam2": lm2},
        extrinsics={"cam1": extr1, "cam2": extr2},
        timestamp_us=1000,
        confidences=confidences,
    )

    # Weighted mean: (0*0.1 + 1*0.9) / (0.1 + 0.9) = 0.9
    np.testing.assert_allclose(result.points[LM(0)], [0.9, 0.9, 0.9])
