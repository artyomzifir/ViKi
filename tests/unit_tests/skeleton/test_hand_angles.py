"""
Tests for viki.skeleton.hand_angles.compute_hand_angles.

Uses synthetic 3-D points and simple rotation matrices to pin the sign
and magnitude of each returned angle.
"""

from __future__ import annotations

import numpy as np
import pytest

from viki.skeleton.hand_angles import HandAngles, compute_hand_angles
from viki.skeleton.models import LM


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _rot(axis: str, deg: float) -> np.ndarray:
    """Standard right-handed rotation matrix around x, y, or z."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"unknown axis: {axis}")


def _neutral_points() -> dict[LM, np.ndarray]:
    """
    Neutral pose in the world frame:

        SHOULDER at (0, 1, 0)   ← "up" from elbow
        ELBOW    at (0, 0, 0)
        WRIST    at (1, 0, 0)   ← forearm along +x
        MIDDLE_MCP at (2, 0, 0) ← hand continues forearm
        THUMB_CMC  at (1, 0, -1) ← thumb sticks out in -z from wrist

    In the resulting forearm-local basis x=+x, y=+y, z=+z. Expected angles:
        flexion = 0, deviation = 0, roll = 0
        forearm_axis = (1, 0, 0)
        palm_normal  = (0, 1, 0)
    """
    return {
        LM.SHOULDER: np.array([0.0, 1.0, 0.0]),
        LM.ELBOW: np.array([0.0, 0.0, 0.0]),
        LM.WRIST: np.array([1.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([2.0, 0.0, 0.0]),
        LM.THUMB_CMC: np.array([1.0, 0.0, -1.0]),
    }


def _rotate_hand(points: dict[LM, np.ndarray], R: np.ndarray) -> None:
    """Rotate MIDDLE_MCP and THUMB_CMC around WRIST in place, by R."""
    wrist = points[LM.WRIST]
    for lm in (LM.MIDDLE_MCP, LM.THUMB_CMC):
        points[lm] = wrist + R @ (points[lm] - wrist)


# ─────────────────────────────────────────────────────────────
# Neutral pose
# ─────────────────────────────────────────────────────────────


def test_neutral_pose_yields_zero_angles() -> None:
    out = compute_hand_angles(_neutral_points())
    assert out.valid is True
    assert out.flexion_deg == pytest.approx(0.0, abs=1e-6)
    assert out.deviation_deg == pytest.approx(0.0, abs=1e-6)
    assert out.roll_deg == pytest.approx(0.0, abs=1e-6)


def test_neutral_forearm_axis_is_unit_x() -> None:
    out = compute_hand_angles(_neutral_points())
    assert out.valid
    np.testing.assert_allclose(out.forearm_axis, [1.0, 0.0, 0.0], atol=1e-6)


def test_neutral_palm_normal_is_unit_y() -> None:
    """palm_normal = normalize(cross(+x, -z)) = +y."""
    out = compute_hand_angles(_neutral_points())
    assert out.valid
    np.testing.assert_allclose(out.palm_normal, [0.0, 1.0, 0.0], atol=1e-6)


# ─────────────────────────────────────────────────────────────
# Isolated rotations
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("flex_deg", [15.0, 45.0, -30.0, 75.0])
def test_flexion_matches_applied_z_rotation(flex_deg: float) -> None:
    """Rotate hand around local +z by φ → flexion == +φ; deviation, roll = 0."""
    pts = _neutral_points()
    _rotate_hand(pts, _rot("z", flex_deg))

    out = compute_hand_angles(pts)
    assert out.valid
    assert out.flexion_deg == pytest.approx(flex_deg, abs=1e-6)
    assert out.deviation_deg == pytest.approx(0.0, abs=1e-6)
    assert out.roll_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dev_deg", [10.0, -20.0, 60.0])
def test_deviation_matches_applied_y_rotation(dev_deg: float) -> None:
    """Rotate hand around local +y by φ → deviation == +φ; flexion, roll = 0."""
    pts = _neutral_points()
    _rotate_hand(pts, _rot("y", dev_deg))

    out = compute_hand_angles(pts)
    assert out.valid
    assert out.deviation_deg == pytest.approx(dev_deg, abs=1e-6)
    assert out.flexion_deg == pytest.approx(0.0, abs=1e-6)
    assert out.roll_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("roll_deg", [30.0, -45.0, 90.0])
def test_roll_matches_applied_x_rotation(roll_deg: float) -> None:
    """Rotate hand around forearm axis (+x) by φ → roll == +φ; flexion, deviation = 0."""
    pts = _neutral_points()
    _rotate_hand(pts, _rot("x", roll_deg))

    out = compute_hand_angles(pts)
    assert out.valid
    assert out.roll_deg == pytest.approx(roll_deg, abs=1e-6)
    assert out.flexion_deg == pytest.approx(0.0, abs=1e-6)
    assert out.deviation_deg == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# Invariance to overall arm/body pose in the world
# ─────────────────────────────────────────────────────────────


def test_world_rotation_does_not_change_local_angles() -> None:
    """
    Rotating the whole scene in the world must not change forearm-local
    angles. This is the whole point of the local basis.
    """
    pts = _neutral_points()
    _rotate_hand(pts, _rot("z", 25.0))  # bake a known flexion in
    base = compute_hand_angles(pts)

    W = _rot("x", 40.0) @ _rot("y", -33.0) @ _rot("z", 12.0)
    rotated = {lm: W @ p for lm, p in pts.items()}
    out = compute_hand_angles(rotated)

    assert out.valid
    assert out.flexion_deg == pytest.approx(base.flexion_deg, abs=1e-6)
    assert out.deviation_deg == pytest.approx(base.deviation_deg, abs=1e-6)
    assert out.roll_deg == pytest.approx(base.roll_deg, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# Invalid inputs
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "missing",
    [LM.WRIST, LM.ELBOW, LM.SHOULDER, LM.THUMB_CMC, LM.MIDDLE_MCP],
)
def test_missing_required_landmark_returns_invalid(missing: LM) -> None:
    pts = _neutral_points()
    del pts[missing]

    out = compute_hand_angles(pts)
    assert out.valid is False


@pytest.mark.parametrize(
    "with_nan",
    [LM.WRIST, LM.ELBOW, LM.SHOULDER, LM.THUMB_CMC, LM.MIDDLE_MCP],
)
def test_nan_in_required_landmark_returns_invalid(with_nan: LM) -> None:
    pts = _neutral_points()
    pts[with_nan] = np.array([np.nan, 0.0, 0.0])

    out = compute_hand_angles(pts)
    assert out.valid is False


def test_straight_arm_returns_invalid() -> None:
    """
    SHOULDER placed colinearly with the forearm direction from the elbow
    makes `up_ref` parallel to `x`, so `y` cannot be resolved.
    """
    pts = _neutral_points()
    pts[LM.SHOULDER] = pts[LM.ELBOW] - (pts[LM.WRIST] - pts[LM.ELBOW])

    out = compute_hand_angles(pts)
    assert out.valid is False


def test_zero_forearm_returns_invalid() -> None:
    """WRIST and ELBOW at the same location → forearm axis undefined."""
    pts = _neutral_points()
    pts[LM.WRIST] = pts[LM.ELBOW].copy()

    out = compute_hand_angles(pts)
    assert out.valid is False


def test_thumb_colinear_with_middle_returns_invalid() -> None:
    """
    If THUMB_CMC lies on the WRIST→MIDDLE_MCP line, palm plane cannot be
    resolved (cross product ≈ 0).
    """
    pts = _neutral_points()
    wrist = pts[LM.WRIST]
    # Put thumb on the same ray as middle.
    pts[LM.THUMB_CMC] = wrist + 0.5 * (pts[LM.MIDDLE_MCP] - wrist)

    out = compute_hand_angles(pts)
    assert out.valid is False


def test_invalid_result_is_fully_nan() -> None:
    """Every scalar/vector on an invalid HandAngles must be NaN."""
    pts = _neutral_points()
    del pts[LM.WRIST]
    out = compute_hand_angles(pts)

    assert out.valid is False
    assert np.isnan(out.flexion_deg)
    assert np.isnan(out.deviation_deg)
    assert np.isnan(out.roll_deg)
    assert np.isnan(out.palm_normal).all()
    assert np.isnan(out.forearm_axis).all()


# ─────────────────────────────────────────────────────────────
# Shape / typing
# ─────────────────────────────────────────────────────────────


def test_valid_result_vectors_are_finite_and_unit_length() -> None:
    out = compute_hand_angles(_neutral_points())
    assert out.valid
    assert np.all(np.isfinite(out.forearm_axis))
    assert np.all(np.isfinite(out.palm_normal))
    assert np.linalg.norm(out.forearm_axis) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(out.palm_normal) == pytest.approx(1.0, abs=1e-6)


def test_result_type_is_handangles() -> None:
    out = compute_hand_angles(_neutral_points())
    assert isinstance(out, HandAngles)
