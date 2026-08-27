"""
compute_end_effector_pose / compute_palm_rotation.

Palm frame (current definition — knuckle spread, not the thumb):
    x = normalise(MIDDLE_MCP - WRIST)
    z = normalise((MIDDLE_MCP - WRIST) x (PINKY_MCP - INDEX_MCP))
    y = z x x
Required landmarks: WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP. When one is missing
the function falls back to the centroid of the available palm landmarks with an
identity rotation (still ``valid``); it is only invalid when no palm landmark is
finite.
"""

import numpy as np
import pytest

from viki.contracts import LM
from viki.perception.hand_angles import compute_end_effector_pose, compute_palm_rotation

_TS = 123456
_REQUIRED = (LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.PINKY_MCP)
_R_NEUTRAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _neutral():
    return {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([0.0, 1.0, 0.0]),
        LM.INDEX_MCP: np.array([0.5, 0.9, 0.0]),
        LM.PINKY_MCP: np.array([-0.5, 0.9, 0.0]),
    }


def _rotate(points, R):
    return {lm: R @ p for lm, p in points.items()}


# ── primary path ──────────────────────────────────────────────────────


def test_neutral_pose_matches_hand_construction():
    out = compute_end_effector_pose(_neutral(), _TS)
    assert out.valid is True
    assert out.timestamp_us == _TS
    np.testing.assert_allclose(out.position, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(out.R_world_palm, _R_NEUTRAL, atol=1e-6)


def test_R_is_a_proper_rotation():
    R = compute_end_effector_pose(_neutral(), _TS).R_world_palm.astype(np.float64)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("axis,deg", [("x", 30.0), ("y", -47.0), ("z", 90.0)])
def test_world_rotation_composes_on_the_left(axis, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    W = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]
    out = compute_end_effector_pose(_rotate(_neutral(), W), _TS)
    np.testing.assert_allclose(out.R_world_palm, W @ _R_NEUTRAL, atol=1e-5)


def test_position_follows_the_wrist():
    pts = {lm: p + np.array([1.0, 2.0, 3.0]) for lm, p in _neutral().items()}
    out = compute_end_effector_pose(pts, _TS)
    np.testing.assert_allclose(out.position, [1, 2, 3], atol=1e-6)


# ── fallback + invalid ────────────────────────────────────────────────


@pytest.mark.parametrize("missing", _REQUIRED)
def test_missing_required_landmark_falls_back_to_centroid(missing):
    pts = _neutral()
    del pts[missing]
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is True
    np.testing.assert_allclose(out.R_world_palm, np.eye(3), atol=1e-6)
    finite = [p for p in pts.values() if np.all(np.isfinite(p))]
    np.testing.assert_allclose(out.position, np.mean(finite, axis=0), atol=1e-5)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_required_landmark_falls_back(bad):
    pts = _neutral()
    pts[LM.PINKY_MCP] = np.array([bad, bad, bad])
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is True
    np.testing.assert_allclose(out.R_world_palm, np.eye(3), atol=1e-6)


def test_no_palm_landmarks_is_invalid():
    out = compute_end_effector_pose({}, _TS)
    assert out.valid is False
    assert np.isnan(out.position).all()
    assert np.isnan(out.R_world_palm).all()
    assert out.timestamp_us == _TS


def test_collinear_geometry_falls_back():
    # spread parallel to fwd -> cross product ~0 -> no palm frame.
    pts = {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([0.0, 1.0, 0.0]),
        LM.INDEX_MCP: np.array([0.0, 0.4, 0.0]),
        LM.PINKY_MCP: np.array([0.0, 0.6, 0.0]),
    }
    out = compute_end_effector_pose(pts, _TS)
    np.testing.assert_allclose(out.R_world_palm, np.eye(3), atol=1e-6)


# ── compute_palm_rotation helper ──────────────────────────────────────


def test_compute_palm_rotation_matches_neutral():
    n = _neutral()
    R = compute_palm_rotation(n[LM.WRIST], n[LM.INDEX_MCP], n[LM.MIDDLE_MCP], n[LM.PINKY_MCP])
    np.testing.assert_allclose(R, _R_NEUTRAL, atol=1e-6)


def test_compute_palm_rotation_none_on_degenerate():
    z = np.zeros(3)
    assert compute_palm_rotation(z, z, z, z) is None
