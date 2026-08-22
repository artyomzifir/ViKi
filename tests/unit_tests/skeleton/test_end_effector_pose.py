"""
Tests for viki.skeleton.hand_angles.compute_end_effector_pose.

Pins the palm-frame construction, the ROS/URDF RPY convention
(R = Rz(yaw) · Ry(pitch) · Rx(roll), extrinsic XYZ), the SO(3)
invariants of R_world_palm, and the failure modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.models import LM, EndEffectorPose


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_TS = 123_456  # arbitrary timestamp for all cases


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


def _rzyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """R = Rz(yaw) · Ry(pitch) · Rx(roll) — the extraction convention."""
    return _rot("z", yaw_deg) @ _rot("y", pitch_deg) @ _rot("x", roll_deg)


def _neutral_points() -> dict[LM, np.ndarray]:
    """
    Neutral pose in the world frame:
        WRIST      at (0, 0, 0)
        MIDDLE_MCP at (0, 1, 0)
        THUMB_CMC  at (1, 0, 0)

    Expected palm frame:
        x_palm = normalise((0,1,0))         = (0, 1, 0)
        z_palm = normalise((0,1,0)×(1,0,0)) = (0, 0, -1)
        y_palm = (0,0,-1) × (0,1,0)         = (1, 0, 0)
    """
    return {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([0.0, 1.0, 0.0]),
        LM.THUMB_CMC: np.array([1.0, 0.0, 0.0]),
        # Include not-required landmarks to prove they are ignored.
        LM.ELBOW: np.array([0.0, -1.0, 0.0]),
        LM.SHOULDER: np.array([0.0, -2.0, 0.0]),
    }


def _identity_R_points() -> dict[LM, np.ndarray]:
    """
    Pose whose palm frame lines up with the world axes:
        x_palm = +x_world, y_palm = +y_world, z_palm = +z_world

    Verify by construction:
        to_middle = (1, 0, 0)  → x_palm = (1, 0, 0) ✓
        to_thumb  = (0, 1, 0)
        z_palm = normalise(cross((1,0,0), (0,1,0))) = (0, 0, 1) ✓
        y_palm = cross(z_palm, x_palm) = (0, 1, 0) ✓

    R_world_palm is the identity, so any applied W becomes W directly —
    exactly what the gimbal-lock tests need.
    """
    return {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([1.0, 0.0, 0.0]),
        LM.THUMB_CMC: np.array([0.0, 1.0, 0.0]),
    }


def _rotate_world(pts: dict[LM, np.ndarray], R: np.ndarray) -> dict[LM, np.ndarray]:
    return {lm: R @ p for lm, p in pts.items()}


def _translate(pts: dict[LM, np.ndarray], t: np.ndarray) -> dict[LM, np.ndarray]:
    return {lm: p + t for lm, p in pts.items()}


# ─────────────────────────────────────────────────────────────
# Neutral pose — pin palm-frame axes exactly
# ─────────────────────────────────────────────────────────────


def test_neutral_pose_is_valid() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    assert out.valid is True


def test_neutral_position_matches_wrist() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    np.testing.assert_allclose(out.position, [0.0, 0.0, 0.0], atol=1e-6)


def test_neutral_x_palm_column_matches_construction() -> None:
    """R[:, 0] must equal normalise(MIDDLE_MCP - WRIST)."""
    out = compute_end_effector_pose(_neutral_points(), _TS)
    np.testing.assert_allclose(out.R_world_palm[:, 0], [0.0, 1.0, 0.0], atol=1e-6)


def test_neutral_z_palm_column_matches_construction() -> None:
    """R[:, 2] must equal normalise(to_middle × to_thumb)."""
    out = compute_end_effector_pose(_neutral_points(), _TS)
    np.testing.assert_allclose(out.R_world_palm[:, 2], [0.0, 0.0, -1.0], atol=1e-6)


def test_neutral_y_palm_column_completes_right_handed_basis() -> None:
    """R[:, 1] must equal z × x (right-handed)."""
    out = compute_end_effector_pose(_neutral_points(), _TS)
    np.testing.assert_allclose(out.R_world_palm[:, 1], [1.0, 0.0, 0.0], atol=1e-6)


# ─────────────────────────────────────────────────────────────
# SO(3) invariants
# ─────────────────────────────────────────────────────────────


def test_R_is_orthonormal_on_neutral() -> None:
    R = compute_end_effector_pose(_neutral_points(), _TS).R_world_palm
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)


def test_R_has_positive_unit_determinant_on_neutral() -> None:
    R = compute_end_effector_pose(_neutral_points(), _TS).R_world_palm
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "W",
    [
        _rot("x", 37.0),
        _rot("y", -12.0),
        _rot("z", 88.0),
        _rot("x", 20.0) @ _rot("y", 40.0) @ _rot("z", 60.0),
        _rzyx(10.0, 25.0, -40.0),
    ],
)
def test_R_stays_in_SO3_under_arbitrary_world_rotations(W: np.ndarray) -> None:
    """R must remain orthonormal with det=+1 for any world-frame reorientation."""
    out = compute_end_effector_pose(_rotate_world(_neutral_points(), W), _TS)
    R = out.R_world_palm
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# World rotations must compose on the LEFT of R
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "W",
    [
        _rot("x", 25.0),
        _rot("y", -35.0),
        _rot("z", 50.0),
        _rot("x", 10.0) @ _rot("y", 20.0) @ _rot("z", -30.0),
    ],
)
def test_world_rotation_multiplies_R_on_the_left(W: np.ndarray) -> None:
    """
    R_world_palm maps palm→world. Applying W in the world means every world
    vector v becomes W @ v, so the new R is W @ R_base.
    """
    base = compute_end_effector_pose(_neutral_points(), _TS)
    rotated = compute_end_effector_pose(_rotate_world(_neutral_points(), W), _TS)
    np.testing.assert_allclose(rotated.R_world_palm, W @ base.R_world_palm, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# Position: translates with the scene, ignores rotations of frame
# ─────────────────────────────────────────────────────────────


def test_position_translates_with_the_scene() -> None:
    t = np.array([3.0, -2.0, 7.0])
    pts = _translate(_neutral_points(), t)
    out = compute_end_effector_pose(pts, _TS)
    np.testing.assert_allclose(out.position, t, atol=1e-6)


def test_translation_does_not_change_R() -> None:
    base = compute_end_effector_pose(_neutral_points(), _TS)
    shifted = compute_end_effector_pose(
        _translate(_neutral_points(), np.array([10.0, -5.0, 2.0])),
        _TS,
    )
    np.testing.assert_allclose(shifted.R_world_palm, base.R_world_palm, atol=1e-6)


def test_R_is_scale_invariant() -> None:
    """
    Scaling all landmarks by a positive constant preserves the palm frame.
    """
    base = compute_end_effector_pose(_neutral_points(), _TS)
    scaled = {lm: 4.7 * p for lm, p in _neutral_points().items()}
    out = compute_end_effector_pose(scaled, _TS)
    np.testing.assert_allclose(out.R_world_palm, base.R_world_palm, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# RPY convention: R = Rz(yaw)·Ry(pitch)·Rx(roll), extrinsic XYZ
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "roll, pitch, yaw",
    [
        (0.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (0.0, 25.0, 0.0),
        (0.0, 0.0, -40.0),
        (12.0, -18.0, 33.0),
        (-45.0, 30.0, 60.0),
    ],
)
def test_rpy_reconstructs_R(roll: float, pitch: float, yaw: float) -> None:
    """
    Round-trip: reconstruct R from returned rpy_deg using the same
    convention and confirm it matches R_world_palm.
    """
    W = _rzyx(roll, pitch, yaw)
    out = compute_end_effector_pose(_rotate_world(_neutral_points(), W), _TS)

    r, p, y = np.radians(out.rpy_deg.astype(np.float64))
    R_reconstructed = (
        _rot("z", np.degrees(y))
        @ _rot("y", np.degrees(p))
        @ _rot("x", np.degrees(r))
    )
    np.testing.assert_allclose(R_reconstructed, out.R_world_palm, atol=1e-5)


def test_pure_yaw_around_world_z_only_moves_yaw_from_the_baseline() -> None:
    """
    Applying a pure Rz(dyaw) on the LEFT changes R by that same Rz.
    Effect on rpy: yaw shifts by +dyaw modulo 360°, pitch and roll stay put.
    Test with a base pose whose baseline rpy we compute first.
    """
    base = compute_end_effector_pose(_neutral_points(), _TS)
    dyaw = 33.0
    rotated = compute_end_effector_pose(
        _rotate_world(_neutral_points(), _rot("z", dyaw)),
        _TS,
    )

    assert rotated.rpy_deg[0] == pytest.approx(base.rpy_deg[0], abs=1e-5)  # roll
    assert rotated.rpy_deg[1] == pytest.approx(base.rpy_deg[1], abs=1e-5)  # pitch
    # yaw wraps in (−180, 180]; compare via matrix instead of raw values to
    # sidestep wrap-around brittleness at ±180.
    expected_R = _rot("z", dyaw) @ base.R_world_palm
    np.testing.assert_allclose(rotated.R_world_palm, expected_R, atol=1e-6)


def test_rpy_is_finite_when_valid() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    assert out.valid
    assert np.all(np.isfinite(out.rpy_deg))


# ─────────────────────────────────────────────────────────────
# Gimbal lock at |pitch| = 90°
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_gimbal_lock_at_pitch_plus_minus_90_still_reconstructs_R(sign: float) -> None:
    """
    Even at gimbal lock the extracted rpy must reproduce R_world_palm when
    fed back through Rz(yaw)·Ry(pitch)·Rx(roll). The individual roll/yaw
    values are not independently unique in this regime — the convention
    picks roll=0 and lets yaw absorb the ambiguity.

    Uses the identity-R base pose so that the applied Ry(±90°) *is* the
    final R and drives the extractor into the singularity branch.
    """
    pitch = sign * 90.0
    W = _rzyx(0.0, pitch, 0.0)
    out = compute_end_effector_pose(_rotate_world(_identity_R_points(), W), _TS)

    assert out.valid
    assert np.all(np.isfinite(out.rpy_deg))
    assert abs(out.rpy_deg[1]) == pytest.approx(90.0, abs=1e-4)

    r, p, y = np.radians(out.rpy_deg.astype(np.float64))
    R_reconstructed = (
        _rot("z", np.degrees(y))
        @ _rot("y", np.degrees(p))
        @ _rot("x", np.degrees(r))
    )
    np.testing.assert_allclose(R_reconstructed, out.R_world_palm, atol=1e-4)


def test_gimbal_lock_convention_sets_roll_to_zero() -> None:
    """The extractor sets roll=0 at gimbal lock (documented convention)."""
    W = _rzyx(0.0, 90.0, 0.0)
    out = compute_end_effector_pose(_rotate_world(_identity_R_points(), W), _TS)
    assert out.rpy_deg[0] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# Elbow / shoulder are NOT required — differs from compute_hand_angles
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("not_required", [LM.ELBOW, LM.SHOULDER])
def test_missing_arm_landmark_does_not_invalidate_pose(not_required: LM) -> None:
    pts = _neutral_points()
    del pts[not_required]
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is True


@pytest.mark.parametrize("not_required", [LM.ELBOW, LM.SHOULDER])
def test_nan_in_non_required_landmark_does_not_invalidate_pose(
    not_required: LM,
) -> None:
    pts = _neutral_points()
    pts[not_required] = np.array([np.nan, np.nan, np.nan])
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is True


# ─────────────────────────────────────────────────────────────
# Invalid inputs
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", [LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP])
def test_missing_required_landmark_returns_invalid(missing: LM) -> None:
    pts = _neutral_points()
    del pts[missing]
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


@pytest.mark.parametrize("with_nan", [LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP])
def test_nan_in_required_landmark_returns_invalid(with_nan: LM) -> None:
    pts = _neutral_points()
    pts[with_nan] = np.array([np.nan, 0.0, 0.0])
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


@pytest.mark.parametrize("with_inf", [LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP])
def test_inf_in_required_landmark_returns_invalid(with_inf: LM) -> None:
    pts = _neutral_points()
    pts[with_inf] = np.array([np.inf, 0.0, 0.0])
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


def test_wrist_equals_middle_mcp_returns_invalid() -> None:
    """to_middle = 0 → x_palm undefined."""
    pts = _neutral_points()
    pts[LM.MIDDLE_MCP] = pts[LM.WRIST].copy()
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


def test_wrist_equals_thumb_returns_invalid() -> None:
    """to_thumb = 0 → cross(to_middle, to_thumb) = 0 → z_palm undefined."""
    pts = _neutral_points()
    pts[LM.THUMB_CMC] = pts[LM.WRIST].copy()
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


def test_thumb_colinear_with_middle_returns_invalid() -> None:
    """Thumb on the WRIST→MIDDLE_MCP line: palm normal is degenerate."""
    pts = _neutral_points()
    wrist = pts[LM.WRIST]
    pts[LM.THUMB_CMC] = wrist + 2.0 * (pts[LM.MIDDLE_MCP] - wrist)
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


def test_thumb_antiparallel_with_middle_returns_invalid() -> None:
    """Thumb along -(MIDDLE - WRIST): still colinear, cross = 0."""
    pts = _neutral_points()
    wrist = pts[LM.WRIST]
    pts[LM.THUMB_CMC] = wrist - 3.0 * (pts[LM.MIDDLE_MCP] - wrist)
    out = compute_end_effector_pose(pts, _TS)
    assert out.valid is False


def test_invalid_result_is_fully_nan() -> None:
    pts = _neutral_points()
    del pts[LM.WRIST]
    out = compute_end_effector_pose(pts, _TS)

    assert out.valid is False
    assert np.isnan(out.position).all()
    assert np.isnan(out.R_world_palm).all()
    assert np.isnan(out.rpy_deg).all()


def test_invalid_result_still_echoes_timestamp() -> None:
    """Downstream consumers align by timestamp; must survive the invalid branch."""
    pts = _neutral_points()
    del pts[LM.WRIST]
    out = compute_end_effector_pose(pts, timestamp_us=987_654)
    assert out.valid is False
    assert out.timestamp_us == 987_654


# ─────────────────────────────────────────────────────────────
# Shape / dtype / typing
# ─────────────────────────────────────────────────────────────


def test_result_type_is_end_effector_pose() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    assert isinstance(out, EndEffectorPose)


def test_result_shapes() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    assert out.position.shape == (3,)
    assert out.R_world_palm.shape == (3, 3)
    assert out.rpy_deg.shape == (3,)


def test_result_dtypes_are_float32() -> None:
    out = compute_end_effector_pose(_neutral_points(), _TS)
    assert out.position.dtype == np.float32
    assert out.R_world_palm.dtype == np.float32
    assert out.rpy_deg.dtype == np.float32


def test_valid_timestamp_is_echoed_unchanged() -> None:
    out = compute_end_effector_pose(_neutral_points(), timestamp_us=42_000)
    assert out.timestamp_us == 42_000


def test_as_dict_is_json_shaped() -> None:
    """Recorder relies on plain-python nested structure — no ndarrays leaking."""
    out = compute_end_effector_pose(_neutral_points(), _TS)
    d = out.as_dict()
    assert set(d.keys()) == {
        "position",
        "R_world_palm",
        "rpy_deg",
        "valid",
        "timestamp_us",
    }
    assert isinstance(d["position"], list) and len(d["position"]) == 3
    assert isinstance(d["R_world_palm"], list) and len(d["R_world_palm"]) == 3
    assert isinstance(d["R_world_palm"][0], list) and len(d["R_world_palm"][0]) == 3
    assert isinstance(d["rpy_deg"], list) and len(d["rpy_deg"]) == 3
    assert isinstance(d["valid"], bool)
    assert isinstance(d["timestamp_us"], int)
