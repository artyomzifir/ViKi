"""viki.prepare.interpolate.fill_se3_spline."""

import numpy as np

from viki.prepare.interpolate import fill_linear, fill_se3_spline


def _traj(series: np.ndarray) -> np.ndarray:
    """(T,) -> (T, 1, 3) with the same series on every coord."""
    return np.repeat(series[:, None, None], 3, axis=2)


def test_spline_recovers_a_smooth_curve():
    t = np.linspace(0, 2 * np.pi, 40)
    clean = np.sin(t)
    gappy = clean.copy()
    gappy[10:15] = np.nan
    gappy[28] = np.nan
    out = fill_se3_spline(_traj(gappy))[:, 0, 0]
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[10:15], clean[10:15], atol=0.02)


def test_falls_back_to_linear_with_few_samples():
    s = np.full(10, np.nan)
    s[0], s[9] = 0.0, 9.0  # only 2 valid -> linear
    out = fill_se3_spline(_traj(s))[:, 0, 0]
    np.testing.assert_allclose(out, np.linspace(0, 9, 10), atol=1e-9)


def test_all_nan_column_left_untouched():
    s = np.full(8, np.nan)
    out = fill_se3_spline(_traj(s))
    assert np.isnan(out).all()


def test_no_gaps_is_identity():
    s = np.arange(12, dtype=float)
    tr = _traj(s)
    np.testing.assert_array_equal(fill_se3_spline(tr), tr)


def test_max_gap_fills_short_run_but_preserves_long_occlusion():
    s = np.arange(12, dtype=float)
    s[2:4] = np.nan
    s[6:10] = np.nan
    out = fill_se3_spline(_traj(s), max_gap=2)[:, 0, 0]
    np.testing.assert_allclose(out[2:4], [2.0, 3.0], atol=1e-9)
    assert np.isnan(out[6:10]).all()


def test_fill_linear_still_exported():
    s = np.array([0.0, np.nan, 2.0])
    np.testing.assert_allclose(fill_linear(_traj(s))[:, 0, 0], [0, 1, 2])
