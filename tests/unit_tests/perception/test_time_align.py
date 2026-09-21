"""
Temporal alignment of free-running cameras — ``viki.perception.triangulate``.

A "synced group" is index-aligned, not simultaneous. Each observation row
already carries its own capture time; these tests pin that the resampler uses it
to remove the first-order ``v*dt`` error, and that it declines rather than
extrapolates when the evidence is not there.
"""

from __future__ import annotations

import numpy as np

from viki.perception.triangulate import _time_align_uv

TICK = 33_333  # 30 fps


def _linear_track(n_frames, cam, t0, speed_px_per_us):
    """One camera moving a single landmark at a constant pixel rate."""
    uv, cams, fidx, ts = [], [], [], []
    for f in range(n_frames):
        t = t0 + f * TICK
        uv.append([[speed_px_per_us * t, 0.0]])
        cams.append(cam); fidx.append(f); ts.append(t)
    return uv, cams, fidx, ts


def _build(offset_us, n=4, speed=1e-3):
    """Two cameras on the same motion, camera B lagging the tick by offset."""
    uvA, cA, fA, tA = _linear_track(n, "a", 0, speed)
    uvB, cB, fB, tB = _linear_track(n, "b", offset_us, speed)
    uv = np.array(uvA + uvB, float)
    return (uv, np.array(cA + cB), np.array(fA + fB), np.array(tA + tB, np.int64),
            list(range(n)), [i * TICK for i in range(n)])


def test_two_skewed_cameras_are_brought_onto_one_instant():
    """Constant motion: after alignment both cameras must agree exactly.

    The target is the group's median capture time, so the two meet in the
    middle rather than both being dragged to the recorder's tick.
    """
    offset = 12_000  # 12 ms apart
    uv, cams, fidx, ts, frames, sync = _build(offset, n=5)
    out, stats = _time_align_uv(uv, cams, fidx, ts, frames, sync, max_gap_us=50_000)

    a_rows = [i for i, c in enumerate(cams) if c == "a"]
    b_rows = [i for i, c in enumerate(cams) if c == "b"]
    # Each camera moves towards the other, so the frames at either end lack a
    # bracketing neighbour on the side they need; the interior must agree.
    for ra, rb in list(zip(a_rows, b_rows))[1:-1]:
        assert np.allclose(out[ra, 0], out[rb, 0], atol=1e-6)
    assert stats["rows_shifted"] > 0
    # each camera travels about half the skew
    assert abs(stats["abs_dt_ms_median"] - offset / 2000.0) < 0.5


def test_the_input_array_is_never_modified():
    uv, cams, fidx, ts, frames, sync = _build(12_000)
    before = uv.copy()
    _time_align_uv(uv, cams, fidx, ts, frames, sync, max_gap_us=50_000)
    assert np.array_equal(uv, before)


def test_a_camera_already_on_the_tick_is_untouched():
    uv, cams, fidx, ts, frames, sync = _build(0)
    out, stats = _time_align_uv(uv, cams, fidx, ts, frames, sync, max_gap_us=50_000)
    assert np.array_equal(out, uv)
    assert stats["rows_shifted"] == 0


def test_it_refuses_to_interpolate_across_too_large_a_gap():
    """Past a tracking gap, interpolation would invent motion."""
    uv, cams, fidx, ts, frames, sync = _build(12_000)
    out, stats = _time_align_uv(uv, cams, fidx, ts, frames, sync, max_gap_us=1_000)
    assert np.array_equal(out, uv)
    assert stats["rows_shifted"] == 0
    assert stats["rows_left_as_is"] > 0


def test_a_non_finite_neighbour_does_not_poison_the_row():
    uv, cams, fidx, ts, frames, sync = _build(12_000)
    b_rows = [i for i, c in enumerate(cams) if c == "b"]
    uv[b_rows[1], 0] = [np.nan, np.nan]
    out, _ = _time_align_uv(uv, cams, fidx, ts, frames, sync, max_gap_us=50_000)
    assert np.isfinite(out[b_rows[0], 0]).all()
