"""``_sync_stats`` turns the recorded per-frame host-tick offsets into jitter
+ linear clock-drift numbers (paper §3.3: residual alignment is measured, not
assumed)."""

from viki.cameras.record import _DRIFT_BOUND_MS_PER_MIN, _sync_stats


def _ticks(n, fps=15, offsets=None):
    period = 1_000_000 // fps
    return [
        {"sync_us": i * period, "offsets_us": {d: f(i) for d, f in offsets.items()}}
        for i in range(n)
    ]


def test_flat_offset_is_bounded_zero_drift():
    ts = _ticks(300, offsets={"kinect_0": lambda i: 120, "kinect_1": lambda i: -80})
    s = _sync_stats(ts)
    assert s["bounded"] is True
    assert s["per_device"]["kinect_0"]["max_abs_offset_us"] == 120
    assert abs(s["per_device"]["kinect_0"]["drift_ms_per_min"]) < 1e-6
    assert s["per_device"]["kinect_1"]["std_offset_us"] == 0.0


def test_linear_ramp_reports_drift_and_trips_bound():
    # +4 us every frame at 15 fps => 3.6 ms per minute, above the 1 ms bound.
    ts = _ticks(900, offsets={"kinect_1": lambda i: 4 * i})
    s = _sync_stats(ts)
    drift = s["per_device"]["kinect_1"]["drift_ms_per_min"]
    assert abs(drift - 3.6) < 0.05
    assert s["bounded"] is False
    assert s["worst_drift_ms_per_min"] > _DRIFT_BOUND_MS_PER_MIN


def test_no_offsets_is_vacuously_bounded():
    s = _sync_stats([{"sync_us": 0, "offsets_us": {}}])
    assert s["per_device"] == {}
    assert s["bounded"] is True
