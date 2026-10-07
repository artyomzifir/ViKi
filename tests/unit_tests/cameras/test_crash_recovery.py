"""
Surviving a host lock-up mid-episode — ``viki.cameras.record``.

The canonical ``timestamps.json`` is written only when a recording finishes, so
a freeze used to cost the whole take. The sidecar is appended as frames arrive.
"""

from __future__ import annotations

import json

import pytest

from viki.cameras.record import recover_timestamps


def _row(i):
    return {"sync_us": 1000 * i, "offsets_us": {"kinect_0": 160},
            "device_timestamps_us": {"kinect_0": 1000 * i}}


def test_recovers_every_flushed_row(tmp_path):
    (tmp_path / "timestamps.jsonl").write_text(
        "".join(json.dumps(_row(i)) + "\n" for i in range(5))
    )

    n = recover_timestamps(tmp_path)

    assert n == 5
    rebuilt = json.loads((tmp_path / "timestamps.json").read_text())
    assert [r["sync_us"] for r in rebuilt] == [0, 1000, 2000, 3000, 4000]


def test_a_truncated_final_row_is_dropped_not_fatal(tmp_path):
    """A lock-up can cut the last line mid-write; the rest must still survive."""
    good = "".join(json.dumps(_row(i)) + "\n" for i in range(3))
    (tmp_path / "timestamps.jsonl").write_text(good + '{"sync_us": 3000, "offs')

    n = recover_timestamps(tmp_path)

    assert n == 3
    assert len(json.loads((tmp_path / "timestamps.json").read_text())) == 3


def test_blank_lines_are_ignored(tmp_path):
    (tmp_path / "timestamps.jsonl").write_text(
        json.dumps(_row(0)) + "\n\n" + json.dumps(_row(1)) + "\n"
    )
    assert recover_timestamps(tmp_path) == 2


def test_no_sidecar_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        recover_timestamps(tmp_path)
