"""Tests for atomic HDF5 artifact helpers."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from viki.retarget.archive import load_archive, write_hdf5_archive


def test_hdf5_archive_round_trip_and_schema(tmp_path: Path):
    path = tmp_path / "plan.h5"
    q = np.arange(12, dtype=np.float64).reshape(3, 4)
    write_hdf5_archive(
        path,
        {"q": q, "robot": "ur10_official_description", "ready": True},
        schema="viki_plan_hdf5_v2",
    )
    assert not list(tmp_path.glob("*.tmp"))
    with h5py.File(path) as raw:
        assert raw.attrs["archive_format"] == "viki_plan_hdf5_v2"
    with load_archive(path) as archive:
        np.testing.assert_allclose(archive["q"], q)
        assert archive["robot"] == "ur10_official_description"
        assert bool(archive["ready"])


def test_legacy_npz_trajectory_is_rejected(tmp_path: Path):
    path = tmp_path / "plan.npz"
    np.savez(path, q=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="must be HDF5"):
        load_archive(path)
