"""Small HDF5 archive helpers for optimisation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


class Hdf5Archive:
    """Minimal h5py-backed archive with the subset of npz API used here."""

    def __init__(self, path: Path):
        self._file = h5py.File(path, "r")
        self.files = list(self._file.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._file

    def __getitem__(self, key: str) -> Any:
        value = self._file[key][()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "Hdf5Archive":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def load_archive(path: Path):
    """Open an HDF5 trajectory archive, or a legacy npz archive."""
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return Hdf5Archive(path)
    return np.load(path, allow_pickle=True)


def write_hdf5_archive(path: Path, values: dict[str, Any]) -> None:
    """Write arrays and scalar metadata to a flat HDF5 archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["archive_format"] = "viki_optimization_hdf5_v1"
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, Path):
                value = str(value)
            if isinstance(value, str):
                h5.create_dataset(key, data=np.array(value, dtype=STRING_DTYPE))
            elif isinstance(value, bool):
                h5.create_dataset(key, data=np.bool_(value))
            elif isinstance(value, (int, float, np.generic)):
                h5.create_dataset(key, data=value)
            else:
                h5.create_dataset(key, data=np.asarray(value))
