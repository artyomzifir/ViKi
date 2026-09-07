"""Small, atomic HDF5 archive helpers for trajectory artifacts."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
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
    """Open an HDF5 trajectory archive."""
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"trajectory artifacts must be HDF5, got {path}")
    return Hdf5Archive(path)


def write_hdf5_archive(
    path: Path,
    values: dict[str, Any],
    *,
    schema: str = "viki_trajectory_hdf5_v1",
) -> None:
    """Atomically write arrays and scalar metadata to a flat HDF5 archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["archive_format"] = schema
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
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
