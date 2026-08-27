"""Compatibility shim — moved to :mod:`viki.retarget.archive`."""

from viki.retarget.archive import Hdf5Archive, load_archive, write_hdf5_archive  # noqa: F401

__all__ = ["Hdf5Archive", "load_archive", "write_hdf5_archive"]
