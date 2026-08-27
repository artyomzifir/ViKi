"""Compatibility shim — fusion moved to :mod:`viki.prepare.fuse`."""

from viki.prepare.fuse import fuse_trajectories  # noqa: F401

__all__ = ["fuse_trajectories"]
