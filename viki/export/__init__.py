"""
viki.export
-----------
Pipeline stage 6: labelled + screened episodes -> a LeRobot dataset (paper §3.9).

Delegates to the optional ``lerobot`` package (``pip install viki[export]``).
"""

from viki.export.run import export_dataset  # noqa: F401

__all__ = ["export_dataset"]
