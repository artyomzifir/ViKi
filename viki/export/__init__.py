"""
viki.export
-----------
Pipeline stage 6: retargeted episodes -> a dataset.

Two writers, because they answer different needs:

``export_trajectories``
    Self-contained trajectory bundle — joint trajectory, tool pose, gripper
    command, the human pose it came from, and a manifest. numpy only. This is
    the default, and the one that works without optional dependencies.

``export_dataset``
    LeRobot dataset for policy training, including camera video. Delegates to
    the optional ``lerobot`` package (``pip install 'viki[export]'``, which
    pulls in torch) and requires episodes that have been replayed and screened.
"""

from viki.export.run import export_dataset  # noqa: F401
from viki.export.trajectory import export_trajectories  # noqa: F401

__all__ = ["export_dataset", "export_trajectories"]
