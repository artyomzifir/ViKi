"""viki.retarget.frames.world_to_robot + robots alias resolution."""

import numpy as np
import pytest

from viki.config import Config
from viki.retarget.frames import world_to_robot
from viki.retarget.robots import ROBOT_CONFIGS, normalize_robot


def _cfg(**kw) -> Config:
    from types import MappingProxyType

    return Config(MappingProxyType(dict(kw)))


def test_world_to_robot_identity_when_unset():
    np.testing.assert_allclose(world_to_robot(_cfg()), np.eye(4))


def test_world_to_robot_from_config():
    R = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    t = [0.1, 0.2, 0.3]
    T = world_to_robot(
        _cfg(RETARGET_BASE_ROTATION=R, RETARGET_BASE_TRANSLATION=t)
    )
    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], t)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_robot_aliases_resolve():
    assert normalize_robot("ur10_description") is ROBOT_CONFIGS["ur10"]
    assert normalize_robot("  ur3  ") is ROBOT_CONFIGS["ur3"]


def test_unknown_robot_raises():
    with pytest.raises(ValueError):
        normalize_robot("panda")
