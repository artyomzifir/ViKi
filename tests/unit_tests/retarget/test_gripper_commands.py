import numpy as np

from viki.contracts import GripperState
from viki.gripper import BinaryGripper


def test_binary_gripper_owns_one_command_dimension():
    model = BinaryGripper()
    states = [
        GripperState(False, 1.0, 1.0),
        GripperState(True, 0.0, 0.8),
    ]
    assert model.command_names == ("closed",)
    np.testing.assert_array_equal(model.encode(states), [[0.0], [1.0]])
