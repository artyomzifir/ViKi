"""Low-dimensional simulator state for the first imitation-learning baseline."""

from __future__ import annotations

import numpy as np

OBSERVATION_SCHEMA = "viki_cube_sim_state_v1"
ACTION_SCHEMA = "ur10_finger_position_target_v1"


def observation(model, data) -> np.ndarray:
    from experiments.scene import ARM_JOINTS

    values = []
    for name in ARM_JOINTS:
        joint = model.joint(name).id
        values.append(float(data.qpos[model.jnt_qposadr[joint]]))
    for name in ARM_JOINTS:
        joint = model.joint(name).id
        values.append(float(data.qvel[model.jnt_dofadr[joint]]))
    finger = model.joint("finger_joint").id
    values.extend((float(data.qpos[model.jnt_qposadr[finger]]),
                   float(data.qvel[model.jnt_dofadr[finger]])))
    for name in ("red", "green", "blue"):
        body = model.body(f"{name}_cube").id
        joint = model.joint(f"{name}_free").id
        values.extend(data.xpos[body])
        values.extend(data.xquat[body])
        start = int(model.jnt_dofadr[joint])
        values.extend(data.qvel[start:start + 6])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (53,) or not np.isfinite(result).all():
        raise ValueError("invalid 53-dimensional cube simulator observation")
    return result
