# URDF / MJCF Robot Models

## For IK / retargeting (Pinocchio + PINK)

Loaded automatically via `robot_descriptions` — no manual files needed.

| Robot | Description name | DOF (arm) | EE frame |
|---|---|---|---|
| UR10 | `ur10_description` | 6 | `tool0` |
| KUKA iiwa 14 | `iiwa14_description` | 7 | `iiwa_link_ee` |
| Franka Panda | `panda_description` | 7+2 fingers | `panda_hand_tcp` |

```python
from robot_descriptions.loaders.pinocchio import load_robot_description
robot = load_robot_description("ur10_description")
```

## For MuJoCo visualisation (full meshes)

Sparse clone from MuJoCo Menagerie (Google DeepMind):

```bash
git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie
git sparse-checkout set universal_robots_ur5e kuka_iiwa_14 franka_emika_panda
```

Or via robot_descriptions:
```python
from robot_descriptions.loaders.mujoco import load_robot_description
robot = load_robot_description("ur5e_mj_description")
```

| Menagerie name | robot_descriptions name |
|---|---|
| `universal_robots_ur5e` | `ur5e_mj_description` |
| `kuka_iiwa_14` | `iiwa14_mj_description` |
| `franka_emika_panda` | `panda_mj_description` |

## Notes

- **Meshes not needed for IK** — Pinocchio uses only kinematic structure.
- **Panda home config**: `[0, -π/4, 0, -3π/4, 0, π/2, π/4, 0.04, 0.04]`
  (joint 3 has limits [-3.07, -0.07], so `pin.neutral()` is invalid)
