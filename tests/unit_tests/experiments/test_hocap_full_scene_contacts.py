from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")

from experiments.hocap_grasp_env import HocapGraspEnv
from experiments.hocap_object_guided_replay import _step


def test_full_scene_contact_filters_table_support_and_other_objects(monkeypatch):
    environment = object.__new__(HocapGraspEnv)
    environment.table_geom = 1
    environment.object_geom = 2
    environment.other_object_geoms = {3: 7, 4: 8}
    environment.model = SimpleNamespace(
        geom_bodyid=[0, 0, 0, 0, 0, 1],
        body=lambda body_id: SimpleNamespace(name="left_distal_phalanx"),
    )
    environment.data = SimpleNamespace(
        ncon=5,
        contact=[
            SimpleNamespace(geom1=1, geom2=2),
            SimpleNamespace(geom1=1, geom2=3),
            SimpleNamespace(geom1=3, geom2=4),
            SimpleNamespace(geom1=3, geom2=5),
            SimpleNamespace(geom1=1, geom2=5),
        ],
    )
    forces = [10.0, 10.0, 10.0, 0.19, 0.2]

    def contact_force(model, data, contact_index, output):
        output[:] = 0.0
        output[0] = forces[contact_index]

    monkeypatch.setattr("experiments.hocap_grasp_env.mujoco.mj_contactForce", contact_force)
    assert environment._other_object_contacts() == []
    assert environment._robot_table_contacts() == ["left_distal_phalanx"]
    forces[3] = 0.2
    assert environment._other_object_contacts() == [7]


def test_guided_step_does_not_abort_on_success():
    info = {
        "success": True, "lift_m": 0.04, "bilateral_contact": True,
        "non_pad_contact": False, "other_object_contact_ids": [],
        "robot_table_contact_bodies": [], "other_object_shift_m": {},
        "contact_attachment_active": False,
        "pad_forces_n": {"left": 0.3, "right": 0.3},
    }
    environment = SimpleNamespace(step=lambda action: (None, 0.0, True, False, info))
    metrics = {
        "steps": 0, "peak_lift_m": 0.0, "bilateral_steps": 0,
        "non_pad_contact_steps": 0, "attached_steps": 0,
        "other_object_contact_ids": set(), "robot_table_contact_bodies": set(),
        "max_other_object_shift_m": 0.0,
        "peak_pad_force_n": {"left": 0.0, "right": 0.0},
    }
    assert _step(environment, None, metrics)
    assert metrics["steps"] == 1
    info["success"] = False
    assert not _step(environment, None, metrics)
