from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")

from experiments.hocap_grasp_env import HocapGraspEnv


def test_non_pad_object_contact_is_distinguished_from_table_and_pads(monkeypatch):
    environment = object.__new__(HocapGraspEnv)
    environment.model = object()
    environment.object_geom = 1
    environment.table_geom = 2
    environment.pad_geoms = {3: "left", 4: "right"}
    environment.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2),
            SimpleNamespace(geom1=3, geom2=1),
            SimpleNamespace(geom1=1, geom2=4),
            SimpleNamespace(geom1=1, geom2=5),
        ],
    )
    contact_forces = [0.0, 0.4, 0.5, 0.19]

    def contact_force(model, data, contact_index, output):
        output[:] = 0.0
        output[0] = contact_forces[contact_index]

    monkeypatch.setattr("experiments.hocap_grasp_env.mujoco.mj_contactForce", contact_force)
    pad_forces, table_support, non_pad_contact = environment._contacts()
    assert pad_forces == {"left": pytest.approx(0.4), "right": pytest.approx(0.5)}
    assert table_support
    assert not non_pad_contact

    contact_forces[-1] = 0.2
    assert environment._contacts()[2]
