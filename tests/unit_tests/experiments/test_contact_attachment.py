import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from experiments.contact_attachment import ContactAttachment


@pytest.fixture
def attachment():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/>'
        '<worldbody><body name="wrist"><joint type="slide" axis="1 0 0"/>'
        '<geom size="0.01" mass="1"/></body>'
        '<body name="object" pos="0 0 0.2"><freejoint/>'
        '<geom size="0.01" mass="0.1"/></body></worldbody>'
        '<equality><weld name="grasp" body1="wrist" body2="object" '
        'active="false"/></equality></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ContactAttachment(model, data, "grasp", "wrist", "object")


def observe(attachment, left=0.5, right=0.5, non_pad=False,
            command=0.5, position=0.5, pads=None, dt=0.04):
    if pads is None:
        pads = [np.array([-0.025, 0, 0.2]), np.array([0.025, 0, 0.2])]
    attachment.observe(
        {"left": left, "right": right}, pads, non_pad, command, position, dt,
    )


def test_requires_sustained_bilateral_contact_between_pads(attachment):
    observe(attachment, right=0, dt=0.2)
    assert not attachment.attached
    observe(attachment, non_pad=True, dt=0.2)
    assert not attachment.attached
    observe(attachment, pads=[np.array([1, 0, 0.2]), np.array([2, 0, 0.2])], dt=0.2)
    assert not attachment.attached
    observe(attachment, command=0.1, dt=0.2)
    assert not attachment.attached
    observe(attachment)
    assert not attachment.attached
    observe(attachment)
    assert attachment.attached
    assert bool(attachment.data.eq_active[attachment.weld_id])
    assert attachment.attach_forces_n == {"left": 0.5, "right": 0.5}
    assert attachment.attach_pose_jump_m == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(attachment.model.eq_data[attachment.weld_id, 3:6], [0, 0, 0.2])


def test_opens_to_release_and_does_not_spontaneously_reattach(attachment):
    observe(attachment, dt=0.08)
    assert attachment.attached
    observe(attachment, command=0.3, position=0.5)
    assert attachment.attached
    observe(attachment, command=0.3, position=0.3)
    assert not attachment.attached
    assert attachment.released
    assert not bool(attachment.data.eq_active[attachment.weld_id])
    observe(attachment, dt=0.2)
    assert not attachment.attached
    attachment.reset()
    observe(attachment, dt=0.08)
    assert attachment.attached


def test_any_pad_mode_attaches_immediately_and_releases_on_phase(attachment):
    assisted = ContactAttachment(
        attachment.model, attachment.data, "grasp", "wrist", "object",
        hold_s=0.0005, attach_on_any_pad=True,
    )
    observe(
        assisted, left=0.001, right=0.0, non_pad=True,
        command=0.0, position=0.0,
        pads=[np.array([1, 0, 0.2]), np.array([2, 0, 0.2])], dt=0.0005,
    )
    assert assisted.attached
    assisted.release()
    assert assisted.released and not assisted.attached
    observe(assisted, dt=0.1)
    assert not assisted.attached
