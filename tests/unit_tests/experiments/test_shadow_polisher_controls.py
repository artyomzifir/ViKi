"""Feasibility and phase invariants for the one-scene Shadow pilot."""
import numpy as np
import pytest

from experiments.shadow_polisher_controls import command_margins, envelope, polished_commands


NAMES = ['rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
         'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
         'rh_THJ4', 'rh_THJ2']


def test_zero_is_bitwise_identity_and_does_not_mutate_source():
    original = np.random.default_rng(1).random((190, len(NAMES))).astype(np.float32)
    snapshot = original.copy()
    result = polished_commands(original, NAMES, [0, 0, 0, 0])
    assert result.dtype == np.float32
    assert np.array_equal(result, snapshot)
    assert np.array_equal(original, snapshot)
    assert result is not original


def test_residual_only_changes_frozen_carry_interior_and_release_is_preserved():
    original = np.zeros((190, len(NAMES)), dtype=np.float32)
    result = polished_commands(original, NAMES, [.15, -.03, .08, -.1])
    gate = envelope(190)
    assert np.array_equal(result[:44], original[:44])
    assert np.array_equal(result[125:], original[125:])
    assert np.array_equal(result[189], original[189])
    assert np.max(np.abs(np.diff(gate))) < .1
    assert result[80, NAMES.index('rh_LFJ2')] == pytest.approx(.12, abs=1e-7)
    assert result[80, NAMES.index('rh_THJ2')] == pytest.approx(-.1, abs=1e-7)


def test_rejects_bad_parameters_and_incompatible_plan():
    original = np.zeros((190, len(NAMES)), dtype=np.float32)
    with pytest.raises(ValueError):
        polished_commands(original, NAMES, [.23, 0, 0, 0])
    with pytest.raises(ValueError):
        polished_commands(original[:100], NAMES, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        polished_commands(original, NAMES[:-1] + [NAMES[0]], [0, 0, 0, 0])
    original[0, 0] = np.nan
    with pytest.raises(ValueError):
        polished_commands(original, NAMES, [0, 0, 0, 0])


def test_position_velocity_and_timing_violations_are_reported():
    commands = np.array([[0.], [.1], [.2]])
    timestamps = np.array([0, 100_000, 200_000])
    assert command_margins(commands, timestamps, [-.1], [.3], [2.]) == pytest.approx((.1, 1.))
    assert command_margins(commands, timestamps, [-.1], [.15], [.5]) == pytest.approx((-.05, -.5))
    with pytest.raises(ValueError):
        command_margins(commands, [0, 100_000, np.nan], [-.1], [.3], [2.])
    with pytest.raises(ValueError):
        command_margins(commands, [0, 100_000, 100_000], [-.1], [.3], [2.])


def test_terminal_gate_rejects_initial_only_support_and_last_frame_only_settling():
    from experiments.shadow_polisher_controls import terminal_placement_gate
    contact = np.zeros((190, 2), dtype=bool)
    support = np.ones(190)
    speed = np.zeros(190)
    assert terminal_placement_gate(contact, support, speed) == (True, True, True)
    support[-15:] = 0
    assert terminal_placement_gate(contact, support, speed) == (True, False, True)
    support[-15:] = 1
    support[-7] = .99  # one physics step without positive table force
    assert terminal_placement_gate(contact, support, speed) == (True, False, True)
    support[-7] = 1
    speed[-8] = .04
    speed[-1] = 0
    assert terminal_placement_gate(contact, support, speed) == (True, True, False)
    speed[-8] = 0
    contact[-3, 0] = True
    assert terminal_placement_gate(contact, support, speed) == (False, True, True)


def test_carry_retiming_keeps_pose_count_and_valid_segment_order():
    from experiments.shadow_polisher_controls import retime_carry
    ts = np.arange(190, dtype=np.int64) * 33_333
    slowed = retime_carry(ts, 1.5)
    assert len(slowed) == len(ts)
    assert np.array_equal(slowed[:42], ts[:42])
    assert np.all(np.diff(slowed)[41:125] >= np.diff(ts)[41:125])
    assert np.array_equal(np.diff(slowed)[125:], np.diff(ts)[125:])
    assert np.array_equal(retime_carry(ts, 1.), ts)
    with pytest.raises(ValueError):
        retime_carry(ts, 1.7)


def test_export_validation_rejects_tampered_source_hash_and_timestamps():
    from experiments.shadow_polisher_controls import retime_carry, validate_command_export
    ts = np.arange(190, dtype=np.int64) * 33_333
    payload = {'schema':np.asarray('shadow_polisher_commands_v1'),
               'plan_sha256':np.asarray('real-plan'), 'track_sha256':np.asarray('real-track'),
               'joint_names':np.asarray(NAMES), 'carry_timing_factor':np.asarray(1.5),
               'timestamps':retime_carry(ts, 1.5),
               'commands':np.zeros((190,len(NAMES)), dtype=np.float32)}
    expected = {k:str(payload[k]) for k in ('schema','plan_sha256','track_sha256')}
    commands, times, factor = validate_command_export(payload, expected, NAMES, ts,
        (190,len(NAMES)), np.dtype('float32'))
    assert commands.shape == (190,len(NAMES))
    assert np.array_equal(times, payload['timestamps']) and factor == 1.5
    tampered = {**payload, 'track_sha256':np.asarray('other-track')}
    with pytest.raises(ValueError, match='track_sha256'):
        validate_command_export(tampered, expected, NAMES, ts, commands.shape, commands.dtype)
    tampered = {**payload, 'timestamps':ts.copy()}
    with pytest.raises(ValueError, match='timestamps'):
        validate_command_export(tampered, expected, NAMES, ts, commands.shape, commands.dtype)
