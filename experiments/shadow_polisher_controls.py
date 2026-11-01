"""Pure command construction for the opt-in Shadow physics pilot."""

import numpy as np


PARAMETER_BOUNDS = np.array([
    [-0.08, 0.22],  # four-finger flexion, radians
    [-0.10, 0.18],  # little-finger additional flexion
    [-0.08, 0.15],  # thumb flexion
    [-0.15, 0.15],  # thumb opposition (THJ2)
], dtype=float)


def envelope(n_frames, rise=(43, 58), fall=(110, 125)):
    """C1 smoothstep envelope wholly inside the constant saved grasp phase."""
    if n_frames != 190 or rise != (43, 58) or fall != (110, 125):
        raise ValueError('this pilot is frozen to the 190-frame G10_2 carry phase')
    x = np.arange(n_frames, dtype=float)
    def smooth(a, b):
        u = np.clip((x - a) / (b - a), 0, 1)
        return u * u * (3 - 2 * u)
    return smooth(*rise) * (1 - smooth(*fall))


def polished_commands(original, names, parameters):
    """Return new commands without modifying the source, including both release tails."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError('expected four finite parameters')
    if np.any(parameters < PARAMETER_BOUNDS[:, 0]) or np.any(parameters > PARAMETER_BOUNDS[:, 1]):
        raise ValueError('parameter outside predeclared bounds')
    result = np.asarray(original).copy()
    if result.ndim != 2 or result.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('command shape or unique names disagree')
    if not np.isfinite(result).all():
        raise ValueError('source commands are nonfinite')
    gate = envelope(len(result))
    joint_index = {name: i for i, name in enumerate(names)}
    for digit in ('FF', 'MF', 'RF', 'LF'):
        for joint, weight in ((3, .5), (2, 1.), (1, 1.)):
            result[:, joint_index[f'rh_{digit}J{joint}']] += gate * parameters[0] * weight
    for joint, weight in ((3, .5), (2, 1.), (1, 1.)):
        result[:, joint_index[f'rh_LFJ{joint}']] += gate * parameters[1] * weight
    result[:, joint_index['rh_THJ4']] += gate * parameters[2]
    result[:, joint_index['rh_THJ2']] += gate * parameters[3]
    return result


def command_margins(commands, timestamps, lower, upper, velocity):
    """Return signed minimum position and segment-velocity margins in SI units."""
    commands = np.asarray(commands, dtype=float)
    dt = np.diff(np.asarray(timestamps, dtype=float)) / 1e6
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if (commands.ndim != 2 or len(dt) != len(commands) - 1 or
        any(x.shape != (commands.shape[1],) for x in (lower, upper, velocity)) or
        not all(np.isfinite(x).all() for x in (commands, dt, lower, upper, velocity)) or
        np.any(dt <= 0) or np.any(lower >= upper) or np.any(velocity <= 0)):
        raise ValueError('invalid commands, timestamps, or limits')
    position = min(float(np.min(commands - lower)), float(np.min(upper - commands)))
    speed = np.abs(np.diff(commands, axis=0) / dt[:, None])
    velocity_margin = float(np.min(velocity - speed))
    return position, velocity_margin


def retime_carry(timestamps, factor):
    """Slow only intervals ending at frames 42..125, retaining every pose."""
    ts = np.asarray(timestamps)
    if ts.shape != (190,) or ts.dtype.kind not in 'iu' or factor not in (1., 1.5, 2.):
        raise ValueError('unsupported frozen-plan timing factor')
    delta = np.diff(ts.astype(np.int64))
    if np.any(delta <= 0):
        raise ValueError('timestamps must increase')
    delta = delta.copy()
    delta[41:125] = np.rint(delta[41:125] * factor).astype(np.int64)
    return np.r_[ts[0], ts[0] + np.cumsum(delta)]


def terminal_placement_gate(hand_contacts, support_fractions, max_step_speeds, frames=15):
    """Require continuous positive table support, release, and low speed at end."""
    hand_contacts = np.asarray(hand_contacts, dtype=bool)
    support_fractions = np.asarray(support_fractions, dtype=float)
    max_step_speeds = np.asarray(max_step_speeds, dtype=float)
    if (hand_contacts.ndim != 2 or hand_contacts.shape[1] != 2 or
        support_fractions.shape != (len(hand_contacts),) or
        max_step_speeds.shape != (len(hand_contacts),) or
        len(hand_contacts) < frames or not np.isfinite(support_fractions).all() or
        not np.isfinite(max_step_speeds).all()):
        raise ValueError('invalid terminal evidence')
    return (bool(not hand_contacts[-frames:].any()),
            bool(np.all(support_fractions[-frames:] >= 1. - 1e-12)),
            bool(np.all(max_step_speeds[-frames:] < .03)))


def validate_command_export(saved, expected_hashes, joint_names, source_timestamps, command_shape, command_dtype):
    """Fail closed on source provenance, timing, names, shape, and precision."""
    for key, value in expected_hashes.items():
        if key not in saved or str(saved[key]) != value:
            raise ValueError(f'exported command provenance mismatch: {key}')
    if list(map(str, saved['joint_names'])) != list(joint_names):
        raise ValueError('exported command names mismatch')
    factor = float(saved['carry_timing_factor'])
    timestamps = retime_carry(source_timestamps, factor)
    if not np.array_equal(saved['timestamps'], timestamps):
        raise ValueError('exported timestamps do not match declared retiming')
    commands = np.asarray(saved['commands']).copy()
    if commands.shape != command_shape or commands.dtype != command_dtype or not np.isfinite(commands).all():
        raise ValueError('exported command shape, precision, or finiteness mismatch')
    return commands, timestamps, factor
