"""
viki.perception.hand_model
--------------------------
A licence-free parametric articulated **capsule** hand, rigged to the
:class:`viki.contracts.LM` 21-point topology, for fitting to the per-frame hand
point cloud (:mod:`viki.perception.hand_fit`).

Why a capsule model, why these radii
------------------------------------
The hand is a kinematic tree: a free-flyer ``wrist`` (the 6-DOF we ultimately
want) plus 5 finger chains. Each phalanx is a **capsule** and the palm is one
broad longitudinal capsule
(line segment + radius) — the cheapest primitive whose signed point distance has
a closed form and a smooth Jacobian, and a good geometric proxy for a finger.

Segment lengths are **calibrated per user** from the recording itself (the
open-hand frames), never hard-coded. Capsule radii are taken as
``RADIUS_FRAC · segment_length`` clamped to a plausible human band
(``RADIUS_FRAC ≈ 0.35`` ≈ finger cross-section radius / phalanx length). Fitting
the radius from the cloud cross-section thickness would be more principled and is
a **TODO**; the fixed ratio is stable and good enough for a wrist-pose refinement.

FK is Pinocchio (same library the robot retarget stage uses) so we get the model
loader pattern and analytic frame Jacobians for free. The model is generated as a
URDF string from the calibrated params and loaded with ``pin.buildModelFromXML``.

Joint layout (``nv ≈ 26``): free-flyer wrist (6) + per finger
``abduction`` + ``mcp`` flex + ``pip`` flex + ``dip`` flex (4 each; thumb uses
``cmc``/``mcp``/``ip`` names but the same 2+1+1 split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from viki.contracts import LM
from viki.perception.hand_angles import compute_palm_rotation, _rot_to_rpy_extrinsic_xyz

# finger name -> its 4 LM points, MCP/CMC first, TIP last (3 bones each)
FINGERS: dict[str, tuple[LM, LM, LM, LM]] = {
    "thumb": (LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP),
    "index": (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
    "middle": (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    "ring": (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
    "pinky": (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
}
# per finger the 4 revolute joint names root→tip (mcp/cmc is 2-DOF: abd + flex)
_JOINTS: dict[str, tuple[str, str, str, str]] = {
    "thumb": ("thumb_abd", "thumb_cmc", "thumb_mcp", "thumb_ip"),
    **{f: (f"{f}_abd", f"{f}_mcp", f"{f}_pip", f"{f}_dip")
       for f in ("index", "middle", "ring", "pinky")},
}

RADIUS_FRAC = 0.35
_R_MIN, _R_MAX = 0.006, 0.014          # m — human phalanx radius band
_PALM_R = 0.024                        # m — broad palm proxy radius
_FLEX_LIM = (-0.25, 1.95)              # rad
_ABD_LIM = (-0.45, 0.45)
_THUMB_ABD_LIM = (-0.7, 1.0)
_THUMB_FLEX_LIM = (-0.4, 1.4)


@dataclass
class HandParams:
    """Calibrated rest geometry in the right-hand palm frame.

    ``x`` points wrist→fingers and ``+z`` points out through the palmar (grasp)
    side. Positive flexion must therefore curl a finger from ``+x`` toward
    ``+z``, which is a rotation about local ``-y``.
    """
    root_xyz: dict[str, np.ndarray]      # finger -> (3,) MCP/CMC position
    root_rpy: dict[str, np.ndarray]      # finger -> (3,) rot so local +x = finger axis
    bone_len: dict[str, np.ndarray]      # finger -> (3,) proximal/middle/distal lengths (m)
    radius: dict[str, np.ndarray]        # finger -> (3,) capsule radii (m)
    palm_r: float = _PALM_R

    def as_dict(self) -> dict:
        return {
            "root_xyz": {k: v.tolist() for k, v in self.root_xyz.items()},
            "root_rpy": {k: v.tolist() for k, v in self.root_rpy.items()},
            "bone_len": {k: v.tolist() for k, v in self.bone_len.items()},
            "radius": {k: v.tolist() for k, v in self.radius.items()},
            "palm_r": self.palm_r,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HandParams":
        g = lambda m: {k: np.asarray(v, float) for k, v in m.items()}
        return cls(g(d["root_xyz"]), g(d["root_rpy"]), g(d["bone_len"]),
                   g(d["radius"]), float(d.get("palm_r", _PALM_R)))


# ── calibration ────────────────────────────────────────────────────────────


def _palm_frame(pts: Mapping[LM, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """(wrist_xyz, R_world_palm) or None."""
    w = pts.get(LM.WRIST)
    R = compute_palm_rotation(
        w, pts.get(LM.INDEX_MCP), pts.get(LM.MIDDLE_MCP), pts.get(LM.PINKY_MCP)
    )
    if w is None or R is None or not np.all(np.isfinite(w)):
        return None
    return np.asarray(w, np.float64), np.asarray(R, np.float64)


def _to_palm(pts: Mapping[LM, np.ndarray], w: np.ndarray, R: np.ndarray) -> dict[LM, np.ndarray]:
    out: dict[LM, np.ndarray] = {}
    for lm, p in pts.items():
        p = np.asarray(p, np.float64)
        if p.shape == (3,) and np.all(np.isfinite(p)):
            out[lm] = R.T @ (p - w)
    return out


def _rpy_for_dir(direction: np.ndarray) -> np.ndarray:
    """rpy (extrinsic xyz) of a rotation whose local +x is ``direction`` and whose
    local +z stays close to the palm normal (+z)."""
    x = direction / (np.linalg.norm(direction) + 1e-12)
    z_ref = np.array([0.0, 0.0, 1.0])
    y = np.cross(z_ref, x)
    ny = np.linalg.norm(y)
    if ny < 1e-6:                       # finger points along palm z — rare
        y = np.array([0.0, 1.0, 0.0])
    else:
        y = y / ny
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return _rot_to_rpy_extrinsic_xyz(R)


def calibrate_from_frames(frames: list[Mapping[LM, np.ndarray]]) -> HandParams:
    """Median rest geometry over ``frames`` (should be open-hand, fingers spread).

    Each frame is a ``{LM: world xyz}`` mapping (missing / NaN allowed). Needs at
    least one frame with a resolvable palm and, per finger, at least one frame
    with all four landmarks finite.
    """
    palm_local: list[dict[LM, np.ndarray]] = []
    for fr in frames:
        pf = _palm_frame(fr)
        if pf is not None:
            palm_local.append(_to_palm(fr, *pf))
    if not palm_local:
        raise ValueError("calibrate_from_frames: no frame with a resolvable palm")

    root_xyz, root_rpy, bone_len, radius = {}, {}, {}, {}
    for f, lms in FINGERS.items():
        roots, lens, dirs = [], [], []
        for pl in palm_local:
            pp = [pl.get(lm) for lm in lms]
            if any(p is None for p in pp):
                continue
            pp = [np.asarray(p, float) for p in pp]
            b = [pp[1] - pp[0], pp[2] - pp[1], pp[3] - pp[2]]
            ln = np.array([np.linalg.norm(v) for v in b])
            if np.any(ln < 1e-4):
                continue
            roots.append(pp[0])
            lens.append(ln)
            dirs.append((pp[3] - pp[0]) / (np.linalg.norm(pp[3] - pp[0]) + 1e-12))
        if not roots:
            # finger never fully seen — fall back to a generic straight finger
            roots = [np.array([0.06 if f != "thumb" else 0.02,
                               {"index": 0.03, "middle": 0.0, "ring": -0.02,
                                "pinky": -0.04, "thumb": 0.04}[f], 0.0])]
            lens = [np.array([0.040, 0.025, 0.020]) if f != "thumb"
                    else np.array([0.038, 0.032, 0.025])]
            dirs = [np.array([1.0, 0.0, 0.0])]
        root_xyz[f] = np.median(np.stack(roots), axis=0)
        bl = np.median(np.stack(lens), axis=0)
        bone_len[f] = bl
        root_rpy[f] = _rpy_for_dir(np.median(np.stack(dirs), axis=0))
        radius[f] = np.clip(RADIUS_FRAC * bl, _R_MIN, _R_MAX)

    # One broad wrist→middle-MCP capsule is a conservative palm proxy. The
    # former five overlapping metacarpal capsules counted dense palm pixels
    # five times and caused unstable switches between coincident primitives.
    palm_width = float(np.linalg.norm(root_xyz["index"] - root_xyz["pinky"]))
    # Keep the proxy narrower than half the palm span: an overly spherical
    # capsule steals correspondences from proximal phalanges and makes wrist
    # roll weakly observable. The adaptive ROI adds its own 3 cm margin.
    palm_r = float(np.clip(0.32 * palm_width, 0.020, 0.030))
    return HandParams(root_xyz, root_rpy, bone_len, radius, palm_r=palm_r)


# ── URDF generation + Pinocchio model ─────────────────────────────────────


def _limits(joint: str) -> tuple[float, float]:
    if joint.endswith("_abd"):
        return _THUMB_ABD_LIM if joint.startswith("thumb") else _ABD_LIM
    return _THUMB_FLEX_LIM if joint.startswith("thumb") else _FLEX_LIM


def _urdf(params: HandParams) -> str:
    def link(name: str) -> str:
        # a tiny inertial keeps Pinocchio happy; geometry is handled by capsules
        return (f'  <link name="{name}"><inertial><mass value="0.02"/>'
                f'<inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>'
                f'</inertial></link>\n')

    x = ['<robot name="viki_hand">\n', link("palm"), link("wrist_link")]
    x.append('  <joint name="wrist" type="floating">'
             '<parent link="palm"/><child link="wrist_link"/></joint>\n')

    for f, jn in _JOINTS.items():
        rx = params.root_xyz[f]
        rr = params.root_rpy[f]
        L = params.bone_len[f]
        abd, j_mcp, j_pip, j_dip = jn
        for ln in (f"{f}_mcp_link", f"{f}_prox", f"{f}_mid", f"{f}_dist", f"{f}_tip"):
            x.append(link(ln))
        lo_a, hi_a = _limits(abd)
        lo_f, hi_f = _limits(j_mcp)
        x.append(
            f'  <joint name="{abd}" type="revolute">'
            f'<parent link="wrist_link"/><child link="{f}_mcp_link"/>'
            f'<origin xyz="{rx[0]:.6f} {rx[1]:.6f} {rx[2]:.6f}" '
            f'rpy="{rr[0]:.6f} {rr[1]:.6f} {rr[2]:.6f}"/>'
            f'<axis xyz="0 0 1"/><limit lower="{lo_a}" upper="{hi_a}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{j_mcp}" type="revolute">'
            f'<parent link="{f}_mcp_link"/><child link="{f}_prox"/>'
            f'<origin xyz="0 0 0"/><axis xyz="0 -1 0"/>'
            f'<limit lower="{lo_f}" upper="{hi_f}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{j_pip}" type="revolute">'
            f'<parent link="{f}_prox"/><child link="{f}_mid"/>'
            f'<origin xyz="{L[0]:.6f} 0 0"/><axis xyz="0 -1 0"/>'
            f'<limit lower="{lo_f}" upper="{hi_f}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{j_dip}" type="revolute">'
            f'<parent link="{f}_mid"/><child link="{f}_dist"/>'
            f'<origin xyz="{L[1]:.6f} 0 0"/><axis xyz="0 -1 0"/>'
            f'<limit lower="{lo_f}" upper="{hi_f}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{f}_tip_j" type="fixed">'
            f'<parent link="{f}_dist"/><child link="{f}_tip"/>'
            f'<origin xyz="{L[2]:.6f} 0 0"/></joint>\n'
        )
    x.append("</robot>\n")
    return "".join(x)


@dataclass
class CapsuleHand:
    """A loaded Pinocchio hand + its capsule endpoint frames."""
    model: object
    data: object
    params: HandParams
    # each capsule: (frame_id_a, frame_id_b, radius)
    capsules: list[tuple[int, int, float]] = field(default_factory=list)
    # LM index -> Pinocchio frame id (WRIST + the 4 joints per finger sit exactly
    # on the LM points by construction), for the landmark-anchor residual
    lm_frames: dict[int, int] = field(default_factory=dict)
    q_lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_hi: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv


def build(params: HandParams) -> CapsuleHand:
    import pinocchio as pin

    model = pin.buildModelFromXML(_urdf(params))
    data = model.createData()

    fid = model.getFrameId
    caps: list[tuple[int, int, float]] = []
    lm_frames: dict[int, int] = {int(LM.WRIST): fid("wrist_link")}
    # A single broad capsule keeps the proven capsule-distance Jacobian while
    # approximating the palm plate without overlapping five primitives.
    caps.append((fid("wrist_link"), fid("middle_prox"), params.palm_r))
    for f, lms in FINGERS.items():
        r = params.radius[f]
        caps.append((fid(f"{f}_prox"), fid(f"{f}_mid"), float(r[0])))
        caps.append((fid(f"{f}_mid"), fid(f"{f}_dist"), float(r[1])))
        caps.append((fid(f"{f}_dist"), fid(f"{f}_tip"), float(r[2])))
        for lm, fr in zip(lms, (f"{f}_prox", f"{f}_mid", f"{f}_dist", f"{f}_tip")):
            lm_frames[int(lm)] = fid(fr)

    return CapsuleHand(
        model=model, data=data, params=params, capsules=caps, lm_frames=lm_frames,
        q_lo=np.asarray(model.lowerPositionLimit, float),
        q_hi=np.asarray(model.upperPositionLimit, float),
    )


# ── FK + warm start ───────────────────────────────────────────────────────


def _fk(hand: CapsuleHand, q: np.ndarray) -> None:
    import pinocchio as pin

    pin.forwardKinematics(hand.model, hand.data, np.asarray(q, float))
    pin.updateFramePlacements(hand.model, hand.data)


def fk_capsule_endpoints(hand: CapsuleHand, q: np.ndarray) -> np.ndarray:
    """(C, 2, 3) world positions of every capsule's two endpoints for config ``q``."""
    _fk(hand, q)
    P = hand.data.oMf
    out = np.empty((len(hand.capsules), 2, 3), float)
    for i, (a, b, _r) in enumerate(hand.capsules):
        out[i, 0] = P[a].translation
        out[i, 1] = P[b].translation
    return out


def fk_landmark_positions(hand: CapsuleHand, q: np.ndarray, lm_order: list[int]) -> np.ndarray:
    """(len(lm_order), 3) world positions of the LM-anchored frames for ``q``."""
    _fk(hand, q)
    P = hand.data.oMf
    return np.array([P[hand.lm_frames[int(lm)]].translation for lm in lm_order], float)


def fk_capsule_and_landmarks(
    hand: CapsuleHand, q: np.ndarray, lm_order: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """One FK pass, both readouts: ``((C, 2, 3) capsule endpoints, (L, 3) LM frames)``.

    The residual assembly needs both every iteration; folding them into a single
    ``forwardKinematics`` call halves the FK cost on the fit's hot path.
    """
    _fk(hand, q)
    P = hand.data.oMf
    ep = np.empty((len(hand.capsules), 2, 3), float)
    for i, (a, b, _r) in enumerate(hand.capsules):
        ep[i, 0] = P[a].translation
        ep[i, 1] = P[b].translation
    lm = np.array([P[hand.lm_frames[int(i)]].translation for i in lm_order], float) \
        if lm_order else np.empty((0, 3), float)
    return ep, lm


def capsule_radii(hand: CapsuleHand) -> np.ndarray:
    return np.array([r for _a, _b, r in hand.capsules], float)


def _signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Full u→v angle with its sign determined by ``axis``.

    Landmarks are noisy and their measured bend is rarely perfectly coplanar
    with the model's flexion axis. Using ``atan2(axis·cross, dot)`` therefore
    collapses a real bend toward zero as soon as the cross product tilts away
    from that axis. Keep the full ``acos(dot)`` magnitude and use the projected
    cross product only for the sign.
    """
    u = np.asarray(u, float); v = np.asarray(v, float); axis = np.asarray(axis, float)
    nu, nv, na = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(axis)
    if min(nu, nv, na) < 1e-9:
        return 0.0
    u, v, axis = u / nu, v / nv, axis / na
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    magnitude = float(np.arccos(dot))
    projected_cross = float(np.dot(axis, np.cross(u, v)))
    return magnitude if projected_cross >= 0.0 else -magnitude


def q_from_landmarks(hand: CapsuleHand, pts: Mapping[LM, np.ndarray]) -> np.ndarray:
    """Geometric warm start for ``q`` from a fused ``{LM: world xyz}`` frame.

    Wrist SE(3) is taken from the palm frame. Abduction is the signed in-plane
    component of the proximal bone relative to its calibrated root direction;
    flexion is signed about the calibrated finger flexion axis. This removes the
    old ``arccos`` ambiguity and initializes abduction. Missing landmarks leave
    the corresponding joints at rest.
    """
    import pinocchio as pin

    q = pin.neutral(hand.model).copy()
    pf = _palm_frame(pts)
    if pf is not None:
        w, R = pf
        se3 = pin.SE3(R, w)
        q[:7] = pin.SE3ToXYZQUAT(se3)      # free-flyer: xyz + xyzw quaternion

    for f, lms in FINGERS.items():
        pp = [pts.get(lm) for lm in lms]
        if any(p is None or not np.all(np.isfinite(p)) for p in pp):
            continue
        pp = [np.asarray(p, float) for p in pp]
        bones = [pp[1] - pp[0], pp[2] - pp[1], pp[3] - pp[2]]
        abd, j_mcp, j_pip, j_dip = _JOINTS[f]
        R_palm = R if pf is not None else np.eye(3)
        R_root = pin.rpy.rpyToMatrix(np.asarray(hand.params.root_rpy[f], float))
        d0 = R_root.T @ R_palm.T @ bones[0]
        abd_angle = float(np.arctan2(d0[1], d0[0]))
        ca, sa = np.cos(abd_angle), np.sin(abd_angle)
        R_abd = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        # The active right-hand model curls toward palm +z, hence local -y.
        # Keep this identical to the URDF joint axes above or the warm start
        # will bend toward the back of the hand.
        flex_axis_w = R_palm @ R_root @ R_abd @ np.array([0.0, -1.0, 0.0])
        rest_dir_w = R_palm @ R_root @ R_abd @ np.array([1.0, 0.0, 0.0])
        ang = {
            abd: abd_angle,
            j_mcp: _signed_angle(rest_dir_w, bones[0], flex_axis_w),
            j_pip: _signed_angle(bones[0], bones[1], flex_axis_w),
            j_dip: _signed_angle(bones[1], bones[2], flex_axis_w),
        }
        for jname, a in ang.items():
            jid = hand.model.getJointId(jname)
            qi = hand.model.joints[jid].idx_q
            lo, hi = _limits(jname)
            q[qi] = float(np.clip(a, lo, hi))
    return q
