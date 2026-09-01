"""
viki.perception.hand_model
--------------------------
A licence-free parametric articulated **capsule** hand, rigged to the
:class:`viki.contracts.LM` 21-point topology, for fitting to the per-frame hand
point cloud (:mod:`viki.perception.hand_fit`).

Why a capsule model, why these radii
------------------------------------
The hand is a kinematic tree: a free-flyer ``wrist`` (the 6-DOF we ultimately
want) plus 5 finger chains. Each phalanx and each palm metacarpal is a **capsule**
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
_PALM_R = 0.012                        # m — metacarpal capsule radius
_FLEX_LIM = (-0.25, 1.95)              # rad
_ABD_LIM = (-0.45, 0.45)
_THUMB_ABD_LIM = (-0.7, 1.0)
_THUMB_FLEX_LIM = (-0.4, 1.4)


@dataclass
class HandParams:
    """Calibrated rest geometry, all in the palm frame (x≈forward, z≈palm normal)."""
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

    return HandParams(root_xyz, root_rpy, bone_len, radius)


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
            f'<origin xyz="0 0 0"/><axis xyz="0 1 0"/>'
            f'<limit lower="{lo_f}" upper="{hi_f}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{j_pip}" type="revolute">'
            f'<parent link="{f}_prox"/><child link="{f}_mid"/>'
            f'<origin xyz="{L[0]:.6f} 0 0"/><axis xyz="0 1 0"/>'
            f'<limit lower="{lo_f}" upper="{hi_f}" effort="1" velocity="1"/></joint>\n'
            f'  <joint name="{j_dip}" type="revolute">'
            f'<parent link="{f}_mid"/><child link="{f}_dist"/>'
            f'<origin xyz="{L[1]:.6f} 0 0"/><axis xyz="0 1 0"/>'
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
    for f in _JOINTS:
        r = params.radius[f]
        caps.append((fid("wrist_link"), fid(f"{f}_prox"), params.palm_r))
        caps.append((fid(f"{f}_prox"), fid(f"{f}_mid"), float(r[0])))
        caps.append((fid(f"{f}_mid"), fid(f"{f}_dist"), float(r[1])))
        caps.append((fid(f"{f}_dist"), fid(f"{f}_tip"), float(r[2])))

    return CapsuleHand(
        model=model, data=data, params=params, capsules=caps,
        q_lo=np.asarray(model.lowerPositionLimit, float),
        q_hi=np.asarray(model.upperPositionLimit, float),
    )


# ── FK + warm start ───────────────────────────────────────────────────────


def fk_capsule_endpoints(hand: CapsuleHand, q: np.ndarray) -> np.ndarray:
    """(C, 2, 3) world positions of every capsule's two endpoints for config ``q``."""
    import pinocchio as pin

    q = np.asarray(q, float)
    pin.forwardKinematics(hand.model, hand.data, q)
    pin.updateFramePlacements(hand.model, hand.data)
    P = hand.data.oMf
    out = np.empty((len(hand.capsules), 2, 3), float)
    for i, (a, b, _r) in enumerate(hand.capsules):
        out[i, 0] = P[a].translation
        out[i, 1] = P[b].translation
    return out


def capsule_radii(hand: CapsuleHand) -> np.ndarray:
    return np.array([r for _a, _b, r in hand.capsules], float)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))


def q_from_landmarks(hand: CapsuleHand, pts: Mapping[LM, np.ndarray]) -> np.ndarray:
    """Geometric warm start for ``q`` from a fused ``{LM: world xyz}`` frame.

    Wrist SE(3) is taken exactly from the palm frame; each finger's flexion
    angles are the (unsigned) angles between consecutive landmark bones, clamped
    to the joint limits. Missing landmarks → that joint stays at 0 (rest).
    """
    import pinocchio as pin

    q = pin.neutral(hand.model).copy()
    pf = _palm_frame(pts)
    if pf is not None:
        w, R = pf
        se3 = pin.SE3(R, w)
        q[:7] = pin.SE3ToXYZQUAT(se3)      # free-flyer: xyz + xyzw quaternion

    # forward direction of the palm (local +x) in world, for the MCP flex angle
    fwd_w = (R @ np.array([1.0, 0.0, 0.0])) if pf is not None else np.array([1.0, 0.0, 0.0])

    for f, lms in FINGERS.items():
        pp = [pts.get(lm) for lm in lms]
        if any(p is None or not np.all(np.isfinite(p)) for p in pp):
            continue
        pp = [np.asarray(p, float) for p in pp]
        bones = [pp[1] - pp[0], pp[2] - pp[1], pp[3] - pp[2]]
        abd, j_mcp, j_pip, j_dip = _JOINTS[f]
        ang = {
            j_mcp: _angle(fwd_w, bones[0]),
            j_pip: _angle(bones[0], bones[1]),
            j_dip: _angle(bones[1], bones[2]),
        }
        for jname, a in ang.items():
            jid = hand.model.getJointId(jname)
            qi = hand.model.joints[jid].idx_q
            lo, hi = _limits(jname)
            q[qi] = float(np.clip(a, lo, hi))
    return q
