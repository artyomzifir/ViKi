"""
viki.perception.run
-------------------
The whole perception stage as one call: ``raw/`` → the weighted, smoothed,
cross-camera-fused hand skeleton + end-effector pose + gripper state
(``rec.npz`` then ``cln.npz``) that the IK solver consumes, plus (optionally)
the per-frame coloured point cloud for the viewer.

``perceive_episode`` = ``extract_episode`` (detect → lift → per-camera world
keypoints) + ``prepare_episode`` (interpolate → fuse → spline → smooth → EE pose
→ gripper) + ``build_cloud``. It takes a ``report`` callback so the background
queue can show progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerceiveOpts:
    model: str = "mediapipe"          # registry.MODELS id
    hand: str = "right"
    track_lm: list[int] | None = None
    min_confidence: float = 0.5
    interp_max_gap: int = 0
    sg_window: int = 7
    sg_polyorder: int = 2
    flip: bool = False
    build_cloud: bool = False
    cloud_stride: int = 6

    @classmethod
    def from_dict(cls, d: dict | None) -> "PerceiveOpts":
        from viki import config as cfg

        d = dict(d or {})
        return cls(
            model=(d.get("model") or d.get("backend")            # 'backend' = legacy
                   or getattr(cfg, "POSE_BACKEND", None) or "mediapipe"),
            hand=d.get("hand") or "right",
            track_lm=(
                list(d["track_lm"]) if d.get("track_lm")
                else list(getattr(cfg, "PERCEPTION_TRACK_LM", []) or []) or None
            ),
            min_confidence=float(d.get("min_confidence", 0.5)),
            interp_max_gap=int(d.get("interp_max_gap", getattr(cfg, "PERCEPTION_INTERP_MAX_GAP", 0))),
            sg_window=int(d.get("sg_window", getattr(cfg, "RETARGET_LANDMARK_SG_WINDOW", 7))),
            sg_polyorder=int(d.get("sg_polyorder", getattr(cfg, "RETARGET_LANDMARK_SG_POLYORDER", 2))),
            flip=bool(d.get("flip", False)),
            build_cloud=bool(d.get("build_cloud", False)),
            cloud_stride=int(d.get("cloud_stride", getattr(cfg, "CLOUD_STRIDE", 6))),
        )


def _noop(**_kw):
    return None


def perceive_episode(ep, opts: PerceiveOpts | dict | None = None, report=None) -> str:
    """Run the full perception stage for one episode. Returns the cln.npz path."""
    from viki.perception.cloud import build_cloud
    from viki.perception.extract import extract_episode
    from viki.prepare.run import prepare_episode

    if not isinstance(opts, PerceiveOpts):
        opts = PerceiveOpts.from_dict(opts)
    report = report or _noop

    report(stage="extract", frame=0, total=0)
    extract_episode(
        ep,
        model=opts.model,
        hand=opts.hand,
        track_lm=opts.track_lm,
        min_confidence=opts.min_confidence,
        flip=opts.flip,
        report=report,
    )

    report(stage="fuse")
    prepare_episode(ep, opts.sg_window, opts.sg_polyorder, interp_max_gap=opts.interp_max_gap)

    if opts.build_cloud:
        report(stage="cloud")
        build_cloud(ep, stride=opts.cloud_stride)

    report(stage="done")
    logger.info("perceive %s -> %s", ep.id, ep.cln_npz)
    return str(ep.cln_npz)
