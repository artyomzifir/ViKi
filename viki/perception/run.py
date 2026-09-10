"""
viki.perception.run
-------------------
The whole perception stage as one call: ``raw/`` → the weighted, smoothed,
cross-camera-fused hand skeleton + end-effector pose + gripper state
(``rec.npz`` then ``cln.npz``) that the IK solver consumes, plus (optionally)
the per-frame coloured point cloud for the viewer.

``perceive_episode`` = ``extract_episode`` (detect → lift → per-camera world
keypoints) + ``prepare_episode`` (interpolate → fuse → gap-fill → smooth → EE pose
→ gripper) + ``build_cloud``. It takes a ``report`` callback so the background
queue can show progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from viki.perception.profiles import DEFAULT_PERCEPTION_PROFILE

logger = logging.getLogger(__name__)


@dataclass
class PerceiveOpts:
    profile: str | None = DEFAULT_PERCEPTION_PROFILE
    model: str = "mediapipe"          # registry.MODELS id
    hand: str = "right"
    track_lm: list[int] | None = None
    min_confidence: float = 0.5
    interp_max_gap: int = 0
    sg_window: int = 7
    sg_polyorder: int = 2
    # Custom-path override of the triangulation agreement gate, in milliradians.
    # Stated as an angle so the same value means the same thing at any capture
    # resolution; a named profile ignores it, because a profile owns its own.
    reproj_inlier_mrad: float | None = None
    flip: bool = False
    build_cloud: bool = False
    cloud_stride: int = 6

    @classmethod
    def from_dict(cls, d: dict | None) -> "PerceiveOpts":
        from viki import config as cfg

        d = dict(d or {})
        return cls(
            # Missing means the stable versioned pipeline.  Explicit null/""
            # still selects the custom configuration path from the UI/API.
            profile=(d["profile"] or None) if "profile" in d else DEFAULT_PERCEPTION_PROFILE,
            model=(d.get("model") or d.get("backend")            # 'backend' = legacy
                   or getattr(cfg, "POSE_BACKEND", None) or "mediapipe"),
            hand=d.get("hand") or "right",
            track_lm=(
                list(d["track_lm"]) if d.get("track_lm")
                else list(getattr(cfg, "PERCEPTION_TRACK_LM", []) or []) or None
            ),
            min_confidence=float(d.get("min_confidence", 0.5)),
            interp_max_gap=int(d.get("interp_max_gap", getattr(cfg, "PERCEPTION_INTERP_MAX_GAP", 0))),
            sg_window=int(d.get("sg_window", 7)),
            reproj_inlier_mrad=(
                float(d["reproj_inlier_mrad"])
                if d.get("reproj_inlier_mrad") not in (None, "", 0)
                else None
            ),
            sg_polyorder=int(d.get("sg_polyorder", 2)),
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
    from viki.perception.profiles import get_profile
    from viki.prepare.run import prepare_episode

    if not isinstance(opts, PerceiveOpts):
        opts = PerceiveOpts.from_dict(opts)
    profile = get_profile(opts.profile)
    if profile is not None:
        # A baseline profile owns every accuracy-affecting knob.  Hand side and
        # cloud generation remain episode/UI choices rather than algorithm
        # parameters.
        opts.model = profile.detector_model
        opts.track_lm = list(profile.track_lm)
        opts.min_confidence = profile.min_confidence
        opts.interp_max_gap = profile.interp_max_gap
        opts.sg_window = profile.sg_window
        opts.sg_polyorder = profile.sg_polyorder
        opts.flip = profile.flip
    report = report or _noop

    logger.info(
        "perceive %s: profile=%s model=%s hand=%s cloud=%s",
        ep.id, opts.profile or "config", opts.model, opts.hand, opts.build_cloud,
    )
    report(stage="extract", frame=0, total=0)
    extract_episode(
        ep,
        model=opts.model,
        hand=opts.hand,
        track_lm=opts.track_lm,
        min_confidence=opts.min_confidence,
        depth_radius_px=profile.depth_radius_px if profile is not None else None,
        depth_radius_mrad=profile.depth_radius_mrad if profile is not None else None,
        save_observations=(profile.save_observations if profile is not None else None),
        tracking_confidence=(profile.tracking_confidence if profile is not None else None),
        profile=opts.profile,
        flip=opts.flip,
        report=report,
    )

    report(stage="fuse")
    prepare_episode(ep, opts.sg_window, opts.sg_polyorder,
                    interp_max_gap=opts.interp_max_gap, report=report,
                    profile=opts.profile,
                    triangulation=(
                        {"reproj_inlier_mrad": opts.reproj_inlier_mrad}
                        if opts.reproj_inlier_mrad is not None else None
                    ))

    if opts.build_cloud:
        report(stage="cloud")
        build_cloud(ep, stride=opts.cloud_stride)

    report(stage="done")
    logger.info("perceive %s -> %s", ep.id, ep.cln_npz)
    return str(ep.cln_npz)
