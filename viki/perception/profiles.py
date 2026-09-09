"""Named, immutable perception recipes used as reproducible baselines.

Configuration remains useful for experiments.  A named profile is different:
its parameters are code-owned and must not silently change when
``user_configuration.json`` is edited between recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from viki.contracts import HAND_LM_COUNT


CLEAN_LANDMARKS_V1 = "clean-triangulated-landmarks-v1"
STABLE_FUSED_HAND_V1 = "stable-fused-hand-v1"
STABLE_FUSED_HAND_V2 = "stable-fused-hand-v2"
FUSED_HAND_NO_EXTRAP_V1 = "fused-hand-no-extrap-v1"
STABLE_FUSED_HAND_V3 = "stable-fused-hand-v3"
# Keep the absolute-confidence V3 available for controlled A/B runs, but use
# the already validated V2 recipe as the product baseline until V3 is shown to
# improve fixed-episode metrics without reducing usable observations.
DEFAULT_PERCEPTION_PROFILE = STABLE_FUSED_HAND_V2


@dataclass(frozen=True)
class PerceptionProfile:
    name: str
    description: str
    detector_model: str
    min_confidence: float
    depth_radius_px: int
    save_observations: bool
    track_lm: tuple[int, ...]
    flip: bool
    fusion_mode: str
    fused_interpolation: str
    fused_extrapolate_edges: bool
    interp_max_gap: int
    sg_window: int
    sg_polyorder: int
    confidence_alpha: float
    gripper: str
    coordinate_frame: str
    hand_fit: bool
    pose_source: str
    confidence_calibration: str = "episode_max"
    tracking_confidence: float | None = None
    articulated_hand_fit: str | None = None
    triangulation: dict[str, object] = field(default_factory=dict)

    def manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["track_lm"] = list(self.track_lm)
        # Keep the already-protected CLEAN_LANDMARKS_V1 manifest byte-for-byte
        # compatible.  The post-process field is part of a profile only when
        # that profile actually owns an articulated stage.
        if payload["articulated_hand_fit"] is None:
            payload.pop("articulated_hand_fit")
        # Preserve historical V1 manifests. Cubic was the implicit fused-fill
        # implementation before the method became an explicit profile knob.
        if payload["fused_interpolation"] == "cubic":
            payload.pop("fused_interpolation")
        if payload["fused_extrapolate_edges"] is True:
            payload.pop("fused_extrapolate_edges")
        # V1/V2 manifests predate this field and remain byte-for-byte immutable.
        if payload["confidence_calibration"] == "episode_max":
            payload.pop("confidence_calibration")
        if payload["tracking_confidence"] is None:
            payload.pop("tracking_confidence")
        return payload


_PROFILES = {
    CLEAN_LANDMARKS_V1: PerceptionProfile(
        name=CLEAN_LANDMARKS_V1,
        description=(
            "Validated pick_up_u baseline: MediaPipe observations, strict "
            "two-view triangulation, unlimited coordinate gap fill, "
            "Savitzky-Golay 7/2, landmark-derived pose, no capsule hand fit."
        ),
        detector_model="mediapipe",
        min_confidence=0.5,
        depth_radius_px=15,
        save_observations=True,
        track_lm=tuple(range(HAND_LM_COUNT)),
        flip=False,
        fusion_mode="triangulate",
        fused_interpolation="cubic",
        fused_extrapolate_edges=True,
        interp_max_gap=0,
        sg_window=7,
        sg_polyorder=2,
        confidence_alpha=1.0,
        gripper="binary",
        # Historical label retained because V1 manifests are immutable. The
        # arrays are physically in rig coordinates; retarget migrates the label.
        coordinate_frame="robot_base",
        hand_fit=False,
        pose_source="landmarks",
        triangulation={
            "min_score": 0.3,
            "min_ray_deg": 5.0,
            "reproj_inlier_px": 4.0,
            "depth_lambda": 0.1,
            "depth_delta_m": 0.01,
            "depth_spread_scale_m": 0.02,
            "ray_ref_deg": 20.0,
            "loss": "soft_l1",
            "geometry_cameras": [],
        },
    ),
    STABLE_FUSED_HAND_V1: PerceptionProfile(
        name=STABLE_FUSED_HAND_V1,
        description=(
            "Stable perception v1: the validated clean triangulated trajectory "
            "is routed to fused, then landmark-only articulated-landmarks-v1 "
            "is routed to the independent hand_fit overlay."
        ),
        detector_model="mediapipe",
        min_confidence=0.5,
        depth_radius_px=15,
        save_observations=True,
        track_lm=tuple(range(HAND_LM_COUNT)),
        flip=False,
        fusion_mode="triangulate",
        fused_interpolation="cubic",
        fused_extrapolate_edges=True,
        interp_max_gap=0,
        sg_window=7,
        sg_polyorder=2,
        confidence_alpha=1.0,
        gripper="binary",
        # Historical label retained because V1 manifests are immutable. The
        # arrays are physically in rig coordinates; retarget migrates the label.
        coordinate_frame="robot_base",
        # This flag names the retired dense-cloud fitter.  It must remain off:
        # the stable profile uses only the post-process below.
        hand_fit=False,
        pose_source="landmarks",
        articulated_hand_fit="articulated-landmarks-v1",
        triangulation={
            "min_score": 0.3,
            "min_ray_deg": 5.0,
            "reproj_inlier_px": 4.0,
            "depth_lambda": 0.1,
            "depth_delta_m": 0.01,
            "depth_spread_scale_m": 0.02,
            "ray_ref_deg": 20.0,
            "loss": "soft_l1",
            "geometry_cameras": [],
        },
    ),
}

# V1 remains byte-for-byte immutable for protected baselines. V2 changes only
# the interpolator and gripper: linear filling still holds the nearest observed
# value at sequence edges, which is the control used by the 2026-09-09 A/B run.
_PROFILES[STABLE_FUSED_HAND_V2] = replace(
    _PROFILES[STABLE_FUSED_HAND_V1],
    name=STABLE_FUSED_HAND_V2,
    description=(
        "Stable perception v2: the v1 fused + articulated hand geometry with "
        "linear gap filling and continuous, temporally filtered gripper opening."
    ),
    fused_interpolation="linear",
    fused_extrapolate_edges=True,
    gripper="linear",
)

# Controlled candidate: exactly V2 except that fused measurements are not
# fabricated before their first or after their last observation.
_PROFILES[FUSED_HAND_NO_EXTRAP_V1] = replace(
    _PROFILES[STABLE_FUSED_HAND_V2],
    name=FUSED_HAND_NO_EXTRAP_V1,
    description=(
        "V2 no-extrapolation candidate: stable v2 geometry and continuous "
        "gripper with linear filling only between bracketed observations."
    ),
    fused_extrapolate_edges=False,
)

# V3 is the first profile whose confidence remains comparable between episodes.
# It intentionally retains V2's validated detector, interpolation and
# triangulation gates:
# dense zero-threshold experiments admitted more 2-D rows but produced far fewer
# geometrically valid 3-D joints. Detector scores are still supported as
# continuous evidence by backends that expose a real per-joint score.
_PROFILES[STABLE_FUSED_HAND_V3] = replace(
    _PROFILES[STABLE_FUSED_HAND_V2],
    name=STABLE_FUSED_HAND_V3,
    description=(
        "Stable perception v3: v2 detector, fused geometry and continuous "
        "gripper with absolute cross-episode confidence calibration."
    ),
    detector_model="mediapipe",
    min_confidence=0.5,
    confidence_calibration="absolute",
)


def get_profile(name: str | None) -> PerceptionProfile | None:
    if not name:
        return None
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown perception profile {name!r}; available: {sorted(_PROFILES)}"
        ) from exc


def list_profiles() -> list[dict[str, object]]:
    return [profile.manifest() for profile in _PROFILES.values()]
