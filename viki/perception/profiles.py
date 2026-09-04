"""Named, immutable perception recipes used as reproducible baselines.

Configuration remains useful for experiments.  A named profile is different:
its parameters are code-owned and must not silently change when
``user_configuration.json`` is edited between recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from viki.contracts import HAND_LM_COUNT


CLEAN_LANDMARKS_V1 = "clean-triangulated-landmarks-v1"
STABLE_FUSED_HAND_V1 = "stable-fused-hand-v1"


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
    interp_max_gap: int
    sg_window: int
    sg_polyorder: int
    confidence_alpha: float
    gripper: str
    coordinate_frame: str
    hand_fit: bool
    pose_source: str
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
        interp_max_gap=0,
        sg_window=7,
        sg_polyorder=2,
        confidence_alpha=1.0,
        gripper="binary",
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
        interp_max_gap=0,
        sg_window=7,
        sg_polyorder=2,
        confidence_alpha=1.0,
        gripper="binary",
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
