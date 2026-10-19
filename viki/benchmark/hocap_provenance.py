"""Fail-closed input checks for new HO-Cap hand-fit replay/evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from viki.perception.profiles import HOCAP_STABLE_FUSED_HAND_V4


def require_hand_fit_clean(episode: Path, *, profile: str = HOCAP_STABLE_FUSED_HAND_V4) -> dict:
    """Require the current HO-Cap recipe and its installed articulated pose."""
    path = Path(episode) / "cln.npz"
    with np.load(path, allow_pickle=False) as data:
        actual = str(np.asarray(data["perception_profile"]).item())
        expected_overlay = f"articulated-landmarks-v1:optimized"
        required = {"hand_fit_positions", "hand_fit_rotations", "hand_fit_joint_angles",
                    "hand_fit_overlay_variant", "hand_fit_overlay_source_profile",
                    "hand_fit_metrics_json", "geometry_hand_params_json"}
        missing = sorted(required.difference(data.files))
        if actual != profile or missing:
            raise ValueError(f"{path}: expected profile {profile!r} with hand-fit; "
                             f"found {actual!r}, missing {missing}")
        overlay = str(np.asarray(data["hand_fit_overlay_variant"]).item())
        overlay_source = str(np.asarray(data["hand_fit_overlay_source_profile"]).item())
        count = len(data["timestamps"])
        if (overlay != expected_overlay or overlay_source != profile
                or data["hand_fit_positions"].shape != (count, 3)
                or data["hand_fit_rotations"].shape != (count, 3, 3)
                or data["hand_fit_joint_angles"].shape[0] != count
                or str(np.asarray(data["coordinate_frame"]).item()) != "hocap_tag_1"
                or not np.isfinite(data["hand_fit_positions"]).any()):
            raise ValueError(f"{path}: hand-fit overlay is incomplete or misaligned")
    status_path = Path(episode) / "status.json"
    if not status_path.is_file():
        raise ValueError(f"{status_path}: missing preparation status and quality gate")
    stages = json.loads(status_path.read_text()).get("stages", {})
    prepared = stages.get("prepare", {})
    if (not prepared.get("done") or not prepared.get("hand_fit")
            or not (prepared.get("articulated") or {}).get("quality_gate", {}).get("accepted")):
        raise ValueError(f"{status_path}: articulated quality gate is not accepted")
    return {"profile": profile, "pose_source": "hand_fit", "overlay": expected_overlay,
            "frames": count}


def require_hand_fit_plan(path: Path) -> None:
    """Reject plans whose effective IK target silently became landmarks."""
    with h5py.File(path) as plan:
        source = plan["source_pose"][()]
        if isinstance(source, bytes):
            source = source.decode()
        config = json.loads(plan["config_json"][()])
        if source != "hand_fit" or config.get("pose_source") != "hand_fit":
            raise ValueError(f"{path}: expected effective hand_fit IK pose, got {source!r}")
        if config.get("target_position_anchor") != "wrist":
            raise ValueError(
                f"{path}: hand_fit plan must use wrist anchor until fitted "
                "pinch-center routing is implemented"
            )
