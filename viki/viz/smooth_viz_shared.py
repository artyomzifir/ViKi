from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothVizConfig:
    show_raw: bool = True
    show_smooth: bool = True
    axes_length: float = 1.0
    center_on: str = "world"


def extract_wrist(
    raw_points: np.ndarray, landmark_ids: np.ndarray
) -> np.ndarray:
    wrist_idx = int(np.where(landmark_ids == 0)[0][0])
    return np.asarray(raw_points[:, wrist_idx, :], dtype=np.float64)
