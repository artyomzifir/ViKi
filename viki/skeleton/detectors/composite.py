"""
viki.skeleton.detectors.composite
---------------------------------
Aggregates N PartialLandmarkDetector instances into one HandDetection per
frame in the global N-slot layout (default n_slots = LM.N = 23).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from viki.skeleton.detectors.base import (
    FusionMode,
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.skeleton.models import HandDetection, LM, PreparedFrame

logger = logging.getLogger(__name__)


class CompositeLandmarkDetector:
    """
    Orchestrates a list of partial detectors into a single HandDetection.

    The composite is layout‑agnostic: pass `n_slots` to use it for setups
    other than the default 23‑slot skeleton.

    Attributes
    ----------
    n_slots : int
        Number of output slots.
    mode : FusionMode
        Fusion strategy (ANY or ALL).
    _detectors : List[PartialLandmarkDetector]
        List of constituent detectors.
    _warned_shape : set[str]
        Cache for shape warnings to avoid log spam.
    """

    def __init__(
        self,
        detectors: List[PartialLandmarkDetector],
        mode: FusionMode = FusionMode.ANY,
        n_slots: int = LM.N,
    ) -> None:
        """
        Parameters
        ----------
        detectors : List[PartialLandmarkDetector]
            Partial detectors to merge.
        mode : FusionMode, default=FusionMode.ANY
            Fusion strategy.
        n_slots : int, default=LM.N
            Size of the output layout; every detector's indices must lie in [0, n_slots).

        Raises
        ------
        ValueError
            If `n_slots < 1`, if any detector index is out of range,
            or if two detectors share the same name.
        """
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")

        if not detectors:
            logger.warning(
                "CompositeLandmarkDetector: empty detector list — "
                "detect() will always return None"
            )

        self._detectors: List[PartialLandmarkDetector] = list(detectors)
        self._mode = mode
        self._n = n_slots

        # Validate detector slots and unique names once, fail loudly.
        seen_names: set[str] = set()
        for d in self._detectors:
            for i in d.indices:
                if not (0 <= i < n_slots):
                    raise ValueError(
                        f"Detector {d.name!r} indices contain {i}, "
                        f"out of [0, {n_slots})"
                    )
            if d.name in seen_names:
                raise ValueError(f"Duplicate detector name: {d.name!r}")
            seen_names.add(d.name)

        # log-once cache for shape warnings per detector
        self._warned_shape: set[str] = set()

    @property
    def n_slots(self) -> int:
        """Number of output slots."""
        return self._n

    @property
    def mode(self) -> FusionMode:
        """Fusion mode."""
        return self._mode

    def detect(self, frame: PreparedFrame) -> Optional[HandDetection]:
        """
        Run every partial detector and merge their outputs.

        Parameters
        ----------
        frame : PreparedFrame
            Prepared camera frame.

        Returns
        -------
        Optional[HandDetection]
            HandDetection in the n_slots layout, or None if the fusion policy
            is not satisfied or no landmarks were produced.
        """
        if not self._detectors:
            return None

        # Run all detectors; isolate exceptions per detector.
        results: List[Tuple[PartialLandmarkDetector, Optional[PartialDetection2D]]] = []
        for d in self._detectors:
            try:
                results.append((d, d.detect(frame)))
            except Exception:
                logger.exception(
                    "Detector %r raised in detect(); treating as None", d.name
                )
                results.append((d, None))

        successes = [(d, r) for (d, r) in results if r is not None]

        # Apply fusion-mode policy.
        if self._mode == FusionMode.ALL:
            if len(successes) < len(self._detectors):
                return None
        else:
            if not successes:
                return None

        # Merge into the global N-slot buffers.
        px = {LM(idx): np.full(2, np.nan, dtype=np.float32) for idx in range(self._n)}
        z = np.full(self._n, np.nan, dtype=np.float32)
        per_slot_conf = np.zeros(self._n, dtype=np.float32)
        slot_owner_priority: Dict[int, int] = {}

        for d, partial in sorted(successes, key=lambda dr: dr[0].priority):
            # assert partial is not None  # narrowed by the `successes` filter

            if partial.px.shape[0] != len(d.indices):
                if d.name not in self._warned_shape:
                    logger.warning(
                        "Detector %r returned px shape %s but declares "
                        "%d indices; skipping its contribution",
                        d.name, partial.px.shape, len(d.indices),
                    )
                    self._warned_shape.add(d.name)
                continue

            for k, slot in enumerate(d.indices):
                if slot in slot_owner_priority:
                    continue  # higher-priority detector already wrote here
                p = partial.px[k]
                if np.isnan(p).any():
                    continue
                px[LM(slot)] = p
                z[slot] = partial.lm_z_rel[k]
                per_slot_conf[slot] = partial.per_index_confidence[k]
                slot_owner_priority[slot] = d.priority

        if not slot_owner_priority:
            # Nothing usable was written even though detectors "succeeded".
            return None

        # Reduce per-slot confidences to a scalar over filled slots only.
        filled = np.fromiter(slot_owner_priority.keys(), dtype=int)
        scalar_conf = float(per_slot_conf[filled].mean())

        # device_id / timestamp_us are shared across partials in one frame —
        # take from any success.
        any_partial = successes[0][1]
        assert any_partial is not None
        return HandDetection(
            points=px,
            lm_z_rel=z,
            confidence=scalar_conf,
            device_id=any_partial.device_id,
            timestamp_us=any_partial.timestamp_us,
        )

    def close(self) -> None:
        """Close every contained partial detector, isolating exceptions."""
        for d in self._detectors:
            try:
                d.close()
            except Exception:
                logger.exception("Detector %r raised during close()", d.name)

    # def __enter__(self) -> "CompositeLandmarkDetector":
    #     return self

    # def __exit__(self, *exc) -> None:
    #     self.close()
