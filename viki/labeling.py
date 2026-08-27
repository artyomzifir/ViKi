"""
viki.labeling
-------------
Human annotation for an episode: the natural-language task, which hand, optional
phase segments, and a pass/fail outcome. Persisted under ``meta.json["labels"]``.

This is what ``export`` turns into LeRobot's per-frame ``task`` string (deduped
into ``meta/tasks.jsonl``) and the ``next.success`` column. A non-empty ``task``
is required before an episode can be exported.
"""

from __future__ import annotations

from dataclasses import asdict

from viki.contracts import Episode, EpisodeLabels, Segment
from viki.episode import load_meta, save_meta

__all__ = ["load_labels", "save_labels", "validate_labels"]


def load_labels(ep: Episode) -> EpisodeLabels:
    raw = load_meta(ep).get("labels") or {}
    segments = [Segment(**s) for s in raw.get("segments", [])]
    return EpisodeLabels(
        task=raw.get("task", ""),
        hand=raw.get("hand", "right"),
        segments=segments,
        outcome=raw.get("outcome", "unrated"),
        notes=raw.get("notes", ""),
    )


def save_labels(ep: Episode, labels: EpisodeLabels) -> None:
    meta = load_meta(ep)
    meta["labels"] = {
        "task": labels.task,
        "hand": labels.hand,
        "segments": [asdict(s) for s in labels.segments],
        "outcome": labels.outcome,
        "notes": labels.notes,
    }
    save_meta(ep, meta)


def validate_labels(labels: EpisodeLabels, n_frames: int, *, for_export: bool = False) -> None:
    """Raise ``ValueError`` on inconsistent labels."""
    if labels.hand not in ("left", "right"):
        raise ValueError(f"hand must be left|right, got {labels.hand!r}")
    if labels.outcome not in ("good", "bad", "unrated"):
        raise ValueError(f"outcome must be good|bad|unrated, got {labels.outcome!r}")
    for s in labels.segments:
        if not (0 <= s.start < s.end <= n_frames):
            raise ValueError(
                f"segment {s.label!r} out of range: [{s.start}, {s.end}) vs {n_frames} frames"
            )
    seg_sorted = sorted(labels.segments, key=lambda s: s.start)
    for a, b in zip(seg_sorted, seg_sorted[1:]):
        if b.start < a.end:
            raise ValueError(f"segments overlap: {a.label!r} and {b.label!r}")
    if for_export and not labels.task.strip():
        raise ValueError("cannot export an episode with an empty task label")
