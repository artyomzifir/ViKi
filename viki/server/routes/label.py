"""viki.server.routes.label — read/write an episode's EpisodeLabels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki.contracts import Episode, EpisodeLabels, Segment
from viki.labeling import load_labels, save_labels, validate_labels

router = APIRouter(prefix="/label", tags=["label"])


class SegmentIn(BaseModel):
    start: int
    end: int
    label: str


class LabelsIn(BaseModel):
    task: str = ""
    hand: str = "right"
    segments: list[SegmentIn] = []
    outcome: str = "unrated"
    notes: str = ""


def _episode(path: str) -> Episode:
    ep = Episode(root=Path(path))
    if not ep.root.is_dir():
        raise HTTPException(404, f"no episode at {path}")
    return ep


@router.get("")
async def get_labels(episode: str):
    return asdict(load_labels(_episode(episode)))


@router.post("")
async def set_labels(episode: str, body: LabelsIn):
    ep = _episode(episode)
    labels = EpisodeLabels(
        task=body.task,
        hand=body.hand,
        segments=[Segment(**s.model_dump()) for s in body.segments],
        outcome=body.outcome,
        notes=body.notes,
    )
    n = 0
    if ep.cln_npz.exists():
        import numpy as np

        with np.load(ep.cln_npz) as d:
            n = len(d["positions"])
    try:
        validate_labels(labels, n_frames=n or 10**9)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    save_labels(ep, labels)
    return {"status": "ok"}
