"""
viki.episode
------------
Filesystem helpers for an episode directory — the unit of work that each
pipeline stage reads and extends by one artifact (see :class:`viki.contracts.Episode`).

    episodes/<id>/
      meta.json    task / demonstrator / hand / cameras / capture window / labels
      status.json  which stages have run, with per-stage notes
      raw/  rec.npz  cln.npz  plan.h5  replay.h5
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from viki.contracts import Episode

_STAGES = ("record", "extract", "prepare", "retarget", "replay", "label", "export")


def new_episode(episodes_dir: str | Path, meta: dict | None = None) -> Episode:
    """Create ``episodes/<timestamp>/`` with a fresh meta.json + status.json."""
    root = Path(episodes_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (root / "raw").mkdir(parents=True, exist_ok=True)
    ep = Episode(root=root)
    save_meta(ep, {"id": ep.id, "created": datetime.now().isoformat(), **(meta or {})})
    ep.status_path.write_text(json.dumps({"stages": {}}, indent=2))
    return ep


def load_meta(ep: Episode) -> dict:
    if not ep.meta_path.exists():
        return {}
    return json.loads(ep.meta_path.read_text())


def save_meta(ep: Episode, meta: dict) -> None:
    ep.root.mkdir(parents=True, exist_ok=True)
    ep.meta_path.write_text(json.dumps(meta, indent=2, default=str))


def read_status(ep: Episode) -> dict:
    if not ep.status_path.exists():
        return {"stages": {}}
    return json.loads(ep.status_path.read_text())


def mark_stage(ep: Episode, stage: str, **fields: Any) -> None:
    """Record that ``stage`` ran, merging ``fields`` into its status entry."""
    if stage not in _STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {', '.join(_STAGES)}")
    status = read_status(ep)
    entry = status.setdefault("stages", {}).get(stage, {})
    entry.update({"done": True, "at": datetime.now().isoformat(), **fields})
    status["stages"][stage] = entry
    ep.status_path.write_text(json.dumps(status, indent=2, default=str))


def stage_done(ep: Episode, stage: str) -> bool:
    return bool(read_status(ep).get("stages", {}).get(stage, {}).get("done"))
