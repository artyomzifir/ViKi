"""
viki.datasets
-------------
On-disk grouping of recordings. A *dataset* here is just a folder under
``DATASETS_DIR`` that holds episode directories:

    data/datasets/<dataset>/<episode-id>/   meta.json status.json raw/ rec.npz ...

This is the capture-time layout, chosen for convenience — it is **not** the
LeRobot on-disk format. Turning eligible episodes into a LeRobot dataset is a
separate, later step (:mod:`viki.export`).

Everything here is plain filesystem CRUD with a guard that keeps operations
inside ``DATASETS_DIR`` / ``EPISODES_DIR``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from viki import config
from viki.contracts import Episode
from viki.episode import read_status


def datasets_root() -> Path:
    return Path(getattr(config, "DATASETS_DIR", "data/datasets"))


def episodes_root() -> Path:
    """Legacy flat directory kept as a fallback for pre-dataset episodes."""
    return Path(getattr(config, "EPISODES_DIR", "data/episodes"))


def safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_. ").strip()
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"invalid name: {name!r}")
    return cleaned


def dataset_dir(name: str) -> Path:
    return datasets_root() / safe_name(name)


def _assert_inside(path: Path) -> Path:
    """Resolve ``path`` and require it to live under datasets/ or episodes/."""
    p = path.resolve()
    roots = [datasets_root().resolve(), episodes_root().resolve()]
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError(f"path escapes the data roots: {path}")
    return p


# ── datasets ──────────────────────────────────────────────────────────────


def list_datasets() -> list[dict]:
    root = datasets_root()
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        eps = [e for e in d.iterdir() if e.is_dir()]
        out.append({"name": d.name, "path": str(d), "episodes": len(eps)})
    return out


def create_dataset(name: str) -> Path:
    d = dataset_dir(name)
    if d.exists():
        raise FileExistsError(f"dataset {name!r} already exists")
    d.mkdir(parents=True)
    return d


def rename_dataset(old: str, new: str) -> Path:
    src, dst = dataset_dir(old), dataset_dir(new)
    if not src.is_dir():
        raise FileNotFoundError(f"no dataset {old!r}")
    if dst.exists():
        raise FileExistsError(f"dataset {new!r} already exists")
    src.rename(dst)
    return dst


def delete_dataset(name: str) -> None:
    d = _assert_inside(dataset_dir(name))
    if d.is_dir():
        shutil.rmtree(d)


# ── episodes ──────────────────────────────────────────────────────────────


def _episode_summary(d: Path, dataset: str | None) -> dict:
    ep = Episode(root=d)
    meta = json.loads(ep.meta_path.read_text()) if ep.meta_path.exists() else {}
    return {
        "id": ep.id,
        "path": str(d),
        "dataset": dataset,
        "task": (meta.get("labels") or {}).get("task", meta.get("task", "")),
        "created": meta.get("created", ""),
        "stages": read_status(ep).get("stages", {}),
        "has": {
            "raw": ep.raw_dir.is_dir(),
            "rec": ep.rec_npz.exists(),
            "cln": ep.cln_npz.exists(),
            "plan": ep.plan_h5.exists(),
            "replay": ep.replay_h5.exists(),
        },
    }


def list_episodes(dataset: str | None = None) -> list[dict]:
    """Episodes in one dataset, or across all datasets + the legacy flat dir."""
    out: list[dict] = []
    if dataset is not None:
        d = dataset_dir(dataset)
        if d.is_dir():
            for e in sorted((p for p in d.iterdir() if p.is_dir()), reverse=True):
                out.append(_episode_summary(e, dataset))
        return out

    root = datasets_root()
    root.mkdir(parents=True, exist_ok=True)
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        for e in sorted((p for p in ds.iterdir() if p.is_dir()), reverse=True):
            out.append(_episode_summary(e, ds.name))

    legacy = episodes_root()
    if legacy.is_dir():
        for e in sorted((p for p in legacy.iterdir() if p.is_dir()), reverse=True):
            out.append(_episode_summary(e, None))
    return out


def delete_episode(path: str) -> None:
    p = _assert_inside(Path(path))
    if p.is_dir():
        shutil.rmtree(p)


def rename_episode(path: str, new_id: str) -> Path:
    p = _assert_inside(Path(path))
    if not p.is_dir():
        raise FileNotFoundError(f"no episode at {path}")
    dst = p.parent / safe_name(new_id)
    if dst.exists():
        raise FileExistsError(f"{new_id!r} already exists in this dataset")
    p.rename(dst)
    return dst


def move_episode(path: str, dataset: str) -> Path:
    p = _assert_inside(Path(path))
    if not p.is_dir():
        raise FileNotFoundError(f"no episode at {path}")
    target = dataset_dir(dataset)
    target.mkdir(parents=True, exist_ok=True)
    dst = target / p.name
    if dst.exists():
        raise FileExistsError(f"{p.name!r} already exists in {dataset!r}")
    shutil.move(str(p), str(dst))
    return dst
