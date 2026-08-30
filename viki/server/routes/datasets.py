"""
viki.server.routes.datasets
---------------------------
CRUD over the on-disk capture layout: datasets (folders under DATASETS_DIR) and
the episode directories inside them. See :mod:`viki.datasets`.

Two routers: ``router`` (/datasets/*) and ``ep_router`` (/episodes/*). Episode
mutations live under their own prefix so they can't be shadowed by the
``/datasets/{name}`` routes.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from viki import datasets

router = APIRouter(prefix="/datasets", tags=["datasets"])
ep_router = APIRouter(prefix="/episodes", tags=["episodes"])


class _Name(BaseModel):
    name: str


class _Rename(BaseModel):
    new: str


class _EpPath(BaseModel):
    path: str


class _EpRename(BaseModel):
    path: str
    new_id: str


class _EpMove(BaseModel):
    path: str
    dataset: str


def _run(fn, *args):
    try:
        result = fn(*args)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    out = {"status": "ok"}
    if isinstance(result, Path):
        out["path"] = str(result)
    return out


# ── datasets ──────────────────────────────────────────────────────────────


@router.get("")
async def list_datasets():
    return {"datasets": datasets.list_datasets()}


@router.get("/{name}/episodes")
async def list_dataset_episodes(name: str):
    return {"episodes": datasets.list_episodes(name)}


@router.post("")
async def create_dataset(body: _Name):
    return _run(datasets.create_dataset, body.name)


@router.patch("/{name}")
async def rename_dataset(name: str, body: _Rename):
    return _run(datasets.rename_dataset, name, body.new)


@router.delete("/{name}")
async def delete_dataset(name: str):
    return _run(datasets.delete_dataset, name)


# ── episodes ──────────────────────────────────────────────────────────────


@ep_router.delete("")
async def delete_episode(body: _EpPath):
    return _run(datasets.delete_episode, body.path)


@ep_router.patch("/rename")
async def rename_episode(body: _EpRename):
    return _run(datasets.rename_episode, body.path, body.new_id)


@ep_router.patch("/move")
async def move_episode(body: _EpMove):
    return _run(datasets.move_episode, body.path, body.dataset)
