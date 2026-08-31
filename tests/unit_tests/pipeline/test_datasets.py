"""viki.datasets — on-disk dataset/episode CRUD + the path-safety guard."""

import pytest

from viki import datasets
from viki.episode import new_episode


@pytest.fixture
def roots(tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    eps = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(ds), raising=False)
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(eps), raising=False)
    return ds, eps


def test_dataset_crud(roots):
    ds, _ = roots
    datasets.create_dataset("rig-A")
    assert (ds / "rig-A").is_dir()
    assert [d["name"] for d in datasets.list_datasets()] == ["rig-A"]

    with pytest.raises(FileExistsError):
        datasets.create_dataset("rig-A")

    datasets.rename_dataset("rig-A", "rig-B")
    assert (ds / "rig-B").is_dir() and not (ds / "rig-A").exists()

    datasets.delete_dataset("rig-B")
    assert not (ds / "rig-B").exists()


def test_list_and_move_episodes(roots):
    ds, eps = roots
    datasets.create_dataset("d1")
    ep = new_episode(ds / "d1", {"task": "pick"})

    listed = datasets.list_episodes("d1")
    assert len(listed) == 1 and listed[0]["task"] == "pick" and listed[0]["dataset"] == "d1"

    # a legacy flat episode shows up in the un-filtered listing
    new_episode(eps, {"task": "old"})
    allep = datasets.list_episodes()
    assert {e["dataset"] for e in allep} == {"d1", None}

    datasets.create_dataset("d2")
    datasets.move_episode(ep.root, "d2")
    assert datasets.list_episodes("d1") == []
    assert len(datasets.list_episodes("d2")) == 1


def test_path_safety(roots, tmp_path):
    for bad in ("/etc", str(tmp_path / "elsewhere"), "../../secrets"):
        with pytest.raises(ValueError):
            datasets.delete_episode(bad)


def test_rename_episode(roots):
    ds, _ = roots
    datasets.create_dataset("d")
    ep = new_episode(ds / "d")
    dst = datasets.rename_episode(ep.root, "take-01")
    assert dst.name == "take-01" and dst.is_dir()


def test_update_episode_meta(roots):
    import json

    ds, _ = roots
    datasets.create_dataset("d")
    ep = new_episode(ds / "d", {"task": "pick", "hand": "right", "demonstrator": "a"})

    datasets.update_episode_meta(ep.root, task="pour the cup", hand="left")

    # directory id is untouched; meta.json carries the edits
    assert (ds / "d" / ep.id).is_dir()
    summary = datasets.list_episodes("d")[0]
    assert summary["task"] == "pour the cup"
    assert summary["hand"] == "left"
    assert summary["demonstrator"] == "a"  # left alone when not passed
    meta = json.loads((ep.root / "meta.json").read_text())
    assert meta["hand"] == "left" and meta["task"] == "pour the cup"

    with pytest.raises(ValueError):
        datasets.update_episode_meta(ep.root, hand="middle")
