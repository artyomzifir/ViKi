"""Build an object-disjoint HO-Cap train/test manifest for simulator RL."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def split_object_groups(groups: dict[tuple[str, ...], list[str]]) -> tuple[list[str], list[str]]:
    ordered = sorted(groups)
    if not ordered:
        raise ValueError("no HO-Cap scenes found")
    target = sum(map(len, groups.values())) // 2
    candidates = []
    for count in range(1, len(ordered)):
        for selected in itertools.combinations(ordered, count):
            train_count = sum(len(groups[group]) for group in selected)
            if train_count != target:
                continue
            train_names = {name for group in selected for name in group}
            if "G10_4" not in train_names:
                continue
            key = "|".join("/".join(group) for group in selected)
            candidates.append((abs(2 * count - len(ordered)), hashlib.sha256(key.encode()).hexdigest(), selected))
    if not candidates:
        raise ValueError("no balanced object-disjoint split containing the pilot object")
    selected = set(min(candidates)[2])
    train = sorted(scene for group in selected for scene in groups[group])
    test = sorted(scene for group in ordered if group not in selected for scene in groups[group])
    return train, test


def mesh_dimensions(archive: tarfile.TarFile, names: set[str]) -> dict[str, list[float]]:
    result = {}
    for name in sorted(names):
        member = archive.extractfile(f"{name}.cleaned_mesh_2000.obj")
        if member is None:
            raise ValueError(f"missing official mesh for {name}")
        vertices = [
            [float(value) for value in line.split()[1:4]]
            for line in member.read().decode().splitlines() if line.startswith("v ")
        ]
        if len(vertices) < 100:
            raise ValueError(f"invalid official mesh for {name}")
        array = np.asarray(vertices)
        result[name] = (array.max(axis=0) - array.min(axis=0)).tolist()
    return result


def build_catalog(root: Path, models_tar: Path) -> dict:
    groups = defaultdict(list)
    scene_objects = {}
    for episode in sorted(root.iterdir()):
        if not episode.is_dir():
            continue
        reference = episode / "gt/object_poses_world.npz"
        if not reference.is_file() or not (episode / "plan.h5").is_file():
            raise ValueError(f"incomplete scene: {episode}")
        with np.load(reference, allow_pickle=False) as poses:
            names = tuple(str(value) for value in poses["object_ids"])
            if len(names) != 4 or len(set(names)) != 4:
                raise ValueError(f"invalid object identities: {episode}")
        groups[names].append(episode.name)
        scene_objects[episode.name] = list(names)
    train, test = split_object_groups(groups)
    if set(train) & set(test) or len(train) != len(test):
        raise AssertionError("scene split is not balanced and disjoint")
    train_objects = {name for scene in train for name in scene_objects[scene]}
    test_objects = {name for scene in test for name in scene_objects[scene]}
    if train_objects & test_objects:
        raise AssertionError("object identity leaked across train/test")
    with tarfile.open(models_tar) as archive:
        dimensions = mesh_dimensions(archive, train_objects | test_objects)
    return {
        "schema": "viki_hocap_rl_object_disjoint_v1",
        "scene_count": len(scene_objects),
        "train_scenes": train,
        "test_scenes": test,
        "train_objects": sorted(train_objects),
        "test_objects": sorted(test_objects),
        "scene_objects": scene_objects,
        "mesh_aabb_extent_m": dimensions,
        "object_mass_kg": None,
        "mass_provenance": "not present in downloaded HO-Cap models/scene metadata; simulator mass must be declared as an assumption",
        "split_rule": "all instances of an object identity stay in one partition; 14/14 scenes; G10_4 pilot in train",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/datasets/hocap_eightview"))
    parser.add_argument("--models-tar", type=Path, default=Path("data/_hocap_assets/models.tar"))
    parser.add_argument("--output", type=Path, default=Path("experiments/hocap_rl_split.json"))
    arguments = parser.parse_args()
    manifest = build_catalog(arguments.root, arguments.models_tar)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"HO-Cap RL split: {len(manifest['train_scenes'])} train, {len(manifest['test_scenes'])} test; "
          f"{len(manifest['train_objects'])}/{len(manifest['test_objects'])} unseen object identities")
