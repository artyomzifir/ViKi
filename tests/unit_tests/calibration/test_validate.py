"""
Pre-record cloud-agreement gate — ``viki.calibration.validate`` (spec §6).
"""

from __future__ import annotations

import numpy as np

from viki.calibration import validate


def _blob(n=1200, seed=0):
    """A box-corner (three perpendicular plane patches) — like real scene
    geometry it fully constrains a rigid ICP, so a mis-registration shows up in
    the recovered translation."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.3, (n, 2))
    faces = [
        np.c_[np.zeros(n), u],
        np.c_[u[:, 0], np.zeros(n), u[:, 1]],
        np.c_[u, np.zeros(n)],
    ]
    return np.vstack(faces) + np.array([0.0, 0.0, 0.4])


def _shift(xyz, t, noise=0.001, seed=1):
    return xyz + np.asarray(t) + np.random.default_rng(seed).normal(0, noise, xyz.shape)


# ── verdict policy (spec §6), tested directly ───────────────────────────

def test_pair_verdict_bands():
    g, a = validate.GREEN, validate.AMBER
    green = {"nn_median_mm": 10, "icp_translation_mm": 12, "icp_rotation_deg": 1.0}
    amber = {"nn_median_mm": 25, "icp_translation_mm": 12, "icp_rotation_deg": 1.0}
    red_by_trans = {"nn_median_mm": 10, "icp_translation_mm": 80, "icp_rotation_deg": 1.0}
    red_by_rot = {"nn_median_mm": 10, "icp_translation_mm": 12, "icp_rotation_deg": 9.0}
    assert validate._pair_verdict(green, g, a) == "green"
    assert validate._pair_verdict(amber, g, a) == "amber"
    assert validate._pair_verdict(red_by_trans, g, a) == "red"
    assert validate._pair_verdict(red_by_rot, g, a) == "red"


# ── end to end over assembled clouds ───────────────────────────────────

def test_agreeing_cameras_are_green():
    a = _blob(seed=0)
    b = _shift(a, [0, 0, 0], noise=0.001, seed=2)
    r = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=None)
    assert r["verdict"] == "green"
    p = r["pairs"][0]
    assert p["icp_translation_mm"] < 5.0
    assert p["nn_median_mm"] < 8.0 and "note" in p  # sub-noise ⇒ flagged


def test_large_rigid_offset_is_not_green():
    a = _blob(seed=0)
    b = _shift(a, [0.06, 0.02, 0.0], noise=0.001, seed=3)
    r = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=None)
    assert r["verdict"] != "green"
    assert r["pairs"][0]["icp_translation_mm"] > 15.0  # the offset was seen


def test_too_few_points_in_box_is_red():
    a = _blob(seed=0)
    b = _blob(seed=5)
    r = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=[10, 11, 10, 11, 10, 11])
    assert r["verdict"] == "red"
    assert r["pairs"][0]["skipped"] is True


def test_three_cameras_verdict_is_the_worst_pair():
    a = _blob(seed=0)
    b = _shift(a, [0, 0, 0], noise=0.001, seed=6)
    c = _shift(a, [0.08, 0.05, 0.0], noise=0.001, seed=7)
    r = validate.pairwise_agreement({"k0": a, "k1": b, "k2": c}, aabb=None)
    assert len(r["pairs"]) == 3
    verdicts = {p["a"] + p["b"]: p["verdict"] for p in r["pairs"]}
    assert verdicts["k0k1"] == "green"
    assert r["verdict"] != "green"  # dragged down by the k*-k2 pairs
