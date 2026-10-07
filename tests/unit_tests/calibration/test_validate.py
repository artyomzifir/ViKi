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
    assert p["nn_median_mm"] < 8.0
    assert "note" not in p


def test_large_rigid_offset_is_not_green():
    a = _blob(seed=0)
    b = _shift(a, [0.06, 0.02, 0.0], noise=0.001, seed=3)
    r = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=None)
    assert r["verdict"] != "green"
    assert r["pairs"][0]["icp_translation_mm"] > 15.0  # the offset was seen


def test_too_few_points_in_box_is_unknown_not_red():
    """Unscorable is not the same as wrong — an empty box must not read as a
    broken calibration."""
    a = _blob(seed=0)
    b = _blob(seed=5)
    r = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=[10, 11, 10, 11, 10, 11])
    assert r["verdict"] == "unknown"
    assert r["pairs"][0]["skipped"] is True
    assert r["pairs"][0]["verdict"] == "unknown"


def test_three_cameras_verdict_is_the_worst_pair():
    a = _blob(seed=0)
    b = _shift(a, [0, 0, 0], noise=0.001, seed=6)
    c = _shift(a, [0.08, 0.05, 0.0], noise=0.001, seed=7)
    r = validate.pairwise_agreement({"k0": a, "k1": b, "k2": c}, aabb=None)
    assert len(r["pairs"]) == 3
    verdicts = {p["a"] + p["b"]: p["verdict"] for p in r["pairs"]}
    assert verdicts["k0k1"] == "green"
    assert r["verdict"] != "green"  # dragged down by the k*-k2 pairs


# ── shared field of view (the wide-baseline gate) ──────────────────────

def _frustum(T=None, fov_px=400, w=640, h=480):
    """A camera at ``T`` (camera→rig; default the rig origin looking down +Z)."""
    return {
        "T_ref_cam": np.eye(4) if T is None else T,
        "K": np.array([[fov_px, 0, w / 2], [0, fov_px, h / 2], [0, 0, 1.0]]),
        "width": w, "height": h,
    }


def _yaw(deg, t):
    """camera→rig for a camera rotated ``deg`` about +Y and placed at ``t``."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = t
    return T


def test_visible_mask_keeps_only_what_is_in_the_frustum():
    pts = np.array([
        [0.0, 0.0, 1.0],     # dead centre
        [0.0, 0.0, -1.0],    # behind the camera
        [5.0, 0.0, 1.0],     # far outside the field of view
    ])
    m = validate._visible_mask(pts, _frustum())
    assert list(m) == [True, False, False]


def test_a_surface_only_one_camera_sees_is_not_counted_against_the_pair():
    """The 2-diff failure in miniature: both cameras agree on the surface they
    share, but one also sees a patch the other cannot. Scored whole that patch
    is pure NN disagreement; masked to the shared view the pair is green."""
    seen_by_both = _blob(seed=0)                      # sits around z ≈ 0.4–0.7
    only_a = seen_by_both + np.array([3.0, 0.0, 0.0])  # way off to the side
    a = np.vstack([seen_by_both, only_a])
    b = _shift(seen_by_both, [0, 0, 0], noise=0.001, seed=2)

    # 640x480: at f=60 the 3 m-away patch still projects inside the image, at
    # f=300 it is far outside while the whole shared blob still fits.
    wide = _frustum(fov_px=60)           # sees both patches
    narrow = _frustum(fov_px=300)        # sees only the shared blob
    assert validate._visible_mask(a, narrow).sum() == len(seen_by_both)

    whole = validate.pairwise_agreement({"k0": a, "k1": b}, aabb=None)
    assert whole["verdict"] == "red"     # charged for what k1 cannot see

    masked = validate.pairwise_agreement(
        {"k0": a, "k1": b}, frusta={"k0": wide, "k1": narrow}, aabb=None)
    assert masked["verdict"] == "green"
    sv = masked["pairs"][0]["shared_view"]
    assert sv["a_points"] == len(seen_by_both) and sv["a_frac"] < 0.55


def test_the_mask_still_sees_a_real_offset():
    """Masking must not launder a genuine mis-registration into green."""
    a = _blob(seed=0)
    b = _shift(a, [0.06, 0.02, 0.0], noise=0.001, seed=3)
    f = {"k0": _frustum(fov_px=200), "k1": _frustum(fov_px=200)}
    r = validate.pairwise_agreement({"k0": a, "k1": b}, frusta=f, aabb=None)
    assert r["verdict"] != "green"
    assert r["pairs"][0]["icp_translation_mm"] > 15.0


def test_too_little_shared_view_is_unknown_not_red():
    a = _blob(seed=0)
    b = _shift(a, [0, 0, 0], noise=0.001, seed=2)
    # k1 is turned 150 deg away from the shared blob: it retains nothing of it.
    f = {"k0": _frustum(fov_px=200), "k1": _frustum(_yaw(150, [0, 0, 0]), fov_px=200)}
    r = validate.pairwise_agreement({"k0": a, "k1": b}, frusta=f, aabb=None)
    p = r["pairs"][0]
    assert r["verdict"] == "unknown" and p["skipped"] is True
    assert p["verdict"] == "unknown"
    assert "share" in p["reason"] or "shared" in p["reason"]


def test_min_overlap_is_tunable():
    """A pair that is scorable at the default threshold becomes 'unknown' once
    the caller demands more shared view."""
    seen_by_both = _blob(seed=0)
    a = np.vstack([seen_by_both, seen_by_both + np.array([3.0, 0.0, 0.0])])
    b = _shift(seen_by_both, [0, 0, 0], noise=0.001, seed=2)
    f = {"k0": _frustum(fov_px=60), "k1": _frustum(fov_px=300)}
    assert validate.pairwise_agreement(
        {"k0": a, "k1": b}, frusta=f, aabb=None)["verdict"] == "green"
    strict = validate.pairwise_agreement(
        {"k0": a, "k1": b}, frusta=f, aabb=None, min_overlap=0.9)
    assert strict["verdict"] == "unknown" and strict["pairs"][0]["skipped"] is True


def test_a_point_on_a_depth_hole_is_not_counted_as_disagreement():
    """The frustum is not enough: a camera can be pointed at something it never
    measured (stereo dropout, shadow, out of range). Those points must drop out
    of the comparison, not read as 40 mm of error."""
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    cam = _frustum()
    depth = np.full((480, 640), 1000.0)          # 1 m everywhere
    u = [int(round(400 * x / 1.0 + 320)) for x in (0.0, 0.1)]
    depth[:, u[1]] = 0                            # blank the second point's column
    cam["depth_mm"] = depth
    assert list(validate._visible_mask(pts, cam)) == [True, False]


def test_a_point_hidden_behind_a_nearer_surface_is_not_counted():
    pts = np.array([[0.0, 0.0, 2.0]])
    cam = _frustum()
    cam["depth_mm"] = np.full((480, 640), 1000.0)  # the camera sees 1 m, not 2 m
    assert not validate._visible_mask(pts, cam)[0]
    # a reading *behind* the point is real disagreement, so it stays in
    cam["depth_mm"] = np.full((480, 640), 3000.0)
    assert validate._visible_mask(pts, cam)[0]


def _depth_map(points, cam, dilate=3):
    """Render ``points`` into ``cam``'s depth image (mm, 0 = no reading).

    Splat, then dilate: the fixture clouds are sparse point samples of a
    surface, and the mask asks "did this camera get a reading on this pixel",
    which on a real sensor is a filled surface, not isolated dots.
    """
    import cv2

    T, K = cam["T_ref_cam"], cam["K"]
    p = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
    u = np.rint(p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.rint(p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"]) & (p[:, 2] > 0)
    d = np.zeros((cam["height"], cam["width"]), np.float32)
    d[v[ok], u[ok]] = (p[ok, 2] * 1000.0).astype(np.float32)
    return cv2.dilate(d, np.ones((dilate, dilate), np.uint8))


def test_a_dropout_region_no_longer_drags_the_pair_red():
    """One camera loses a contiguous patch of depth. The points the other camera
    still has there have no counterpart — with only a frustum test they score as
    error (this is the live 2-diff failure); with the depth lookup they drop out
    of the comparison."""
    blob = _blob(seed=0)
    a = _shift(blob, [0, 0, 0], noise=0.001, seed=2)
    b = blob[blob[:, 2] > 0.55]          # b keeps only the far third of the surface

    cam_a, cam_b = _frustum(fov_px=200), _frustum(fov_px=200)
    cam_a["depth_mm"] = _depth_map(a, cam_a)
    cam_b["depth_mm"] = _depth_map(b, cam_b)
    blind = {"k0": {**cam_a, "depth_mm": None}, "k1": {**cam_b, "depth_mm": None}}

    frustum_only = validate.pairwise_agreement({"k0": a, "k1": b}, frusta=blind, aabb=None)
    with_depth = validate.pairwise_agreement(
        {"k0": a, "k1": b}, frusta={"k0": cam_a, "k1": cam_b}, aabb=None)

    # The NN median is the metric the hole drove: 33.2 mm on the live rig. The
    # verdict is deliberately not asserted here — on a fixture where b is a
    # narrow strip, ICP still slides along it (the planar degeneracy noted in
    # the module docstring), which is a separate matter from the hole.
    assert frustum_only["verdict"] == "red"
    assert frustum_only["pairs"][0]["nn_median_mm"] > 20.0
    assert with_depth["pairs"][0]["nn_median_mm"] < 8.0
    assert with_depth["pairs"][0]["shared_view"]["a_frac"] < 0.8
