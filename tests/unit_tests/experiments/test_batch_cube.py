import numpy as np

from scripts.batch_cube_scene_retarget import matches_color, scene_quality


def test_prompt_color_preflight_rejects_swapped_cube():
    assert matches_color(np.array([0, 55, 127]), "blue")
    assert matches_color(np.array([115, 27, 36]), "red")
    assert matches_color(np.array([6, 30, 22]), "green")
    assert not matches_color(np.array([6, 30, 22]), "blue")


def test_scene_gate_requires_visible_moving_blue_cube(tmp_path):
    path = tmp_path / "object_models.npz"
    positions = np.zeros((20, 3, 3), dtype=float)
    positions[-1, 0, 0] = 0.4
    confidence = np.full((20, 3), 0.9)
    np.savez(path, object_ids=np.array([2, 3, 4]),
             object_labels=np.array(["manipulated_object", "other_object", "other_object"]),
             translation_world=positions, track_confidence=confidence)
    assert scene_quality(path)["accepted"]
    confidence[:4, 0] = 0
    np.savez(path, object_ids=np.array([2, 3, 4]),
             object_labels=np.array(["manipulated_object", "other_object", "other_object"]),
             translation_world=positions, track_confidence=confidence)
    assert not scene_quality(path)["accepted"]
