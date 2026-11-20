from experiments.hocap_rl_catalog import split_object_groups


def test_split_is_balanced_and_object_disjoint():
    groups = {
        ("G01_1",): ["a", "b", "c"],
        ("G02_1",): ["d", "e"],
        ("G10_4",): ["f", "g"],
        ("G04_1",): ["h"],
    }
    train, test = split_object_groups(groups)
    assert len(train) == len(test) == 4
    assert {"f", "g"}.issubset(train)
    assert all(set(scenes).issubset(train) or set(scenes).issubset(test)
               for scenes in groups.values())
