"""Shared pytest config. Lives here (not pyproject.toml) because the test
container bind-mounts ``tests/`` but not the project root."""

import pytest


@pytest.fixture(autouse=True)
def _sandbox_data_dirs(tmp_path, monkeypatch):
    """Point the config-driven ``data/`` output paths at a per-test tmp dir so a
    test that forgets to override them can't litter the repo's ``data/`` (which
    is bind-mounted into the container, so the litter is also root-owned). Tests
    that set their own path still win — this only moves the default."""
    for key, sub in (
        ("DATASETS_DIR", "datasets"),
        ("EPISODES_DIR", "episodes"),
        ("SKELETON_RECS_DIR", "skeleton_recs"),
        ("SKELETON_SMOOTHED_DIR", "skeleton_smoothed"),
        ("SKELETON_DEPTH_BASE_DIR", "depth_bases"),
    ):
        monkeypatch.setattr(f"viki.config.{key}", str(tmp_path / sub), raising=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: needs network (git-clones a robot description) or is long; "
        "run with `--runslow`",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run @slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
