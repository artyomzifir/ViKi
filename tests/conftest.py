"""Shared pytest config. Lives here (not pyproject.toml) because the test
container bind-mounts ``tests/`` but not the project root."""

import pytest


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
