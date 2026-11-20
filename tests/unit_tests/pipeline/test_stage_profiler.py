"""Subprocess CPU accounting and failure provenance for the profiling entrypoint."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def profiler():
    path = Path(__file__).resolve().parents[3] / "scripts/profile_viki_stage.py"
    spec = importlib.util.spec_from_file_location("viki_stage_profiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiler_keeps_exit_status_and_cpu_usage(tmp_path, profiler):
    out = tmp_path / "profile.json"
    code = profiler.main([
        "--out", str(out), "--interval", "0.01", "--", sys.executable,
        "-c", "sum(i*i for i in range(200000)); raise SystemExit(7)",
    ])
    report = json.loads(out.read_text())
    assert code == report["exit_code"] == 7
    assert report["cpu_sec"] > 0
    assert report["wall_sec"] > 0
    assert report["sampled_peak_processes"] >= 1
    assert "not attributable" in report["gpu_scope"]


def test_profiler_preserves_report_and_does_not_launch_again(tmp_path, profiler):
    out = tmp_path / "profile.json"
    out.write_text("existing evidence")
    marker = tmp_path / "launched"
    with pytest.raises(FileExistsError):
        profiler.main(["--out", str(out), "--", sys.executable, "-c",
                       f"open({str(marker)!r}, 'w').close()"])
    assert out.read_text() == "existing evidence"
    assert not marker.exists()


def test_cpu_accounting_includes_waited_for_child(profiler):
    child = "import time; start=time.process_time()\nwhile time.process_time()-start<0.06: pass"
    parent = f"import subprocess,sys; subprocess.run([sys.executable, '-c', {child!r}], check=True)"
    report = profiler.profile([sys.executable, "-c", parent], 0.01, False)
    assert report["exit_code"] == 0
    assert report["cpu_sec"] >= 0.06


def test_missing_command_is_recorded(tmp_path, profiler):
    out = tmp_path / "profile.json"
    assert profiler.main(["--out", str(out), "--", "/missing/viki-command"]) == 127
    assert json.loads(out.read_text())["error"]
