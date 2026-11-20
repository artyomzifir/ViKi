"""The batch must not promote an inaccurate IK optimum as a usable plan."""

import json

import h5py

from scripts.batch_cube_scene_retarget import plan_quality


def _plan(path, *, rmse_mm, p95_mm, critical_error_mm, status="converged"):
    with h5py.File(path, "w") as plan:
        plan.create_dataset("solver_status", data=status.encode())
        plan.create_dataset("position_error_m", data=[
            0.005, *([critical_error_mm / 1000.0] * 20), 0.005, 0.005,
        ])
        plan.create_dataset("metrics_json", data=json.dumps({
            "position_rmse_mm": rmse_mm,
            "position_p95_mm": p95_mm,
            "object_grasp": {"phase_bounds": [1, 5, 15, 21]},
        }).encode())


def test_batch_plan_quality_accepts_accurate_ik(tmp_path):
    path = tmp_path / "plan.h5"
    _plan(path, rmse_mm=7, p95_mm=17, critical_error_mm=6)
    assert plan_quality(path)["accepted"]


def test_batch_plan_quality_rejects_bad_ik_despite_convergence(tmp_path):
    path = tmp_path / "plan.h5"
    _plan(path, rmse_mm=122, p95_mm=303, critical_error_mm=130)
    assert not plan_quality(path)["accepted"]


def test_batch_plan_quality_does_not_reject_safe_carry_for_late_retreat_error(tmp_path):
    path = tmp_path / "plan.h5"
    _plan(path, rmse_mm=15, p95_mm=34, critical_error_mm=6)
    assert plan_quality(path)["accepted"]
