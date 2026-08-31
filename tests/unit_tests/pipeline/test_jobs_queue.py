"""viki.server.jobs — FIFO queue: one worker, order, cancel, progress/log."""

import time

from viki.server import jobs


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_fifo_one_at_a_time_with_progress_and_log():
    order: list[str] = []

    def make(tag):
        def fn(report, log):
            order.append(f"{tag}:start")
            report(stage="work", frame=1, total=2)
            log(f"{tag} running")
            time.sleep(0.15)
            report(stage="work", frame=2, total=2)
            order.append(f"{tag}:end")
            return tag
        return fn

    a = jobs.submit("t", make("A"), episode="epA")
    b = jobs.submit("t", make("B"), episode="epB")
    c = jobs.submit("t", make("C"), episode="epC")

    # while A runs, B and C are queued in order
    assert _wait(lambda: jobs.get(a)["status"] == "running")
    jb, jc = jobs.get(b), jobs.get(c)
    assert jb["status"] == "queued" and jc["status"] == "queued"
    assert jb["queue_pos"] == 1 and jc["queue_pos"] == 2

    assert _wait(lambda: jobs.get(c)["status"] == "done", timeout=8)
    # strictly serialised: no interleave of start/end
    assert order == ["A:start", "A:end", "B:start", "B:end", "C:start", "C:end"]

    ja = jobs.get(a)
    assert ja["status"] == "done" and ja["result"] == "A"
    assert ja["progress"] == {"stage": "work", "frame": 2, "total": 2}
    assert "A running" in ja["log"]


def test_cancel_queued_job():
    held = {"go": False}

    def blocker(report, log):
        while not held["go"]:
            time.sleep(0.02)

    def payload(report, log):
        payload.ran = True
    payload.ran = False

    x = jobs.submit("t", blocker)
    y = jobs.submit("t", payload)

    assert _wait(lambda: jobs.get(x)["status"] == "running")
    assert jobs.cancel(y) is True
    assert jobs.get(y)["status"] == "cancelled"
    assert jobs.cancel(x) is False  # already running

    held["go"] = True
    assert _wait(lambda: jobs.get(x)["status"] == "done")
    time.sleep(0.1)
    assert payload.ran is False  # cancelled job never executed


def test_legacy_zero_arg_fn_still_runs():
    out = {}

    def legacy():
        out["ran"] = True
        return 42

    j = jobs.submit("t", legacy)
    assert _wait(lambda: jobs.get(j)["status"] == "done")
    assert jobs.get(j)["result"] == 42 and out["ran"] is True
