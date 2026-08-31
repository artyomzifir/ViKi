"""
viki.server.jobs
----------------
In-process background jobs for the offline stages.

Jobs run in **lanes**; each lane is one FIFO worker thread. Jobs in the same
lane run one at a time, jobs in different lanes run concurrently:

  * ``"main"`` (default) — perception / prepare / retarget / export / download.
    These share the GPU + MediaPipe + PINK, so they must not overlap.
  * ``"cloud"`` — point-cloud builds. Pure numpy/ctypes CPU work with no shared
    state beyond a read-only ``raw/``, so it runs alongside a ``main`` job (a
    perceive and its cloud for the same episode finish in parallel).

``queued=False`` jobs skip the lanes and run immediately in their own thread
(recording, which is interactive).

Not durable — jobs are lost on restart, fine for a single-user research tool.
Each job's ``fn`` is called as ``fn(report, log)``:
  * ``report(**progress)`` merges into ``job["progress"]`` (e.g. stage, camera,
    frame, total)
  * ``log(msg)`` appends to a bounded ring buffer
"""

from __future__ import annotations

import inspect
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_lanes: dict[str, "queue.Queue[str]"] = {}
_workers: dict[str, threading.Thread] = {}
_worker_lock = threading.Lock()

_LOG_CAP = 200
DEFAULT_LANE = "main"


def _ensure_worker(lane: str) -> "queue.Queue[str]":
    with _worker_lock:
        q = _lanes.get(lane)
        if q is None:
            q = _lanes[lane] = queue.Queue()
        w = _workers.get(lane)
        if w is None or not w.is_alive():
            w = threading.Thread(
                target=_run_worker, args=(q,), daemon=True, name=f"viki-jobs-{lane}"
            )
            _workers[lane] = w
            w.start()
        return q


def _adapt(fn: Callable) -> Callable[[Callable, Callable], Any]:
    """Accept a 0-arg legacy job fn or a 2-arg ``fn(report, log)`` one."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (ValueError, TypeError):
        n = 0
    return fn if n >= 2 else (lambda report, log: fn())


def _make_callbacks(job_id: str):
    def report(**progress):
        with _lock:
            j = _jobs.get(job_id)
            if j is not None:
                j["progress"] = {**j.get("progress", {}), **progress}

    def log(msg: str):
        with _lock:
            j = _jobs.get(job_id)
            if j is not None:
                j["log"].append(str(msg))

    return report, log


def _execute(job_id: str) -> None:
    with _lock:
        j = _jobs.get(job_id)
        if j is None or j["status"] == "cancelled":
            return
        j.update(status="running", started=time.time())
        fn = j.pop("_fn")
    report, log = _make_callbacks(job_id)
    try:
        result = _adapt(fn)(report, log)
        with _lock:
            _jobs[job_id].update(status="done", result=result, finished=time.time())
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _jobs[job_id].update(
                status="error", error=str(exc), trace=traceback.format_exc(),
                finished=time.time(),
            )


def _run_worker(q: "queue.Queue[str]") -> None:
    while True:
        job_id = q.get()
        try:
            _execute(job_id)
        finally:
            q.task_done()


def submit(
    kind: str,
    fn: Callable,
    *,
    episode: str | None = None,
    queued: bool = True,
    lane: str = DEFAULT_LANE,
) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "kind": kind, "episode": episode, "lane": lane,
            "status": "queued" if queued else "running",
            "progress": {}, "log": deque(maxlen=_LOG_CAP),
            "at": time.time(), "started": None, "finished": None,
            "_fn": fn,
        }
    if queued:
        _ensure_worker(lane).put(job_id)
    else:
        threading.Thread(
            target=_execute, args=(job_id,), daemon=True, name=f"viki-job-{kind}"
        ).start()
    return job_id


def cancel(job_id: str) -> bool:
    """Cancel a job that has not started yet. Running jobs cannot be interrupted."""
    with _lock:
        j = _jobs.get(job_id)
        if j is None:
            return False
        if j["status"] == "queued":
            j.update(status="cancelled", finished=time.time())
            j.pop("_fn", None)
            return True
    return False


def _queue_order() -> dict[str, list[str]]:
    """Queued job ids per lane, oldest first."""
    with _lock:
        queued = [
            (jid, j["lane"], j["at"])
            for jid, j in _jobs.items()
            if j["status"] == "queued"
        ]
    order: dict[str, list[str]] = {}
    for jid, lane, _ in sorted(queued, key=lambda t: t[2]):
        order.setdefault(lane, []).append(jid)
    return order


def _public(j: dict, order: dict[str, list[str]]) -> dict:
    out = {k: v for k, v in j.items() if k != "_fn"}
    out["log"] = list(j["log"])[-40:]
    lane_order = order.get(j["lane"], [])
    out["queue_pos"] = lane_order.index(j["id"]) + 1 if j["id"] in lane_order else None
    return out


def get(job_id: str) -> dict | None:
    order = _queue_order()
    with _lock:
        j = _jobs.get(job_id)
        return _public(j, order) if j is not None else None


def all_jobs() -> list[dict]:
    order = _queue_order()
    with _lock:
        js = list(_jobs.values())
    return sorted((_public(j, order) for j in js), key=lambda j: j["at"], reverse=True)
