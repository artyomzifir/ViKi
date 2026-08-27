"""
viki.server.jobs
----------------
Tiny in-process background-job registry shared by the offline-stage routes
(pipeline / replay / export). Not durable — jobs are lost on restart, which is
fine for a single-user research tool.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[[], Any]) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "running", "at": time.time()}

    def _run() -> None:
        try:
            result = fn()
            with _lock:
                _jobs[job_id].update(status="done", result=result, finished=time.time())
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _jobs[job_id].update(
                    status="error", error=str(exc), trace=traceback.format_exc(),
                    finished=time.time(),
                )

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def all_jobs() -> list[dict]:
    with _lock:
        return sorted((dict(j) for j in _jobs.values()), key=lambda j: j["at"], reverse=True)
