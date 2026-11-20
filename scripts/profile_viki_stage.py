"""Profile one subprocess stage on Linux without extra Python dependencies.

Run inside Docker; see docs/2026-11-20-compute-and-parallelism-audit.md.
CPU usage belongs to the target and its waited-for descendants. GPU samples
describe the entire visible device and are explicitly not PID attribution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def _process_tree(root: int) -> dict:
    pending = [root]
    seen: set[int] = set()
    rss = threads = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        base = Path(f"/proc/{pid}")
        try:
            fields = dict(
                line.split(":", 1) for line in (base / "status").read_text().splitlines()
                if ":" in line
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        seen.add(pid)
        rss += int(fields.get("VmRSS", "0 kB").split()[0])
        threads += int(fields.get("Threads", "0"))
        for task in (base / "task").glob("*"):
            try:
                pending.extend(int(value) for value in (task / "children").read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                pass
    return {"processes": len(seen), "threads": threads, "sum_rss_kib": rss}


def _gpu_sample() -> list[dict]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    return [dict(zip(
        ("device", "utilization_percent", "memory_used_mib", "memory_total_mib"),
        (part.strip() for part in row.split(",")),
    )) for row in output.stdout.splitlines() if row.strip()]


def profile(command: list[str], interval: float, gpu: bool) -> dict:
    started = time.perf_counter()
    samples = []
    gpu_error = None
    process = subprocess.Popen(command, start_new_session=True)
    # wait4 isolates target rusage from nvidia-smi subprocesses of the profiler.
    # Linux includes descendants whose usage the target collected before exit.
    try:
        while True:
            sample = {"elapsed_sec": time.perf_counter() - started,
                      **_process_tree(process.pid)}
            if gpu:
                try:
                    sample["gpu_device_totals"] = _gpu_sample()
                except (OSError, subprocess.SubprocessError) as exc:
                    gpu_error = str(exc)
                    gpu = False
            samples.append(sample)
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(interval)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()
        raise
    wall = time.perf_counter() - started
    cpu = usage.ru_utime + usage.ru_stime
    return {
        "schema": "viki_process_profile_v1", "command": command,
        "exit_code": process.returncode, "wall_sec": wall,
        "cpu_sec": cpu, "cpu_user_sec": usage.ru_utime,
        "cpu_system_sec": usage.ru_stime, "mean_cpu_cores": cpu / wall,
        "max_individual_process_rss_kib": usage.ru_maxrss,
        "sampled_peak_sum_rss_kib": max(s["sum_rss_kib"] for s in samples),
        "sampled_peak_processes": max(s["processes"] for s in samples),
        "sampled_peak_threads": max(s["threads"] for s in samples),
        "gpu_scope": "whole visible device; not attributable to target process",
        "gpu_error": gpu_error,
        "native_thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )},
        "samples": samples,
        "limitations": [
            "wall_sec includes monitoring overhead and exit detection latency",
            "CPU covers target and descendants reaped before target exit",
            "sum RSS counts shared pages more than once; samples can miss peaks",
            "GPU device totals include other workloads, without PID attribution",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0.01 <= args.interval <= 60:
        parser.error("provide a command and an interval between 0.01 and 60 seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before starting work; existing evidence is never replaced.
    with args.out.open("x") as stream:
        try:
            report = profile(command, args.interval, args.gpu)
        except OSError as exc:
            report = {"schema": "viki_process_profile_v1", "command": command,
                      "exit_code": 127, "error": str(exc)}
        json.dump(report, stream, indent=2)
        stream.write("\n")
    code = report["exit_code"]
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
