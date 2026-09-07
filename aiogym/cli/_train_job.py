"""Per-run CLI logs and lifecycle, independent of the batch scheduler."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from aiogym.core.io import write_json


_TERMINAL = {"succeeded", "failed", "cancelled", "skipped", "interrupted"}


def _process_identity(pid):
    if os.name != "posix":
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "args="],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 1:
        return None  # Process exited before it could be inspected.
    result.check_returncode()
    return result.stdout.strip() or None


def read_training_status(path):
    """Read a status snapshot without mistaking a reused PID for a live job."""
    path = Path(path)
    if path.is_dir():
        path = path / "status.json"
    job = json.loads(path.read_text(encoding="utf-8"))
    if job["status"] == "running":
        identity = job.get("process_identity")
        if identity is None:
            job.update(status="unknown", error="process identity was not recorded or is unsupported")
        elif _process_identity(job["pid"]) != identity:
            job.update(status="interrupted", error="original training process is no longer running")
    elif job["status"] in {"pending", "starting"}:
        identity = job.get("scheduler_identity")
        if identity is not None and _process_identity(job["scheduler_pid"]) != identity:
            job.update(status="interrupted", error="scheduler exited before this task reported running")
    return job


def training_progress(path):
    """Enrich a lifecycle snapshot from existing curve records, without writing."""
    path = Path(path)
    if path.is_dir():
        path /= "status.json"
    job = read_training_status(path)
    now = time.time()
    updated = path.stat().st_mtime
    initial = job.get("initial_steps")
    actual = initial
    target = None if initial is None or "steps" not in job else initial + job["steps"]
    curve_path = path.parent / "training_curve.json"
    curve = None
    if curve_path.is_file():
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        actual = curve["actual_steps"]
        updated = max(updated, curve_path.stat().st_mtime)
    # Finished legacy runs can recover their budget from final metadata.
    metadata_path = path.parent / "metadata.json"
    if target is None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        initial = metadata["initial_steps"]
        target = initial + metadata["steps"]
        actual = metadata["actual_steps"]
        job.setdefault("scenario", metadata["environment"]["scenario"])
        if job.get("seed") is None:
            job["seed"] = metadata["seed"]
    started = job.get("started_at")
    finished = job.get("finished_at")
    # A dead process has no known end time; do not keep increasing its runtime.
    end = finished if finished is not None else now if job["status"] == "running" else updated
    elapsed = None if started is None else max(0.0, end - started)
    eta = None
    if (job["status"] == "running" and job.get("phase") == "training"
            and curve is not None and initial is not None and target is not None
            and started is not None):
        observed = curve_path.stat().st_mtime - started
        completed = actual - initial
        if observed >= 30 and completed >= 2 * curve["record_every"]:
            eta = max(0.0, target - actual) * observed / completed
    return {**job, "status_file": str(path.resolve()), "progress": {
        "actual_steps": actual, "target_steps": target,
        "elapsed_seconds": elapsed, "eta_seconds": eta,
        "updated_at": updated, "update_age_seconds": max(0.0, now - updated),
    }}


def print_training_status(jobs):
    if not jobs:
        print("No CLI training runs found. Start with: aiogym train heater sac")
        return

    def duration(seconds):
        if seconds is None:
            return "—"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"

    rows = []
    for index, job in enumerate(jobs, 1):
        progress = job["progress"]
        actual, target = progress["actual_steps"], progress["target_steps"]
        steps = "—" if actual is None else f"{actual:,}"
        if target is not None:
            steps += f"/{target:,}"
        rows.append((
            str(index), job.get("scenario", "—"), job.get("algorithm", "—"),
            "—" if job.get("seed") is None else str(job["seed"]),
            job["status"], job.get("phase", "—") if job["status"] == "running" else "—",
            steps, duration(progress["elapsed_seconds"]),
            "—" if progress["eta_seconds"] is None else "~" + duration(progress["eta_seconds"]),
            duration(progress["update_age_seconds"]) + " ago",
        ))
    headings = ("RUN", "SCENARIO", "ALGORITHM", "SEED", "STATE", "PHASE", "STEPS", "ELAPSED", "ETA", "UPDATED")
    widths = [max(len(row[i]) for row in [headings, *rows]) for i in range(len(headings))]
    for row in [headings, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
    for index, job in enumerate(jobs, 1):
        print(f"[{index}] {Path(job['status_file']).parent}")
        if job.get("error"):
            print(f"    {job['error']}")
    print("Steps are last recorded values. Details: <run>/train.log; JSON: aiogym status --json")


class _PhaseHandler(logging.Handler):
    """Keep CLI lifecycle state in sync with workflow phase events."""
    def __init__(self, job):
        super().__init__()
        self.job = job

    def emit(self, record):
        if hasattr(record, "phase"):
            self.job["phase"] = record.phase
            if hasattr(record, "initial_steps"):
                self.job["initial_steps"] = record.initial_steps
                self.job["seed"] = record.seed
            write_json(self.job["status_file"], self.job, overwrite=True)


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def isatty(self):
        return False


def run_training_job(job, work, *, log_output=True):
    """Record this process's outcome, including an exception or termination."""
    path = Path(job["output"])
    path.mkdir(parents=True, exist_ok=True)
    job = dict(job)
    job.update(
        schema_version="aiogym.training_job.v2", status="running", phase="setup",
        pid=os.getpid(), process_identity=_process_identity(os.getpid()),
        started_at=time.time(), log=str(path / "train.log"),
        status_file=str(path / "status.json"),
    )
    write_json(job["status_file"], job, overwrite=True)

    def cancel(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, cancel)
    code = 1
    try:
        with ExitStack() as stack:
            logger = logging.getLogger("aiogym.workflows.train")
            handler = _PhaseHandler(job)
            stack.callback(logger.setLevel, logger.level)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            stack.callback(logger.removeHandler, handler)
            if log_output:
                log = stack.enter_context((path / "train.log").open("a", encoding="utf-8"))
                stack.enter_context(redirect_stdout(_Tee(sys.stdout, log)))
                stack.enter_context(redirect_stderr(_Tee(sys.stderr, log)))
            try:
                print(f"Training {job['algorithm']} started (pid={job['pid']})", file=sys.stderr)
                result = work()
                code = 0
                print(f"Training {job['algorithm']} succeeded", file=sys.stderr)
                return result
            except KeyboardInterrupt:
                code = 130
                raise
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
                raise
            finally:
                if sys.exc_info()[0] is not None:
                    job["error"] = str(sys.exc_info()[1])
                    traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, previous)
        job.update(
            returncode=code, finished_at=time.time(),
            status="succeeded" if code == 0 else "cancelled" if code == 130 else "failed",
        )
        write_json(job["status_file"], job, overwrite=True)


def main():
    path, job_id = sys.argv[1:]
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    if job["job_id"] != job_id:
        raise ValueError("training job identity does not match its status file")
    from .workflows import main as workflow_main

    return run_training_job(
        job, lambda: workflow_main("train", job["command"][5:], _managed=True),
        log_output=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
