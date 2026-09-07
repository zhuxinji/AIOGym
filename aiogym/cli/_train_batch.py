"""Plan CLI training runs and supervise independent training processes."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ._train_job import _TERMINAL, _process_identity, read_training_status

from aiogym.core.io import write_json
from aiogym.rl.algorithms import get_algorithm, resolve_algorithm_kwargs
from aiogym.rl.datasets import load_training_dataset
from aiogym.workflows.train import (
    _dataset_inputs,
    _nonnegative_integer,
    _positive_integer,
)


def prepare_training(args):
    algorithms = args.algorithm
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("training algorithms must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    for seed in seeds:
        if seed is not None:
            _nonnegative_integer("seed", seed)
    for name in ("steps", "record_every", "workers"):
        _positive_integer(name, getattr(args, name))
    if args.evaluate_every is not None:
        _positive_integer("evaluate_every", args.evaluate_every)
    multiple = len(algorithms) * len(seeds) > 1
    if multiple and (args.output is not None or args.resume_from is not None):
        raise ValueError("--output and --resume-from require one algorithm and one seed")
    if len(algorithms) > 1 and args.algorithm_kwargs is not None:
        raise ValueError("--algorithm-kwargs requires one algorithm; multiple seeds are allowed")
    if args.resume_from is None:
        seeds = [0 if seed is None else seed for seed in seeds]
    path_seed = seeds[0]
    if args.output is None and args.resume_from is not None and path_seed is None:
        from aiogym.workflows._checkpoint import _read_manifest, _validated_training_state

        with zipfile.ZipFile(args.resume_from) as archive:
            training = _read_manifest(archive)["policy"]["training"]
        path_seed = _validated_training_state(training)["seed"]
    experiment = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    root = Path("runs") / args.scenario.replace("_", "-") / "training"
    jobs = [
        {
            "scenario": args.scenario,
            "algorithm": algorithm,
            "seed": seed,
            "steps": args.steps,
            "output": str((args.output if args.output is not None else
                root / algorithm / experiment / f"seed-{path_seed if seed is None else seed}"
            ).resolve()),
            "status": "pending",
        }
        for algorithm in algorithms for seed in seeds
    ]
    return jobs


def run_training_batch(args, jobs, *, env, parameters, algorithm_kwargs):
    datasets = {}
    cloning = {}
    for algorithm in args.algorithm:
        backend = get_algorithm(algorithm)
        epochs = args.behavior_cloning_epochs if backend.behavior_cloning is not None else None
        dataset = args.dataset if backend.requires_dataset or epochs is not None else None
        datasets[algorithm], cloning[algorithm] = _dataset_inputs(
            backend=backend, dataset=dataset, epochs=epochs,
            batch_size=args.behavior_cloning_batch_size,
            learning_rate=args.behavior_cloning_learning_rate,
        )
        resolve_algorithm_kwargs(backend, steps=args.steps, values=algorithm_kwargs or {})
    if args.behavior_cloning_epochs is not None and not any(cloning.values()):
        raise ValueError("selected algorithms do not support behavior cloning")
    if args.dataset is not None:
        if not any(path is not None for path in datasets.values()):
            raise ValueError("--dataset requires an algorithm that consumes it or --behavior-cloning-epochs")
        load_training_dataset(args.dataset, env=env)

    common = ["--steps", str(args.steps), "--record-every", str(args.record_every),
              "--randomize" if args.randomize else "--no-randomize",
              "--boundary-probability", str(args.boundary_probability)]
    for name in ("disturbance", "noise", "delay", "fault"):
        common.extend([f"--{name}", getattr(args, name)])
    if args.reward is not None:
        common.extend(["--reward", args.reward])
    if args.evaluate_every is None:
        common.append("--no-evaluation")
    else:
        common.extend(["--evaluate-every", str(args.evaluate_every)])
    for job in jobs:
        algorithm = job["algorithm"]
        output = Path(job["output"])
        output.mkdir(parents=True, exist_ok=False)
        job["log"] = str(output / "train.log")
        job["status_file"] = str(output / "status.json")
        job["command"] = [
            sys.executable, "-u", "-m", "aiogym.cli.main", "train", args.scenario,
            algorithm, "--seed", str(job["seed"]), "--output", str(output), *common,
        ]
        # Freeze inputs in each run before dispatch, including queued tasks.
        for name, value in (("parameters", parameters), ("algorithm-kwargs", algorithm_kwargs)):
            if value is not None:
                path = write_json(output / f"{name}.json", value)
                job["command"].extend([f"--{name}", str(path)])
        if datasets[algorithm] is not None:
            job["command"].extend(["--dataset", str(datasets[algorithm].resolve())])
        if cloning[algorithm] is not None:
            for name in ("epochs", "batch_size", "learning_rate"):
                job["command"].extend([
                    "--behavior-cloning-" + name.replace("_", "-"),
                    str(cloning[algorithm][name]),
                ])
        job["job_id"] = uuid4().hex
        job["worker_command"] = [
            sys.executable, "-u", "-m", "aiogym.cli._train_job",
            job["status_file"], job["job_id"],
        ]
        print(f"Training {algorithm} seed {job['seed']}: {output} (create new run directory)", file=sys.stderr)
    return _run_jobs(jobs, workers=args.workers)


def _signal_process(process, *, kill=False):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        elif kill:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass  # The child may exit between poll() and signalling its process group.


def _run_jobs(jobs, *, workers):
    summary = {
        "schema_version": "aiogym.training_batch.v2",
        "status": "running", "workers": workers, "threads_per_worker": 1,
        "jobs": jobs,
    }
    child_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    pending = deque(jobs)
    running = []
    failed = False
    scheduler_identity = _process_identity(os.getpid())

    def save_status(job, *, overwrite=True):
        write_json(job["status_file"], {
            "schema_version": "aiogym.training_job.v2",
            "workers": workers, "threads_per_worker": 1,
            "scheduler_pid": os.getpid(), "scheduler_identity": scheduler_identity, **job,
        }, overwrite=overwrite)

    def cancel(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, cancel)
    try:
        for job in jobs:
            save_status(job, overwrite=False)
        while pending or running:
            for process, log, job in running[:]:
                code = process.poll()
                if code is None:
                    continue
                recorded = read_training_status(job["status_file"])
                if recorded["status"] in _TERMINAL:
                    for field in ("status", "returncode", "finished_at", "started_at", "pid", "process_identity", "error", "phase", "initial_steps"):
                        if field in recorded:
                            job[field] = recorded[field]
                else:
                    job.update(returncode=code, finished_at=time.time(),
                               status="succeeded" if code == 0 else "failed")
                job["returncode"] = code
                log.close()
                running.remove((process, log, job))
                failed = failed or code != 0
                print(f"{job['algorithm']} seed {job['seed']}: {job['status']} ({job['log']})", file=sys.stderr)
                save_status(job)
            if failed:
                for job in pending:
                    job["status"] = "skipped"
                    save_status(job)
                pending.clear()
            while pending and len(running) < workers:
                job = pending.popleft()
                log = Path(job["log"]).open("x", encoding="utf-8")
                job.update(status="starting", started_at=time.time())
                save_status(job)
                try:
                    process = subprocess.Popen(
                        job.get("worker_command", job["command"]), stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, env=child_env,
                        start_new_session=os.name == "posix",
                    )
                except OSError as error:
                    log.close()
                    job.update(status="failed", error=str(error), finished_at=time.time())
                    save_status(job)
                    raise
                running.append((process, log, job))
                job.update(status="running", pid=process.pid)
                # A managed worker owns all subsequent on-disk running/final states.
                if "worker_command" not in job:
                    save_status(job)
            if running:
                time.sleep(0.1)
        summary["status"] = "failed" if failed else "completed"
    except KeyboardInterrupt:
        summary["status"] = "cancelled"
    finally:
        # Signal all children first, then give the group a single bounded grace period.
        for process, _log, _job in running:
            _signal_process(process)
        deadline = time.monotonic() + 3.0
        for process, log, job in running:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _signal_process(process, kill=True)
                process.wait()
            log.close()
            job.update(status="cancelled", returncode=process.returncode, finished_at=time.time())
        for job in pending:
            job["status"] = "skipped"
        if summary["status"] == "running":
            summary["status"] = "failed"
        signal.signal(signal.SIGTERM, previous_sigterm)
        for job in jobs:
            save_status(job)
    return summary
