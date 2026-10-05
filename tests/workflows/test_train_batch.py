from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import pytest

import aiogym
from aiogym.cli import _train_batch as batch
from aiogym.cli.main import main
from aiogym.cli.workflows import _parser


def test_training_plan_preserves_single_output_and_expands_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _parser("train").parse_args([
        "heater", "sac", "--output", str(tmp_path / "explicit"), "--resume-from", "model.zip",
    ])
    jobs = batch.prepare_training(args)
    assert jobs[0]["output"] == str(tmp_path / "explicit")
    assert jobs[0]["seed"] is None
    args = _parser("train").parse_args(["three_tank", "sac", "ppo", "--seeds", "0", "2"])
    jobs = batch.prepare_training(args)
    assert [(job["algorithm"], job["seed"]) for job in jobs] == [
        ("sac", 0), ("sac", 2), ("ppo", 0), ("ppo", 2),
    ]
    assert len({job["output"] for job in jobs}) == 4
    experiment = Path(jobs[0]["output"]).parent.name
    for job in jobs:
        assert Path(job["output"]) == (
            tmp_path / "runs/three-tank/training" / job["algorithm"]
            / experiment / f"seed-{job['seed']}"
        )
    assert not (tmp_path / "runs").exists()


def test_batch_preflight_routes_dataset_and_freezes_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    env = aiogym.make_env("heater", randomize=True)
    dataset = tmp_path / "dataset"
    aiogym.collect(env=env, output=dataset, policy="pid", episodes=1, max_steps=2)
    env.close()
    parameters = tmp_path / "parameters.json"
    parameters.write_text("{}", encoding="utf-8")
    captured = {}

    def run_jobs(jobs, *, workers):
        captured.update(jobs=jobs, workers=workers)
        return {"status": "completed"}

    monkeypatch.setattr(batch, "_run_jobs", run_jobs)
    assert main([
        "train", "heater", "sac", "ppo", "rlpd", "--seeds", "0", "1",
        "--dataset", str(dataset), "--parameters", str(parameters),
        "--steps", "20", "--no-evaluation", "--noise", "on", "--workers", "2",
    ]) == 0
    assert captured["workers"] == 2
    assert len(captured["jobs"]) == 6
    parameters.write_text('{"changed": true}', encoding="utf-8")
    for job in captured["jobs"]:
        command = job["command"]
        assert ("--dataset" in command) == (job["algorithm"] == "rlpd")
        assert command[command.index("--seed") + 1] == str(job["seed"])
        assert command[command.index("--noise") + 1] == "on"
        assert "--no-evaluation" in command
        frozen = Path(command[command.index("--parameters") + 1])
        assert frozen != parameters
        assert json.loads(frozen.read_text()) == {}
        output = Path(job["output"])
        assert frozen == output / "parameters.json"
        assert Path(job["log"]) == output / "train.log"
        assert Path(job["status_file"]) == output / "status.json"
    assert {path.name for path in (tmp_path / "runs/heater/training").iterdir()} == {
        "sac", "ppo", "rlpd",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_missing_rlpd_dataset_stops_before_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(["train", "heater", "sac", "rlpd"])
    assert error.value.code == 2
    assert "rlpd requires dataset" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("options", [
    ["--output", "shared"], ["--resume-from", "model.zip"],
    ["--algorithm-kwargs", "shared.json"], ["--workers", "0"],
])
def test_batch_rejects_ambiguous_or_invalid_options(tmp_path, monkeypatch, options):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(["train", "heater", "sac", "ppo", *options])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


def _jobs(tmp_path, scripts):
    worker = (
        "import json,sys; from pathlib import Path; "
        "from aiogym.cli._train_job import run_training_job; "
        "job=json.loads(Path(sys.argv[1]).read_text()); "
        "run_training_job(job, lambda: exec(job['command'][-1]), log_output=False)"
    )
    jobs = []
    for index, script in enumerate(scripts):
        output = tmp_path / f"job-{index}"
        output.mkdir()
        jobs.append({
            "algorithm": f"test-{index}", "seed": 0, "status": "pending",
            "output": str(output), "log": str(output / "train.log"),
            "status_file": str(output / "status.json"),
            "command": [sys.executable, "-u", "-c", script],
            "worker_command": [sys.executable, "-u", "-c", worker, str(output / "status.json")],
        })
    return jobs


def test_scheduler_overlaps_only_up_to_worker_limit(tmp_path):
    script = (
        "import json,os,time; "
        "print(json.dumps({'start':time.time(),'threads':os.environ['OMP_NUM_THREADS']})); "
        "time.sleep(0.3); print(json.dumps({'end':time.time()}))"
    )
    jobs = _jobs(tmp_path, [script] * 3)
    result = batch._run_jobs(jobs, workers=2)
    records = [[json.loads(line) for line in Path(job["log"]).read_text().splitlines() if line.startswith("{")]
               for job in jobs]
    assert max(records[0][0]["start"], records[1][0]["start"]) < min(
        records[0][1]["end"], records[1][1]["end"],
    )
    assert records[2][0]["start"] >= min(records[0][1]["end"], records[1][1]["end"])
    assert all(record[0]["threads"] == "1" for record in records)
    assert all(job["returncode"] == 0 for job in jobs)
    assert result["status"] == "completed"
    for job in jobs:
        recorded = json.loads(Path(job["status_file"]).read_text())
        assert recorded["status"] == "succeeded"
        assert recorded["returncode"] == 0
        assert recorded["pid"] == job["pid"]


def test_scheduler_failure_skips_queue_and_keeps_running_job(tmp_path):
    jobs = _jobs(tmp_path, ["raise SystemExit(7)", "import time; time.sleep(0.3)", "print('queued')"])
    result = batch._run_jobs(jobs, workers=2)
    assert result["status"] == "failed"
    assert [job["status"] for job in jobs] == ["failed", "succeeded", "skipped"]
    assert jobs[0]["returncode"] == 7
    assert not Path(jobs[2]["log"]).exists()
    for job in jobs:
        assert json.loads(Path(job["status_file"]).read_text())["status"] == job["status"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group interruption")
def test_scheduler_interrupt_reaps_children_and_preserves_logs(tmp_path):
    script = (
        f"import os,signal,time; time.sleep(0.2); os.kill({os.getpid()}, signal.SIGINT); "
        "time.sleep(30)"
    )
    jobs = _jobs(tmp_path, [script, "import time; print('started'); time.sleep(30)", "print('queued')"])
    previous = signal.getsignal(signal.SIGTERM)
    result = batch._run_jobs(jobs, workers=2)
    assert result["status"] == "cancelled"
    assert [job["status"] for job in jobs] == ["cancelled", "cancelled", "skipped"]
    assert signal.getsignal(signal.SIGTERM) == previous
    for job in jobs:
        assert json.loads(Path(job["status_file"]).read_text())["status"] == job["status"]
    for job in jobs[:2]:
        assert Path(job["log"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(job["pid"], 0)


def test_status_rejects_reused_pid(tmp_path, monkeypatch):
    from aiogym.cli import _train_job

    path = tmp_path / "status.json"
    path.write_text(json.dumps({"status": "running", "pid": os.getpid(), "process_identity": "original"}))
    monkeypatch.setattr(_train_job, "_process_identity", lambda pid: "different process")
    assert _train_job.read_training_status(path)["status"] == "interrupted"


def test_status_discovers_live_continuation_and_does_not_estimate_validation(tmp_path, monkeypatch, capsys):
    from aiogym.cli import _train_job

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_train_job, "_process_identity", lambda pid: "original")
    monkeypatch.setattr(_train_job.time, "time", lambda: 1100.0)
    output = tmp_path / "runs/heater/training/sac/continued/seed-0"
    output.mkdir(parents=True)
    path = output / "status.json"
    job = {"algorithm": "sac", "seed": 0, "status": "running", "phase": "training",
           "pid": 42, "process_identity": "original", "initial_steps": 500_000,
           "steps": 50_000, "started_at": 1000.0}
    path.write_text(json.dumps(job))
    curve = output / "training_curve.json"
    curve.write_text(json.dumps({"initial_steps": 0, "actual_steps": 501_000, "record_every": 500}))
    os.utime(path, (1000, 1000))
    os.utime(curve, (1060, 1060))
    assert main(["status", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]
    assert result["progress"] == {
        "actual_steps": 501_000, "target_steps": 550_000, "elapsed_seconds": 100,
        "eta_seconds": 2940, "updated_at": 1060, "update_age_seconds": 40,
    }
    job["phase"] = "validation"
    path.write_text(json.dumps(job))
    assert _train_job.training_progress(path)["progress"]["eta_seconds"] is None
    assert main(["status"]) == 0
    table = capsys.readouterr().out
    assert "validation" in table and "501,000/550,000" in table
    assert str(output) in table
    assert json.loads(path.read_text()) == job  # Inspection never rewrites lifecycle data.


def test_completion_commands_preserve_configuration_and_checkpoint_roles(tmp_path, capsys):
    import shlex
    from aiogym.cli.workflows import _print_training_result
    from aiogym.workflows._metadata import environment_metadata

    output = tmp_path / "run with spaces"
    output.mkdir()
    env = aiogym.make_env("heater", randomize=True, noise=True)
    try:
        metadata = {"environment": environment_metadata(env), "algorithm": "rlpd", "seed": 2,
                    "actual_steps": 500_000, "record_every": 500,
                    "evaluation": {"evaluate_every": 5000}, "dataset": {"path": "/data/pid demo"}}
    finally:
        env.close()
    _print_training_result(output, metadata)
    lines = capsys.readouterr().err.splitlines()
    compare, resume = [shlex.split(line.strip()) for line in lines if line.startswith("    aiogym")]
    assert compare[compare.index("--checkpoint") + 2] == str(output / "best/model.zip")
    assert compare[compare.index("--benchmark") + 1] == "tracking"
    args = _parser("train").parse_args(resume[2:])
    assert args.resume_from == output / "model.zip"
    assert args.noise == "on" and args.seed == 2 and args.steps == 50_000
    assert str(args.dataset) == "/data/pid demo"
    # Custom parameters must not produce a command targeting incompatible fixed cases.
    first_parameter = next(iter(metadata["environment"]["parameters"]))
    metadata["environment"]["parameters"][first_parameter] *= 1.01
    (output / "parameters.json").write_text(json.dumps(metadata["environment"]["parameters"]))
    _print_training_result(output, metadata)
    commands = [shlex.split(line.strip()) for line in capsys.readouterr().err.splitlines()
                if line.startswith("    aiogym")]
    assert "--benchmark" not in commands[0]
    assert commands[0][commands[0].index("--parameters") + 1] == str(output / "parameters.json")


@pytest.mark.rl
@pytest.mark.e2e
@pytest.mark.skipif(os.name != "posix", reason="POSIX scheduler termination")
def test_real_training_workers_finish_after_scheduler_exits(tmp_path):
    import subprocess
    import time

    pytest.importorskip("stable_baselines3")
    environment = dict(os.environ, PYTHONPATH=str(Path(aiogym.__file__).resolve().parent.parent))
    log = tmp_path / "scheduler.log"
    with log.open("w") as stream:
        process = subprocess.Popen([
            sys.executable, "-m", "aiogym.cli.main", "train", "heater", "sac", "ddpg",
            "--workers", "2", "--steps", "2", "--no-evaluation",
        ], cwd=tmp_path, env=environment, stdout=stream, stderr=stream)
        paths = []
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                paths = list((tmp_path / "runs").rglob("status.json"))
                states = [json.loads(path.read_text()) for path in paths]
                if len(states) == 2 and all(row["status"] == "running" for row in states):
                    break
                assert process.poll() is None, log.read_text()
                time.sleep(0.02)
            else:
                pytest.fail("workers did not start: " + log.read_text())
            process.kill()
            process.wait(timeout=5)
            while time.monotonic() < deadline:
                states = [json.loads(path.read_text()) for path in paths]
                if all(row["status"] in {"succeeded", "failed", "cancelled"} for row in states):
                    break
                time.sleep(0.05)
            assert [row["status"] for row in states] == ["succeeded", "succeeded"]
            for path in paths:
                metadata = json.loads((path.parent / "metadata.json").read_text())
                assert metadata["actual_steps"] == 2
                assert (path.parent / "model.zip").is_file()
                assert (path.parent / "train.log").stat().st_size > 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            for path in paths:
                row = json.loads(path.read_text())
                if row["status"] == "running":
                    try:
                        os.killpg(row["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
