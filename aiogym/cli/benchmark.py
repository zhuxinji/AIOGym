#!/usr/bin/env python3
"""Unified single-task, scenario-task, and named-suite benchmark CLI."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

from aiogym._internal.config import parse_seed_list
from aiogym._internal.paths import run_path, timestamped_artifact_path
from aiogym._internal.vocabulary import OBJECTIVE_NAMES
from aiogym.catalog import list_scenarios, list_suites, list_tasks
from aiogym.evaluation import (
    BenchmarkCase,
    BenchmarkProtocol,
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    build_evaluation_report,
    resolve_protocol,
)
from aiogym.evaluation.artifact import finalize_benchmark_artifacts
from aiogym.evaluation.execution import execute_benchmark_case
from aiogym.evaluation import suite as suite_pipeline


CONFIG_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "suites"
PRESET_DIR = CONFIG_DIR / "presets"


def _sync_suite_paths():
    suite_pipeline.CONFIG_DIR = CONFIG_DIR
    suite_pipeline.PRESET_DIR = PRESET_DIR


def load_suite(name_or_path: str):
    _sync_suite_paths()
    return suite_pipeline.load_suite(name_or_path)


def _scenario_suite(scenario: str) -> dict:
    task_names = [
        task_id.split("/", 1)[1] for task_id in list_tasks(scenario)
    ]
    cases = [
        {"scenario": scenario, "task": task_name}
        for task_name in task_names
    ]
    if not cases:
        cases = [{"scenario": scenario, "objective": "tracking"}]
    suite = {
        "name": scenario,
        "description": (
            f"All registered tasks for the {scenario} scenario."
            if task_names else
            f"Default tracking benchmark for the {scenario} scenario."
        ),
        "scenarios": [scenario],
        "objectives": [],
        "controllers": ["pid", "mpc"],
        "action_mode": "actuator",
        "cases": cases,
    }
    if not task_names:
        suite.update({"episode_steps": 80, "control_dt": 0.5})
    return suite


def load_target(name_or_path: str):
    if name_or_path in list_scenarios() and name_or_path not in list_suites():
        return _scenario_suite(name_or_path)
    return load_suite(name_or_path)


def build_cases(args):
    if getattr(args, "task", None) is not None:
        return _build_single_task_cases(args)
    return suite_pipeline.build_cases(args, load_suite_fn=load_target)


def parse_controllers(raw: str) -> list[str]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


def parse_setpoint_vector(raw: str, option: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",")]
    except ValueError as ex:
        raise ValueError(f"{option} must be a comma-separated numeric vector") from ex
    if not values or any(part.strip() == "" for part in raw.split(",")):
        raise ValueError(f"{option} must be a comma-separated numeric vector")
    return values


def parse_setpoint_step(raw: str) -> dict:
    step_text, separator, values_text = raw.partition(":")
    if not separator:
        raise ValueError("--setpoint-step must use STEP:VALUE1,VALUE2 syntax")
    try:
        step = int(step_text)
    except ValueError as ex:
        raise ValueError("--setpoint-step STEP must be a non-negative integer") from ex
    if step < 0:
        raise ValueError("--setpoint-step STEP must be a non-negative integer")
    return {
        "at_step": step,
        "values": parse_setpoint_vector(values_text, "--setpoint-step"),
    }


def controller_specs(args, baseline_protocol: BenchmarkProtocol):
    specs = []
    controller_names = parse_controllers(args.controllers or "pid,mpc")
    seeds = parse_seed_list(args.seed_list, args.seed, args.episodes)
    for name in controller_names:
        if name in {"sb3", "onnx"}:
            path = args.sb3_path if name == "sb3" else args.onnx_path
            action_mode = (
                (args.sb3_action_mode or "setpoint")
                if name == "sb3" else (args.onnx_action_mode or "setpoint")
            )
            if not path:
                raise ValueError(f"controller '{name}' requires --{name}-path")
            policy_protocol = replace(baseline_protocol, action_mode=action_mode)
            config = {"path": path, "action_mode": action_mode}
            if name == "sb3":
                config["algo"] = args.sb3_algo
            specs.append({
                "name": name,
                "protocol": policy_protocol,
                "seed_list": seeds,
                "config": config,
            })
            continue
        profile = (
            args.oracle_profile or args.controller_profile
            if name == "oracle" else args.controller_profile
        )
        config = {"profile": profile} if profile else {}
        if name == "mpc" and args.mpc_horizon is not None:
            config["P"] = args.mpc_horizon
        specs.append({
            "name": name,
            "protocol": baseline_protocol,
            "seed_list": seeds,
            "config": config,
        })
    return specs


def _build_single_task_cases(args):
    protocol_config = {
        "action_mode": "actuator",
        "task": args.task,
        "tracking_q_y": args.tracking_q_y,
        "tracking_r_move": args.tracking_r_move,
        "randomize_setpoints": args.randomize_setpoints,
        "disturbance_obs": args.disturbance_obs,
        "previous_action_obs": args.previous_action_obs,
        "normalize_observations": args.normalize_observations,
        "tracking_error_obs": args.tracking_error_obs,
        "initial_setpoint": args.parsed_initial_setpoint,
        "setpoint_schedule": args.parsed_setpoint_schedule,
    }
    if args.episode_steps is not None:
        protocol_config["episode_steps"] = args.episode_steps
    if args.control_dt is not None:
        protocol_config["control_dt"] = args.control_dt
    baseline_protocol = resolve_protocol(
        args.suite,
        args.objective,
        protocol_config,
    )
    specs = controller_specs(args, baseline_protocol)
    cases = []
    for spec in specs:
        protocol = spec["protocol"]
        task_meta = protocol.metadata()["task_identity"]
        name = (
            f"{protocol.objective}:{protocol.scenario}:"
            f"{task_meta['name']}:{spec['name']}"
        )
        case_spec = BenchmarkCase.from_protocol(
            protocol,
            controller=spec["name"],
            seeds=spec["seed_list"],
            controller_config=spec["config"],
            case_id=name,
        )
        cases.append({
            "name": name,
            "scenario": protocol.scenario,
            "task": task_meta["name"],
            "task_status": task_meta["status"],
            "task_profile_hash": task_meta["profile_hash"],
            "objective": protocol.objective,
            "objective_source": protocol.objective_source,
            "controller": spec["name"],
            "action_mode": protocol.action_mode,
            "controller_config": dict(case_spec.controller_config),
            "protocol": protocol,
            "case_spec": case_spec,
            "seeds": list(spec["seed_list"]),
        })
    suite = {
        "name": args.suite,
        "description": f"Single task {args.suite}/{args.task}.",
        "scenarios": [args.suite],
        "objectives": [baseline_protocol.objective],
        "controllers": [spec["name"] for spec in specs],
        "action_mode": baseline_protocol.action_mode,
        "task": args.task,
    }
    return suite, cases


def run_case(
    suite_case: dict,
    include_tracebacks: bool,
    *,
    save_rollout: bool | None = None,
    rollout_steps: int | None = None,
):
    if save_rollout is None:
        save_rollout = suite_case["objective"] == "tracking"
    return execute_benchmark_case(
        suite_case["case_spec"],
        include_episodes=True,
        save_rollout=save_rollout,
        rollout_steps=rollout_steps,
        suite_case=suite_case["name"],
        include_tracebacks=include_tracebacks,
    )


def print_case(row: dict):
    status = row["execution_status"].upper()
    prefix = (
        f"{status:7s} {row['objective']:9s} {row['scenario']:10s} "
        f"{row.get('task', 'default'):28s} {row['controller']:14s}"
    )
    if row["execution_status"] not in ("passed", "degraded"):
        print(f"{prefix} {row.get('message', '')}")
        return
    metric = row["metric"]
    value = row.get(metric, 0.0)
    std = row.get(f"{metric}_std", 0.0)
    value_text = _format_metric_value(value)
    std_text = _format_metric_value(std)
    step_ms = float(row.get("runtime_seconds_per_step") or 0.0) * 1000.0
    print(
        f"{prefix} {metric}={value_text:>9s} +/- {std_text} "
        f"score={row['normalized_score']:8.2f} profit={row['profit']:9.2f} "
        f"track={row['track']:8.2f} safety={row.get('constraint_violation_count', 0.0):6.1f} "
        f"fallback={row.get('controller_fallback_count', 0):3} "
        f"step={step_ms:7.2f}ms"
    )


def _format_metric_value(value) -> str:
    number = float(value or 0.0)
    if number != 0.0 and abs(number) < 0.01:
        return f"{number:.3e}"
    return f"{number:.2f}"


def artifact_dir_for(
    args,
    suite: dict,
    cases: list[dict],
    run_id: str | None = None,
) -> str:
    if args.task is None:
        return suite_pipeline.artifact_dir_for(
            suite["name"],
            args.artifact_dir,
            overwrite=args.overwrite,
            run_id=run_id,
        )
    if args.artifact_dir is not None and not str(args.artifact_dir).strip():
        raise ValueError("--artifact-dir must not be empty")
    base = (
        Path(args.artifact_dir)
        if args.artifact_dir is not None
        else run_path(cases[0]["scenario"], cases[0]["task"])
    )
    if args.overwrite:
        return str(base)
    return str(timestamped_artifact_path(base, run_id))


def _command_config(args, artifact_dir: str, cases: list[dict]) -> dict:
    return {
        "target": args.suite,
        "scenario": cases[0]["scenario"] if args.task is not None else None,
        "task": args.task,
        "objective": args.objective,
        "objectives": args.objectives,
        "controllers": list(dict.fromkeys(
            case["controller"] for case in cases
        )),
        "controller_profile": args.controller_profile,
        "oracle_profile": args.oracle_profile,
        "mpc_horizon": args.mpc_horizon,
        "episodes": args.episodes,
        "episode_steps": args.episode_steps,
        "initial_setpoint": args.initial_setpoint,
        "setpoint_schedule": args.setpoint_step,
        "clear_setpoint_schedule": args.clear_setpoint_schedule,
        "seed": args.seed,
        "seed_list": args.seed_list,
        "control_dt": args.control_dt,
        "tracking_q_y": args.tracking_q_y,
        "tracking_r_move": args.tracking_r_move,
        "randomize_setpoints": args.randomize_setpoints,
        "disturbance_obs": args.disturbance_obs,
        "previous_action_obs": args.previous_action_obs,
        "normalize_observations": args.normalize_observations,
        "tracking_error_obs": args.tracking_error_obs,
        "sb3_path": args.sb3_path,
        "sb3_algo": args.sb3_algo,
        "sb3_action_mode": args.sb3_action_mode,
        "onnx_path": args.onnx_path,
        "onnx_action_mode": args.onnx_action_mode,
        "save_rollouts": bool(
            args.save_rollouts
            or args.plot
            or any(case["objective"] == "tracking" for case in cases)
        ),
        "plot": args.plot,
        "rollout_steps": args.rollout_steps,
        "overwrite": args.overwrite,
        "output_dir": artifact_dir,
    }


def _single_task_payload_fields(
    args,
    cases: list[dict],
    configs: list[dict],
    artifact_dir: str,
) -> dict:
    first = cases[0]
    protocol = first["protocol"]
    return {
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "benchmark": "public_benchmark",
        "run_dir": artifact_dir,
        "scenario": first["scenario"],
        "task": first["task"],
        "task_status": first["task_status"],
        "task_profile_hash": first["task_profile_hash"],
        "objective": first["objective"],
        "objective_source": first["objective_source"],
        "controllers": list(dict.fromkeys(
            case["controller"] for case in cases
        )),
        "protocol": protocol.metadata(),
        "config": _command_config(args, artifact_dir, cases),
        "benchmark_config": {
            "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
            "scenario": first["scenario"],
            "task": first["task"],
            "task_status": first["task_status"],
            "task_profile_hash": first["task_profile_hash"],
            "objective": first["objective"],
            "objective_source": first["objective_source"],
            "controllers": list(dict.fromkeys(
                case["controller"] for case in cases
            )),
            "protocol": protocol.metadata(),
            "cases": configs,
        },
    }


def main(argv=None, prog=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run all tasks for a scenario, one scenario task, or a named "
            "benchmark suite."
        ),
    )
    ap.add_argument(
        "suite",
        metavar="TARGET",
        help=(
            "scenario, built-in suite, or JSON path; "
            f"scenarios: {', '.join(list_scenarios())}; "
            f"suites: {', '.join(list_suites())}"
        ),
    )
    ap.add_argument(
        "task",
        nargs="?",
        metavar="TASK",
        help="optional task name when TARGET is a scenario",
    )
    ap.add_argument("--scenarios", default=None, help="comma-separated override")
    ap.add_argument("--objectives", default=None, help="comma-separated override")
    ap.add_argument(
        "--objective",
        default=None,
        choices=OBJECTIVE_NAMES,
        help="single objective override; otherwise tasks use their defaults",
    )
    ap.add_argument("--controllers", default=None, help="comma-separated override")
    ap.add_argument(
        "--controller-profile",
        default=None,
        help="controller tuning profile for PID/MPC baselines",
    )
    ap.add_argument(
        "--oracle-profile",
        default=None,
        help="Oracle-specific tuning profile",
    )
    ap.add_argument(
        "--mpc-horizon",
        type=int,
        default=None,
        help="override MPC prediction horizon P",
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument(
        "--initial-setpoint",
        default=None,
        help="override the initial controlled-output setpoint as VALUE1,VALUE2",
    )
    ap.add_argument(
        "--setpoint-step",
        action="append",
        default=None,
        metavar="STEP:VALUE1,VALUE2",
        help="replace the task schedule with an absolute setpoint event; repeatable",
    )
    ap.add_argument(
        "--clear-setpoint-schedule",
        action="store_true",
        help="remove the task's default setpoint events",
    )
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument(
        "--tracking-q-y",
        type=float,
        default=None,
        help="override scalar tracking weight Q; task-owned when omitted",
    )
    ap.add_argument(
        "--tracking-r-move",
        type=float,
        default=None,
        help="override move weight R; task-owned when omitted",
    )
    ap.add_argument(
        "--randomize-setpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="sample one reproducible bounded SP or task SP schedule per episode",
    )
    ap.add_argument(
        "--disturbance-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include current disturbances in observations; task-owned when omitted",
    )
    ap.add_argument(
        "--previous-action-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the previous applied action; task-owned when omitted",
    )
    ap.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="scale observations with fixed physical bounds; task-owned when omitted",
    )
    ap.add_argument(
        "--tracking-error-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace setpoints with normalized tracking errors; task-owned when omitted",
    )
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument(
        "--sb3-action-mode",
        default=None,
        choices=["actuator", "setpoint"],
    )
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument(
        "--onnx-action-mode",
        default=None,
        choices=["actuator", "setpoint"],
    )
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fail-on-degraded", action="store_true",
                    help="exit non-zero when any controller reports fallback/degraded diagnostics")
    ap.add_argument("--artifact-dir", default=None,
                    help="base artifact directory; defaults to runs/<suite>_suite")
    ap.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace managed artifacts in the base directory; use --no-overwrite for a timestamped sibling",
    )
    ap.add_argument(
        "--save-rollouts",
        action="store_true",
        help="save rollout traces; tracking benchmarks do this automatically",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="save rollout traces needed for non-tracking rollout plots",
    )
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--tracebacks", action="store_true")
    args = ap.parse_args(raw_args)
    if args.task is not None and args.suite not in list_scenarios():
        ap.error("TASK can only be used when TARGET is a registered scenario")
    if args.objective is not None and args.objectives is not None:
        ap.error("--objective and --objectives cannot be used together")
    if args.task is not None and args.objectives is not None:
        ap.error("single-task benchmarks use --objective, not --objectives")
    if args.task is None and args.objective is not None:
        args.objectives = args.objective
    if args.task is not None and args.scenarios is not None:
        ap.error("single-task benchmarks do not accept --scenarios")
    try:
        args.parsed_initial_setpoint = (
            parse_setpoint_vector(args.initial_setpoint, "--initial-setpoint")
            if args.initial_setpoint is not None else None
        )
        args.parsed_setpoint_schedule = (
            [parse_setpoint_step(raw) for raw in args.setpoint_step]
            if args.setpoint_step is not None
            else ([] if args.clear_setpoint_schedule else None)
        )
    except ValueError as ex:
        ap.error(str(ex))

    suite, cases = build_cases(args)
    episode_steps = int(args.episode_steps) if args.episode_steps is not None else suite.get("episode_steps")
    control_dt = float(args.control_dt) if args.control_dt is not None else suite.get("control_dt")
    started = perf_counter()
    rows = []
    results = []
    configs = []
    rollouts = []
    errors = []

    for suite_case in cases:
        artifact = run_case(
            suite_case,
            include_tracebacks=args.tracebacks,
            save_rollout=(
                suite_case["objective"] == "tracking"
                or args.save_rollouts
                or args.plot
            ),
            rollout_steps=args.rollout_steps,
        )
        rows.append(artifact["row"])
        print_case(artifact["row"])
        if artifact["status"] in ("passed", "degraded"):
            results.append(artifact["result"])
            configs.append(artifact["config"])
            if artifact.get("rollout") is not None:
                rollouts.append(artifact["rollout"])
            if args.fail_fast and args.fail_on_degraded and artifact["status"] == "degraded":
                break
        else:
            errors.append({
                "suite_case": suite_case["name"],
                "scenario": suite_case["scenario"],
                "task": suite_case["task"],
                "objective": suite_case["objective"],
                "controller": suite_case["controller"],
                "status": artifact["status"],
                "error": artifact["error"],
            })
            if args.fail_fast and artifact["status"] == "failed":
                break

    counts = {
        "passed": sum(1 for row in rows if row["execution_status"] == "passed"),
        "degraded": sum(1 for row in rows if row["execution_status"] == "degraded"),
        "skipped": sum(1 for row in rows if row["execution_status"] == "skipped"),
        "failed": sum(1 for row in rows if row["execution_status"] == "failed"),
        "total": len(rows),
    }
    summary_table = suite_pipeline.build_summary_table(rows)
    suite_config = suite_pipeline.effective_suite_config(
        suite, cases, episode_steps, control_dt
    )
    artifact_dir = artifact_dir_for(args, suite, cases)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "benchmark_suite",
        "suite": suite["name"],
        "description": suite["description"],
        "suite_config": suite_config,
        "suite_source_config": suite,
        "defaults": {
            "episodes": args.episodes,
            "episode_steps": episode_steps,
            "seed": args.seed,
            "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
            "control_dt": control_dt,
            "fail_on_degraded": bool(args.fail_on_degraded),
            "overwrite": bool(args.overwrite),
        },
        "counts": counts,
        "runtime_seconds": float(perf_counter() - started),
        "rows": rows,
        "summary_table": summary_table,
        "degraded_cases": [row for row in rows if row["execution_status"] == "degraded"],
        "configs": configs,
        "results": results,
        "rollouts": rollouts,
        "report": build_evaluation_report(results) if results else {},
        "errors": errors,
        "artifact_dir": artifact_dir,
        "config": _command_config(args, artifact_dir, cases),
    }
    identities = {
        (case["scenario"], case["task"], case["objective"])
        for case in cases
    }
    if len(identities) == 1:
        first = cases[0]
        payload.update({
            "scenario": first["scenario"],
            "task": first["task"],
            "task_status": first["task_status"],
            "task_profile_hash": first["task_profile_hash"],
            "objective": first["objective"],
            "objective_source": first["objective_source"],
            "controllers": list(dict.fromkeys(
                case["controller"] for case in cases
            )),
            "protocol": first["protocol"].metadata(),
        })
    if args.task is not None:
        payload.update(
            _single_task_payload_fields(args, cases, configs, artifact_dir)
        )
    finalize_benchmark_artifacts(
        artifact_dir,
        payload,
        create_plots=True,
        markdown_report=True,
        replace_existing=args.overwrite,
    )

    print(
        f"saved artifacts {artifact_dir} "
        f"passed={counts['passed']} degraded={counts['degraded']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    if counts["failed"]:
        raise SystemExit(1)
    if args.fail_on_degraded and counts["degraded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
