"""Config-driven public benchmark orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..._internal.config import as_list, load_config, protocol_data
from ..._internal.paths import run_path, timestamped_artifact_path
from ..._internal.serialization import jsonable
from ..cases import BenchmarkCase
from ..metric_catalog import PUBLIC_BENCHMARK_SCHEMA_VERSION
from ..protocols import resolve_protocol
from ..results import build_evaluation_report
from .runner import execute_benchmark_case


def run_benchmark(
    config: str | Path | Mapping[str, Any],
    *,
    objective: str | None = None,
) -> dict:
    """Run a config-driven controller benchmark and write reusable artifacts."""

    cfg = _benchmark_config_dict(config)
    removed = sorted(set(cfg) & {
        "model", "env", "protocol_kwargs", "controller", "seed_list",
        "seed", "raise_on_error", "run_dir", "controller_config",
    })
    if removed:
        raise ValueError(
            f"unsupported benchmark config field(s): {', '.join(removed)}; "
            "use the canonical public API fields"
        )
    scenario = cfg.get("scenario", "cstr")
    protocol_options = protocol_data(cfg)
    environment = cfg.get("environment", {})
    if not isinstance(environment, Mapping):
        raise TypeError("config['environment'] must be a mapping")
    protocol_options.update(dict(environment))
    raw_protocol = cfg.get("protocol")
    if isinstance(raw_protocol, Mapping):
        protocol_options.update(dict(raw_protocol))
    elif raw_protocol is not None:
        raise TypeError("config['protocol'] must be a mapping")
    protocol = resolve_protocol(
        scenario,
        objective=objective,
        data=protocol_options,
    )
    scenario = protocol.scenario
    task_meta = protocol.metadata()["task_identity"]
    controller_names = as_list(cfg.get("controllers", ["pid"]))
    if not controller_names:
        raise ValueError("benchmark must include at least one controller")
    controller_configs = dict(cfg.get("controller_configs", {}))
    seeds = tuple(int(seed) for seed in cfg.get("seeds", [0]))
    if not seeds:
        raise ValueError("benchmark must include at least one seed")
    strict = bool(cfg.get("strict", False))
    include_episodes = bool(cfg.get("include_episodes", True))
    save_rollouts = bool(cfg.get("save_rollouts", False))
    rollout_steps = cfg.get("rollout_steps")
    overwrite = cfg.get("overwrite", True)
    if not isinstance(overwrite, bool):
        raise TypeError("config['overwrite'] must be a boolean")
    base_out_dir = Path(
        cfg["output_dir"]
        if "output_dir" in cfg
        else run_path(scenario, task_meta["name"])
    )
    out_dir = (
        base_out_dir
        if overwrite
        else timestamped_artifact_path(base_out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rows = []
    rollouts = []
    errors = []
    case_specs = []
    for name in controller_names:
        case_spec = BenchmarkCase.from_protocol(
            protocol,
            controller=str(name),
            seeds=seeds,
            controller_config=controller_configs.get(str(name), {}),
        )
        case_specs.append(case_spec)
        artifact = execute_benchmark_case(
            case_spec,
            include_episodes=include_episodes,
            save_rollout=save_rollouts,
            rollout_steps=rollout_steps,
        )
        if artifact["status"] == "failed":
            if strict:
                raise RuntimeError(
                    f"controller '{name}' benchmark failed: "
                    f"{artifact['error']['message']}"
                )
            rows.append(artifact["row"])
            errors.append({"controller": str(name), **artifact["error"]})
            continue
        results.append(artifact["result"])
        rows.append(artifact["row"])
        if artifact["rollout"] is not None:
            rollouts.append(artifact["rollout"])

    benchmark_config = {
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "scenario": scenario,
        "task": task_meta["name"],
        "task_status": task_meta["status"],
        "task_profile_hash": task_meta["profile_hash"],
        "objective": protocol.objective,
        "objective_source": protocol.objective_source,
        "controllers": [str(name) for name in controller_names],
        "seed_list": list(seeds),
        "protocol": protocol.metadata(),
        "cases": [case.metadata() for case in case_specs],
    }
    payload = {
        "schema_version": PUBLIC_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "public_benchmark",
        "run_dir": str(out_dir),
        "config": jsonable(cfg),
        "benchmark_config": benchmark_config,
        "scenario": scenario,
        "task": task_meta["name"],
        "task_status": task_meta["status"],
        "task_profile_hash": task_meta["profile_hash"],
        "objective": protocol.objective,
        "objective_source": protocol.objective_source,
        "controllers": [str(name) for name in controller_names],
        "rows": rows,
        "results": results,
        "rollouts": rollouts,
        "report": build_evaluation_report(results) if results else {},
        "errors": errors,
    }
    from ..artifact import finalize_benchmark_artifacts

    return finalize_benchmark_artifacts(
        out_dir,
        payload,
        replace_existing=overwrite,
    )


def _benchmark_config_dict(
    config: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    return load_config(config)
