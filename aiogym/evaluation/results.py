"""Evaluation schemas, compact rows, aggregation, and case acceptance."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    ROLLOUT_SCHEMA,
    SCORECARD_GROUPS,
    SCORECARD_METRICS,
)
from .metrics.robustness import (
    paired_robustness_summary,
    robustness_extrema,
)
from ..models.cases import (
    validate_case_profile,
)


def result_schema():
    return {
        "version": EVALUATION_SCHEMA_VERSION,
        "rollout": dict(ROLLOUT_SCHEMA),
        "episode_metrics": dict(METRIC_DEFINITIONS),
        "scorecard": {
            group: list(metrics)
            for group, metrics in SCORECARD_GROUPS.items()
        },
    }


_ROBUSTNESS_METRICS = tuple(
    metric
    for metric in SCORECARD_METRICS
    if metric in METRIC_DIRECTIONS
)

def build_evaluation_report(results: Sequence[Mapping[str, Any]]):
    """Return one complete grouped scorecard table for every result."""

    scorecard = {
        group: [
            _table_row(result, metrics)
            for result in results
        ]
        for group, metrics in SCORECARD_GROUPS.items()
    }
    paired_robustness = paired_robustness_report(results)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scorecard": scorecard,
        "views": {
            "regulation": scorecard["regulation"],
            "economic": scorecard["economics"],
            "safety": scorecard["safety"],
            "robustness": paired_robustness,
            "controller": scorecard["controller"],
        },
        "paired_robustness": paired_robustness,
        "evaluation_pass_count": 1,
        "metric_definitions": dict(METRIC_DEFINITIONS),
    }


def paired_robustness_report(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build robustness summaries for every declared nominal/shifted pair."""

    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for result in results:
        pair_id = result.get("pair_id")
        pair_role = result.get("pair_role")
        if pair_id is None and pair_role is None:
            continue
        if pair_role not in {"nominal", "shifted"}:
            raise ValueError(
                "paired result pair_role must be one of: nominal, shifted"
            )
        key = (
            str(pair_id),
            str(result.get("base_case_id") or ""),
            str(result.get("controller_name") or ""),
        )
        if pair_role in groups.setdefault(key, {}):
            raise ValueError(
                f"duplicate {pair_role} result for robustness pair {key}"
            )
        groups[key][str(pair_role)] = result

    summaries = []
    for (pair_id, base_case_id, controller_name), members in sorted(
        groups.items()
    ):
        missing = {"nominal", "shifted"} - set(members)
        if missing:
            summaries.append(
                {
                    "pair_id": pair_id,
                    "base_case_id": base_case_id,
                    "controller_name": controller_name,
                    "status": "incomplete",
                    "missing_roles": sorted(missing),
                }
            )
            continue
        if not (
            members["nominal"].get("episode_metrics")
            and members["shifted"].get("episode_metrics")
        ):
            summaries.append(
                {
                    "pair_id": pair_id,
                    "base_case_id": base_case_id,
                    "controller_name": controller_name,
                    "status": "not-evaluated",
                    "reason": "paired episode metrics were not recorded",
                }
            )
            continue
        metric_keys = [
            metric
            for metric in _ROBUSTNESS_METRICS
            if metric in METRIC_DIRECTIONS
        ]
        summary = paired_robustness_summary(
            members["nominal"].get("episode_metrics", []),
            members["shifted"].get("episode_metrics", []),
            metric_keys,
            METRIC_DIRECTIONS,
        )
        summaries.append(
            {
                "pair_id": pair_id,
                "base_case_id": base_case_id,
                "controller_name": controller_name,
                "status": "complete",
                "ranking_eligible": all(
                    bool(member.get("ranking_eligible", True))
                    for member in members.values()
                ),
                **summary,
            }
        )
    return summaries

def _aggregate_metric_keys(per_episode):
    keys = set()
    for row in per_episode:
        for key, value in row.items():
            if key in ("episode", "seed", "steps"):
                continue
            if isinstance(value, (int, float, np.number)):
                keys.add(key)
    return sorted(keys)

def _table_row(result: Mapping[str, Any], keys: Sequence[str]):
    row = {
        "controller": result.get("controller_name"),
        "goal": result.get("goal"),
        "case": result.get("case", "default"),
        "case_status": result.get("case_status", "implicit-default"),
        "case_profile_hash": result.get("case_profile_hash"),
        "control_structure": dict(result.get("controller", {})).get("control_structure"),
        "controller_status": result.get("controller_status", "ok"),
        "controller_diagnostics": result.get("controller_diagnostics", {}),
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list", []),
    }
    for key in keys:
        if key in result:
            row[key] = result[key]
        std_key = f"{key}_std"
        if std_key in result:
            row[std_key] = result[std_key]
    return row
def _robustness_row(result: Mapping[str, Any]):
    row = _table_row(result, _ROBUSTNESS_METRICS)
    row.update(robustness_extrema(
        result.get("episode_metrics", []),
        _ROBUSTNESS_METRICS,
        METRIC_DIRECTIONS,
    ))
    return row


def compact_result_row(
    result: Mapping[str, Any],
    *,
    scenario: str | None = None,
    goal: str | None = None,
    action_mode: str | None = None,
    case: str | None = None,
    case_status: str | None = None,
    case_profile_hash: str | None = None,
    run_case_id: str | None = None,
    controller: str | None = None,
) -> dict[str, Any]:
    """Project one canonical evaluation result into a stable summary row."""

    result = deepcopy(dict(result))
    controller_meta = dict(result.get("controller") or {})
    diagnostics = dict(result.get("controller_diagnostics") or {})
    metric = str(result.get("metric") or "")
    name = str(
        result.get("controller_name")
        or controller_meta.get("name")
        or controller
        or ""
    )
    controller_status = str(result.get("controller_status", "ok"))
    execution_status = str(result.get(
        "execution_status",
        "degraded" if controller_status == "degraded" else "passed",
    ))
    row = {
        "run_case_id": run_case_id,
        "scenario": (
            scenario
            or result.get("scenario")
            or dict(result.get("model") or {}).get("scenario")
        ),
        "case": case or result.get("case") or "default",
        "case_status": case_status or result.get("case_status") or "implicit-default",
        "case_profile_hash": case_profile_hash or result.get("case_profile_hash"),
        "goal": goal or result.get("goal"),
        "goal_source": result.get("goal_source"),
        "acceptance_status": result.get("acceptance_status", "not-defined"),
        "reward_spec_id": result.get("reward_spec_id"),
        "track_id": result.get("track_id"),
        "track_hash": result.get("track_hash"),
        "resolved_case_hash": result.get("resolved_case_hash"),
        "scorecard_spec_id": result.get("scorecard_spec_id"),
        "ranking_spec_id": result.get("ranking_spec_id"),
        "policy_scope": result.get("policy_scope"),
        "seed_namespace": result.get("seed_namespace"),
        "action_mode": action_mode or controller_meta.get("action_mode"),
        "controller": name,
        "control_structure": controller_meta.get("control_structure"),
        "execution_status": execution_status,
        "controller_status": controller_status,
        "controller_solve_count": diagnostics.get("solve_count", 0),
        "controller_solver_success_count": diagnostics.get("solver_success_count", 0),
        "controller_solver_limited_count": diagnostics.get("solver_limited_count", 0),
        "controller_solver_failure_count": diagnostics.get("solver_failure_count", 0),
        "controller_fallback_count": diagnostics.get("fallback_count", 0),
        "controller_last_solver_error": diagnostics.get("last_solver_error"),
        "metric": metric,
        "official_score": result.get("official_score"),
        "ranking_eligible": result.get("ranking_eligible", True),
        "safety_gate": deepcopy(result.get("safety_gate")),
        "safety_mode": result.get("safety_mode", "ordinary"),
        "initial_safety_debt": result.get(
            "initial_safety_debt",
            False,
        ),
        "case_id": result.get("case_id"),
        "pair_id": result.get("pair_id"),
        "pair_role": result.get("pair_role"),
        "base_case_id": result.get("base_case_id"),
        "episodes": result.get("episodes"),
        "seed": result.get("seed"),
        "seed_list": result.get("seed_list", []),
    }
    if metric:
        row[metric] = result.get(metric)
        row[f"{metric}_std"] = result.get(f"{metric}_std")
    for key in (*SCORECARD_METRICS, "return", "runtime_total_seconds"):
        row[key] = result.get(key)
    return row


def evaluate_case_acceptance(
    profile: Mapping[str, Any] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate optional metric thresholds separately from execution status."""

    if profile is None:
        return {"status": "not-defined", "checks": []}
    validate_case_profile(profile)
    acceptance = profile.get("acceptance", {})
    thresholds = acceptance.get("metrics") if isinstance(acceptance, Mapping) else None
    if not thresholds:
        return {"status": "not-defined", "checks": []}
    checks = []
    met = True
    for metric, bounds in thresholds.items():
        value = result.get(metric)
        check = {"metric": metric, "value": value, **dict(bounds)}
        check_met = isinstance(value, (int, float))
        if check_met and "min" in bounds:
            check_met = float(value) >= float(bounds["min"])
        if check_met and "max" in bounds:
            check_met = float(value) <= float(bounds["max"])
        check["met"] = bool(check_met)
        checks.append(check)
        met = met and bool(check_met)
    return {"status": "met" if met else "not-met", "checks": checks}
