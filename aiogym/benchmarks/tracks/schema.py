"""Schema and policy-contract validation for benchmark tracks."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiogym.models.cases import (
    case_profile_hash,
    load_case,
)
from aiogym.rewards.registry import get_reward_spec


BENCHMARK_TRACK_SCHEMA_VERSION = "aiogym.benchmark_track.v1"
TRACK_SPLITS = ("training", "validation", "test")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "description",
        "scenario",
        "goal",
        "reward_spec",
        "scorecard_spec",
        "policy_scope",
        "policy_contract",
        "training",
        "validation",
        "test",
        "ranking",
        "release_audit",
    }
)
_POLICY_CONTRACT_FIELDS = frozenset(
    {
        "action_mode",
        "control_dt",
        "disturbance_obs",
        "previous_action_obs",
        "normalize_observations",
        "tracking_error_obs",
        "integral_obs",
        "case_id_obs",
        "observation_mode",
        "temporal_observation",
        "history_length",
        "include_action_history",
        "recurrent_state_shape",
        "action_shape",
        "action_low",
        "action_high",
        "action_features",
        "observation_shape",
        "observation_features",
        "controlled_output_shape",
        "controlled_output_features",
        "reward_spec_id",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "case",
        "base_case",
        "variant_id",
        "overrides",
        "weight",
        "group",
        "pair_id",
        "condition",
        "base_case_id",
    }
)
_RANKING_FIELDS = frozenset(
    {
        "id",
        "primary_utility",
        "safety_gate",
        "case_score",
        "track_aggregation",
        "anchor_id",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _mapping_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedTrackCase:
    """One schema-validated case entry resolved from a track split."""

    split: str
    case_id: str
    profile: Mapping[str, Any]
    resolved_case_hash: str
    weight: float = 1.0
    group: str | None = None
    pair_id: str | None = None
    condition: str | None = None
    base_case_id: str | None = None
    variant_id: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "case_id": self.case_id,
            "resolved_case_hash": self.resolved_case_hash,
            "weight": self.weight,
            "group": self.group,
            "pair_id": self.pair_id,
            "condition": self.condition,
            "base_case_id": self.base_case_id,
            "variant_id": self.variant_id,
            "profile": deepcopy(dict(self.profile)),
        }


@dataclass(frozen=True, init=False)
class TrackSpec:
    """Immutable benchmark track with explicit goal, reward, and cases."""

    _canonical_json: str
    _official: bool

    def __init__(
        self,
        declaration: Mapping[str, Any],
        *,
        official: bool = False,
    ) -> None:
        data = deepcopy(dict(declaration))
        validate_track_declaration(data)
        object.__setattr__(self, "_canonical_json", _canonical_json(data))
        object.__setattr__(self, "_official", bool(official))

    @property
    def declaration(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def id(self) -> str:
        return str(self.declaration["id"])

    @property
    def scenario(self) -> str:
        return str(self.declaration["scenario"])

    @property
    def goal(self) -> str:
        return str(self.declaration["goal"])

    @property
    def reward_spec_id(self) -> str:
        return str(self.declaration["reward_spec"])

    @property
    def policy_scope(self) -> str:
        return str(self.declaration["policy_scope"])

    @property
    def policy_contract(self) -> dict[str, Any]:
        return deepcopy(self.declaration["policy_contract"])

    @property
    def scorecard_spec_id(self) -> str:
        return str(self.declaration["scorecard_spec"])

    @property
    def official(self) -> bool:
        return self._official

    @property
    def ranking_declaration(self) -> dict[str, Any]:
        return deepcopy(dict(self.declaration["ranking"]))

    @property
    def ranking_spec_id(self) -> str:
        return str(self.ranking_declaration["id"])

    def safety_gate_spec(self):
        from aiogym.benchmarks.safety_gates import get_safety_gate_spec

        return get_safety_gate_spec(
            str(self.ranking_declaration["safety_gate"])
        )

    @property
    def track_hash(self) -> str:
        return hashlib.sha256(
            self._canonical_json.encode("utf-8")
        ).hexdigest()

    def seed_namespace(self, split: str) -> str:
        _require_split(split)
        return str(self.declaration[split]["seed_spec"]["namespace"])

    @property
    def train_distribution_id(self) -> str | None:
        value = self.declaration["training"].get("distribution_id")
        return None if value is None else str(value)

    def training_distribution(self):
        """Load and validate the programmatic distribution for this Track."""

        distribution_id = self.train_distribution_id
        if distribution_id is None:
            raise ValueError(
                f"benchmark track {self.id!r} has no training distribution"
            )
        from aiogym.generation.registry import load_distribution

        distribution = load_distribution(distribution_id)
        if distribution.scenario_id != self.scenario:
            raise ValueError(
                "training distribution scenario does not match Track"
            )
        if distribution.goal != self.goal:
            raise ValueError(
                "training distribution goal does not match Track"
            )
        declared_dt = float(self.policy_contract["control_dt"])
        if not math.isclose(
            distribution.control_dt,
            declared_dt,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "training distribution control_dt does not match Track"
            )
        return distribution

    def resolved_cases(self, split: str) -> tuple[ResolvedTrackCase, ...]:
        _require_split(split)
        raw_cases = self.declaration[split]["cases"]
        return tuple(
            _resolve_track_case(
                self.scenario,
                split,
                entry,
            )
            for entry in raw_cases
        )

    def validate_policy_contract(self, env_factory=None) -> dict[str, Any]:
        """Validate every case against one fixed policy-facing contract."""

        if env_factory is None:
            from aiogym._environment.builder import (
                build_case_environment,
            )

            env_factory = lambda case: build_case_environment(
                self.scenario,
                case,
                self.reward_spec_id,
            )
        cases = _unique_resolved_cases(self)
        baseline = None
        for case in cases:
            env = env_factory(case)
            try:
                actual = policy_contract_for_env(env)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            if baseline is None:
                baseline = actual
            else:
                _compare_contracts(
                    baseline,
                    actual,
                    context=f"case {case.case_id!r}",
                )
            _compare_declared_contract(
                self.policy_contract,
                actual,
                context=f"case {case.case_id!r}",
            )
        return dict(baseline or {})

    def metadata(self) -> dict[str, Any]:
        return {
            **self.declaration,
            "track_hash": self.track_hash,
            "resolved_cases": {
                split: [
                    case.metadata()
                    for case in self.resolved_cases(split)
                ]
                for split in TRACK_SPLITS
            },
        }


def validate_track_declaration(declaration: Mapping[str, Any]) -> None:
    if not isinstance(declaration, Mapping):
        raise TypeError("benchmark track must be a mapping")
    unknown = set(declaration) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(
            "unknown benchmark track fields: "
            + ", ".join(sorted(unknown))
        )
    required = (
        "schema_version",
        "id",
        "scenario",
        "goal",
        "reward_spec",
        "scorecard_spec",
        "policy_scope",
        "policy_contract",
        *TRACK_SPLITS,
    )
    missing = [name for name in required if name not in declaration]
    if missing:
        raise ValueError(
            "benchmark track is missing required fields: "
            + ", ".join(missing)
        )
    if declaration["schema_version"] != BENCHMARK_TRACK_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark track schema: "
            f"{declaration['schema_version']!r}"
        )
    for name in ("id", "scenario", "scorecard_spec"):
        if not isinstance(declaration[name], str) or not declaration[name]:
            raise ValueError(f"benchmark track {name} must be non-empty")
    if declaration["goal"] not in {"regulation", "economic"}:
        raise ValueError(
            "benchmark track goal must be one of: economic, regulation"
        )
    reward_spec = get_reward_spec(str(declaration["reward_spec"]))
    if reward_spec.goal != declaration["goal"]:
        raise ValueError(
            f"reward spec {reward_spec.id!r} has goal {reward_spec.goal!r}, "
            f"not track goal {declaration['goal']!r}"
        )
    if declaration["policy_scope"] not in {"generalist", "specialist"}:
        raise ValueError(
            "benchmark track policy_scope must be one of: "
            "generalist, specialist"
        )
    ranking = declaration.get("ranking")
    if not isinstance(ranking, Mapping):
        raise TypeError("benchmark track ranking must be a mapping")
    unknown_ranking = set(ranking) - _RANKING_FIELDS
    if unknown_ranking:
        raise ValueError(
            "unknown benchmark ranking fields: "
            + ", ".join(sorted(unknown_ranking))
        )
    required_ranking = (
        "id",
        "primary_utility",
        "safety_gate",
        "case_score",
        "track_aggregation",
    )
    missing_ranking = [
        name for name in required_ranking if not ranking.get(name)
    ]
    if missing_ranking:
        raise ValueError(
            "benchmark ranking is missing required fields: "
            + ", ".join(missing_ranking)
        )
    expected_utility = (
        "negative_regulation_cost_rate"
        if declaration["goal"] == "regulation"
        else "profit_rate"
    )
    if ranking["primary_utility"] != expected_utility:
        raise ValueError(
            "benchmark ranking primary_utility conflicts with Track goal"
        )
    if ranking["case_score"] not in {
        "fixed-anchor-v1",
        "diagnostic-only-v1",
    }:
        raise ValueError("unsupported benchmark case_score rule")
    if ranking["track_aggregation"] not in {
        "weighted-geometric-mean-v1",
        "arithmetic-mean-v1",
    }:
        raise ValueError("unsupported benchmark track_aggregation rule")
    if ranking["case_score"] == "fixed-anchor-v1" and not isinstance(
        ranking.get("anchor_id"),
        str,
    ):
        raise ValueError(
            "fixed-anchor-v1 ranking requires a non-empty anchor_id"
        )
    from aiogym.benchmarks.safety_gates import get_safety_gate_spec

    get_safety_gate_spec(str(ranking["safety_gate"]))
    contract = declaration["policy_contract"]
    if not isinstance(contract, Mapping):
        raise TypeError("benchmark track policy_contract must be a mapping")
    unknown_contract = set(contract) - _POLICY_CONTRACT_FIELDS
    if unknown_contract:
        raise ValueError(
            "unknown policy contract fields: "
            + ", ".join(sorted(unknown_contract))
        )
    required_contract = (
        "action_mode",
        "control_dt",
        "disturbance_obs",
        "previous_action_obs",
        "normalize_observations",
        "tracking_error_obs",
        "case_id_obs",
    )
    missing_contract = [
        name for name in required_contract if name not in contract
    ]
    if missing_contract:
        raise ValueError(
            "policy contract is missing required fields: "
            + ", ".join(missing_contract)
        )
    if contract["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError(
            "policy contract action_mode must be one of: actuator, setpoint"
        )
    control_dt = float(contract["control_dt"])
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("policy contract control_dt must be positive")
    for name in (
        "disturbance_obs",
        "previous_action_obs",
        "normalize_observations",
        "tracking_error_obs",
        "case_id_obs",
    ):
        if not isinstance(contract[name], bool):
            raise TypeError(f"policy contract {name} must be a boolean")
    if contract["case_id_obs"]:
        raise ValueError(
            "case_id_obs=true is not supported by generalist benchmark tracks"
        )
    observation_mode = contract.get("observation_mode", "full_state")
    if observation_mode not in {"full_state", "measured_output"}:
        raise ValueError(
            "policy contract observation_mode must be one of: "
            "full_state, measured_output"
        )
    temporal = contract.get("temporal_observation", "single_step")
    if temporal not in {"single_step", "history", "recurrent"}:
        raise ValueError(
            "policy contract temporal_observation must be one of: "
            "single_step, history, recurrent"
        )
    history_length = contract.get("history_length", 1)
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or history_length <= 0
    ):
        raise ValueError(
            "policy contract history_length must be a positive integer"
        )
    include_actions = contract.get("include_action_history", False)
    if not isinstance(include_actions, bool):
        raise TypeError(
            "policy contract include_action_history must be a boolean"
        )
    recurrent_shape = contract.get("recurrent_state_shape", [])
    if not isinstance(recurrent_shape, list) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in recurrent_shape
    ):
        raise ValueError(
            "policy contract recurrent_state_shape must contain "
            "positive integers"
        )
    if temporal == "single_step" and (
        history_length != 1 or include_actions or recurrent_shape
    ):
        raise ValueError(
            "single-step policy contract cannot declare temporal state"
        )
    if temporal == "history" and (
        history_length < 2 or recurrent_shape
    ):
        raise ValueError(
            "history policy contract requires history_length >= 2 "
            "and no recurrent state"
        )
    if temporal == "recurrent" and (
        history_length != 1 or include_actions or not recurrent_shape
    ):
        raise ValueError(
            "recurrent policy contract requires recurrent_state_shape "
            "without frame history"
        )

    namespaces = {}
    for split in TRACK_SPLITS:
        section = declaration[split]
        if not isinstance(section, Mapping):
            raise TypeError(f"benchmark track {split} must be a mapping")
        cases = section.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(
                f"benchmark track {split} cases must be a non-empty list"
            )
        for entry in cases:
            _validate_case_entry(entry, split)
        seed_spec = section.get("seed_spec")
        if not isinstance(seed_spec, Mapping):
            raise TypeError(
                f"benchmark track {split} seed_spec must be a mapping"
            )
        namespace = seed_spec.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(
                f"benchmark track {split} seed namespace must be non-empty"
            )
        namespaces[split] = namespace
    if len(set(namespaces.values())) != len(namespaces):
        raise ValueError(
            "training, validation, and test seed namespaces must be distinct"
        )
    distribution_id = declaration["training"].get("distribution_id")
    if distribution_id is not None and (
        not isinstance(distribution_id, str) or not distribution_id
    ):
        raise ValueError(
            "benchmark track training distribution_id must be non-empty"
        )
    if (
        declaration["policy_scope"] == "specialist"
        and len(declaration["training"]["cases"]) != 1
    ):
        raise ValueError(
            "specialist benchmark tracks must declare exactly one training case"
        )


def policy_contract_for_env(env) -> dict[str, Any]:
    """Return the complete policy-facing contract of a constructed env."""

    observation_mode = str(
        getattr(env.unwrapped, "observation_mode", "full_state")
    )
    output_features = [
        str(row.get("name", f"output_{index}"))
        for index, row in enumerate(env.model.setpoint_schema())
    ]
    state_features = (
        [
            str(row.get("name", f"state_{index}"))
            for index, row in enumerate(env.model.state_schema())
        ]
        if observation_mode == "full_state"
        else [f"measured_output:{name}" for name in output_features]
    )
    reference_prefix = (
        "tracking_error" if env.tracking_error_obs else "setpoint"
    )
    observation_features = [
        *state_features,
        *[f"{reference_prefix}:{name}" for name in output_features],
    ]
    if env.disturbance_obs:
        observation_features.extend(
            f"disturbance:{name}"
            for name in env.model.dynamics_disturbance_names()
        )
    if env.previous_action_obs:
        observation_features.extend(
            f"previous_action:{row.get('name', index)}"
            for index, row in enumerate(env.model.action_schema())
        )
    if env.integral_obs and env.model.supports_integral_observation:
        observation_features.extend(
            f"integral_error:{name}" for name in output_features
        )
    temporal_contract = getattr(env, "observation_contract", None)
    temporal = (
        temporal_contract.temporal
        if temporal_contract is not None
        else "single_step"
    )
    history_length = (
        temporal_contract.history_length
        if temporal_contract is not None
        else 1
    )
    include_action_history = bool(
        temporal_contract.include_action_history
        if temporal_contract is not None
        else False
    )
    if env.action_mode == "setpoint":
        action_features = [
            (
                f"setpoint:{item[1]}"
                if item[0] == "y_sp"
                else f"manipulated_variable:{item[1]}"
            )
            for item in env.layout
        ]
    else:
        action_features = [
            str(row.get("name", f"action_{index}"))
            for index, row in enumerate(env.model.action_schema())
        ]
    if temporal == "history":
        base_features = list(observation_features)
        observation_features = [
            f"history[{offset}]:{feature}"
            for offset in range(history_length)
            for feature in base_features
        ]
        if include_action_history:
            observation_features.extend(
                f"action_history[{offset}]:{feature}"
                for offset in range(history_length - 1)
                for feature in action_features
            )
    return {
        "scenario": str(env.scenario),
        "action_mode": str(env.action_mode),
        "control_dt": float(env.control_dt),
        "disturbance_obs": bool(env.disturbance_obs),
        "previous_action_obs": bool(env.previous_action_obs),
        "normalize_observations": bool(env.normalize_observations),
        "tracking_error_obs": bool(env.tracking_error_obs),
        "observation_mode": observation_mode,
        "temporal_observation": temporal,
        "history_length": history_length,
        "include_action_history": include_action_history,
        "recurrent_state_shape": (
            list(temporal_contract.recurrent_state_shape)
            if temporal_contract is not None
            else []
        ),
        "integral_obs": bool(env.integral_obs),
        "case_id_obs": False,
        "action_shape": list(env.action_space.shape),
        "action_low": np.asarray(env.action_space.low).tolist(),
        "action_high": np.asarray(env.action_space.high).tolist(),
        "action_features": action_features,
        "observation_shape": list(env.observation_space.shape),
        "observation_features": observation_features,
        "controlled_output_shape": [len(output_features)],
        "controlled_output_features": output_features,
        "reward_spec_id": str(env.reward_spec_id),
    }


def _validate_case_entry(entry, split):
    if isinstance(entry, str):
        if not entry:
            raise ValueError(f"benchmark track {split} case ID must be non-empty")
        return
    if not isinstance(entry, Mapping):
        raise TypeError(
            f"benchmark track {split} case entries must be strings or mappings"
        )
    unknown = set(entry) - _ENTRY_FIELDS
    if unknown:
        raise ValueError(
            f"unknown benchmark track {split} case fields: "
            + ", ".join(sorted(unknown))
        )
    references = [name for name in ("case", "base_case") if entry.get(name)]
    if len(references) != 1:
        raise ValueError(
            f"benchmark track {split} case entry requires exactly one of "
            "case or base_case"
        )
    if "overrides" in entry and "base_case" not in entry:
        raise ValueError("track case overrides require base_case")
    if "overrides" in entry and not isinstance(entry["overrides"], Mapping):
        raise TypeError("track case overrides must be a mapping")
    pair_id = entry.get("pair_id")
    condition = entry.get("condition")
    if (pair_id is None) != (condition is None):
        raise ValueError(
            "track case pair_id and condition must be declared together"
        )
    if condition not in {None, "nominal", "shifted"}:
        raise ValueError(
            "track case condition must be one of: nominal, shifted"
        )
    if entry.get("base_case_id") is not None and pair_id is None:
        raise ValueError("track case base_case_id requires pair_id")
    if split == "training":
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("training case weight must be finite and positive")


def _resolve_track_case(scenario, split, raw_entry):
    entry = {"case": raw_entry} if isinstance(raw_entry, str) else dict(raw_entry)
    reference = entry.get("case") or entry.get("base_case")
    profile = load_case(
        str(reference),
        scenario=scenario,
        overrides=entry.get("overrides"),
    )
    variant_id = entry.get("variant_id")
    case_name = str(profile["name"])
    case_id = (
        f"{case_name}:{variant_id}"
        if variant_id is not None
        else case_name
    )
    return ResolvedTrackCase(
        split=split,
        case_id=case_id,
        profile=profile,
        resolved_case_hash=case_profile_hash(profile),
        weight=float(entry.get("weight", 1.0)),
        group=entry.get("group"),
        pair_id=entry.get("pair_id"),
        condition=entry.get("condition"),
        base_case_id=entry.get("base_case_id"),
        variant_id=variant_id,
    )


def _unique_resolved_cases(track):
    unique = {}
    for split in TRACK_SPLITS:
        for case in track.resolved_cases(split):
            unique.setdefault(case.resolved_case_hash, case)
    return tuple(unique.values())


def _compare_declared_contract(declared, actual, *, context):
    for name, expected in declared.items():
        value = actual.get(name)
        if value != expected:
            raise ValueError(
                f"policy contract mismatch for {context}: "
                f"{name} expected {expected!r}, got {value!r}"
            )


def _compare_contracts(expected, actual, *, context):
    for name in expected:
        if actual.get(name) != expected[name]:
            raise ValueError(
                f"incompatible policy contract for {context}: "
                f"{name} differs ({expected[name]!r} != "
                f"{actual.get(name)!r})"
            )


def _require_split(split):
    if split not in TRACK_SPLITS:
        raise ValueError(
            "benchmark track split must be one of: "
            + ", ".join(TRACK_SPLITS)
        )


__all__ = [
    "BENCHMARK_TRACK_SCHEMA_VERSION",
    "TRACK_SPLITS",
    "ResolvedTrackCase",
    "TrackSpec",
    "policy_contract_for_env",
    "validate_track_declaration",
]
