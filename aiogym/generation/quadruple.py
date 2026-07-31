"""Physical training distributions for the quadruple-tank process."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from aiogym.models.parameter_profiles import load_parameter_profile
from aiogym.models.registry import apply_model_params, make_model

from .curriculum import (
    QUADRUPLE_CURRICULUM_V1,
    CurriculumSpec,
    get_curriculum_level,
)
from .seed_tree import SeedTree
from .specs import (
    DISTRIBUTION_SCHEMA_VERSION,
    DistributionSpec,
    EpisodeSpec,
)


_INITIAL_STATE_CORRELATION = (
    (1.0, 0.65, 0.30, 0.25),
    (0.65, 1.0, 0.25, 0.30),
    (0.30, 0.25, 1.0, 0.60),
    (0.25, 0.30, 0.60, 1.0),
)


def quadruple_training_distribution(
    level: str = "L2",
    *,
    control_dt: float = 1.0,
    episode_steps: int = 900,
) -> DistributionSpec:
    """Build one versioned L0-L4 quadruple training distribution."""

    difficulty = get_curriculum_level(level)
    profile = load_parameter_profile("quadruple")
    anchors = _operating_point_anchors(profile)
    sensor_distribution = (
        {
            "kind": "additive_gaussian",
            "noise_pct": difficulty.sensor_noise_pct,
        }
        if difficulty.sensor_noise_pct > 0
        else {"kind": "identity"}
    )
    return DistributionSpec(
        {
            "schema_version": DISTRIBUTION_SCHEMA_VERSION,
            "distribution_id": (
                "quadruple-regulation-training-"
                f"{difficulty.level_id.lower()}-v1"
            ),
            "scenario_id": "quadruple",
            "goal": "regulation",
            "control_dt": float(control_dt),
            "episode_steps": int(episode_steps),
            "plant_distribution": {
                "kind": "quadruple_correlated_parameters_v1",
                "anchors": anchors,
                "anchor_source": "quadruple parameter profile v1",
                "regime_weights": {
                    "minimum-phase": 0.5,
                    "nonminimum-phase": 0.5,
                },
                "relative_std": difficulty.plant_relative_std,
                "correlated_groups": [
                    {
                        "parameters": [
                            "outlet_area[0]",
                            "outlet_area[1]",
                            "outlet_area[2]",
                            "outlet_area[3]",
                        ],
                        "mode": "shared_multiplier",
                    },
                    {
                        "parameters": [
                            "pump_gain[0]",
                            "pump_gain[1]",
                        ],
                        "mode": "shared_multiplier",
                    },
                    {
                        "parameters": ["gamma[0]", "gamma[1]"],
                        "mode": "shared_additive_shift",
                    },
                ],
                "phase_margin": 0.05,
            },
            "initial_state_distribution": {
                "kind": "equilibrium_perturbation",
                "operating_action_half_width": (
                    difficulty.operating_action_half_width
                ),
                "relative_std": difficulty.initial_relative_std,
                "correlation": [
                    list(row) for row in _INITIAL_STATE_CORRELATION
                ],
                "state_safety_margin": (
                    difficulty.state_safety_margin
                ),
                "recovery_probability": (
                    difficulty.recovery_probability
                ),
                "projection": "clip_to_declared_state_bounds",
            },
            "reference_distribution": {
                "kind": "feasible_action_mapped_reference_v1",
                "patterns": list(difficulty.reference_patterns),
                "event_count": [
                    difficulty.reference_event_min,
                    difficulty.reference_event_max,
                ],
                "action_half_width": (
                    difficulty.reference_action_half_width
                ),
                "minimum_dwell_steps": (
                    difficulty.minimum_dwell_steps
                ),
                "reference_preview_horizon": 0,
            },
            "disturbance_distribution": {
                "kind": "quadruple_bounded_schedule_v1",
                "probability": difficulty.disturbance_probability,
                "patterns": list(difficulty.disturbance_patterns),
                "max_shift": difficulty.disturbance_max_shift,
                "observability": {
                    "pump_flow_factor": "unmeasured",
                    "outlet_area_factor": "unmeasured",
                },
            },
            "sensor_distribution": sensor_distribution,
            "actuator_distribution": {"kind": "identity"},
            "economic_context_distribution": {
                "kind": "fixed",
                "context": copy.deepcopy(
                    make_model("quadruple").economic_config
                ),
            },
            "mixture_weights": {
                "minimum-phase": 0.5,
                "nonminimum-phase": 0.5,
            },
            "curriculum_id": (
                "quadruple-regulation-curriculum-v1:"
                f"{difficulty.level_id}"
            ),
        }
    )


class QuadrupleTrainingSampler:
    """Generate reproducible, feasible quadruple-tank training episodes."""

    def __init__(
        self,
        distribution: DistributionSpec | None = None,
        *,
        level: str = "L2",
        split: str = "training",
        control_dt: float = 1.0,
        episode_steps: int = 900,
    ) -> None:
        self.distribution = (
            distribution
            if distribution is not None
            else quadruple_training_distribution(
                level,
                control_dt=control_dt,
                episode_steps=episode_steps,
            )
        )
        if not isinstance(self.distribution, DistributionSpec):
            raise TypeError("distribution must be a DistributionSpec")
        if self.distribution.scenario_id != "quadruple":
            raise ValueError(
                "QuadrupleTrainingSampler requires scenario_id='quadruple'"
        )
        self.split = str(split)

    @property
    def distribution_id(self) -> str:
        return self.distribution.distribution_id

    @property
    def distribution_hash(self) -> str:
        return self.distribution.distribution_hash

    def sample(
        self,
        seed: int,
        *,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> EpisodeSpec:
        tree = SeedTree.for_split(
            seed,
            self.distribution.distribution_id,
            self.split,
            worker_index=worker_index,
            episode_index=episode_index,
        )
        return self._sample_from_tree(tree, force_disturbance=None)

    def sample_paired(
        self,
        seed: int,
        *,
        shift_component: str = "disturbance",
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> tuple[EpisodeSpec, EpisodeSpec]:
        """Return a paired nominal/shifted episode with shared random context."""

        if shift_component != "disturbance":
            raise ValueError(
                "paired quadruple sampling currently supports only "
                "shift_component='disturbance'"
            )
        tree = SeedTree.for_split(
            seed,
            self.distribution.distribution_id,
            self.split,
            worker_index=worker_index,
            episode_index=episode_index,
        )
        nominal = self._sample_from_tree(tree, force_disturbance=False)
        shifted = self._sample_from_tree(tree, force_disturbance=True)
        return nominal, shifted

    def _sample_from_tree(
        self,
        tree: SeedTree,
        *,
        force_disturbance: bool | None,
    ) -> EpisodeSpec:
        declaration = self.distribution.declaration
        plant_spec = declaration["plant_distribution"]
        initial_spec = declaration["initial_state_distribution"]
        reference_spec = declaration["reference_distribution"]
        disturbance_spec = declaration["disturbance_distribution"]

        plant_rng = tree.generator("plant")
        initial_rng = tree.generator("initial_state")
        reference_rng = tree.generator("reference")
        disturbance_rng = tree.generator("disturbance")
        parameters, regime = _sample_correlated_parameters(
            plant_spec,
            plant_rng,
        )
        model = apply_model_params(make_model("quadruple"), parameters)
        (
            initial_state,
            equilibrium,
            operating_action,
            initial_mode,
        ) = _sample_equilibrium_initial_state(
            model,
            initial_spec,
            initial_rng,
        )
        reference_schedule = QuadrupleReferenceGenerator().generate(
            model,
            initial_reference=model.controlled_output(equilibrium),
            operating_action=operating_action,
            episode_steps=self.distribution.episode_steps,
            specification=reference_spec,
            rng=reference_rng,
        )
        disturbance_schedule = QuadrupleDisturbanceGenerator().generate(
            model,
            episode_steps=self.distribution.episode_steps,
            specification=disturbance_spec,
            rng=disturbance_rng,
            force_shift=force_disturbance,
        )
        economic_distribution = (
            declaration.get("economic_context_distribution") or {}
        )
        episode = EpisodeSpec.from_distribution(
            self.distribution,
            base_seed=tree.base_seed,
            component_seeds=tree.component_seeds,
            plant_parameters=parameters,
            initial_state=initial_state,
            reference_schedule=reference_schedule,
            disturbance_schedule=disturbance_schedule,
            sensor_model=declaration["sensor_distribution"],
            actuator_model=declaration["actuator_distribution"],
            economic_context=economic_distribution.get("context", {}),
            difficulty_tags=(
                str(declaration["curriculum_id"]).rsplit(":", 1)[-1],
                regime,
                initial_mode,
            ),
        )
        report = validate_quadruple_episode(episode)
        if not report["passed"]:
            failed = ", ".join(
                check["name"]
                for check in report["checks"]
                if not check["passed"]
            )
            raise RuntimeError(
                "generated infeasible quadruple episode: " + failed
            )
        return episode


class QuadrupleCurriculumSampler:
    """Select a fixed level from cumulative transitions, then sample it."""

    def __init__(
        self,
        curriculum: CurriculumSpec = QUADRUPLE_CURRICULUM_V1,
        *,
        split: str = "training",
        control_dt: float = 1.0,
        episode_steps: int = 900,
    ) -> None:
        self.curriculum = curriculum
        self.split = split
        self.control_dt = float(control_dt)
        self.episode_steps = int(episode_steps)

    def sample(
        self,
        seed: int,
        *,
        transition_count: int,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> EpisodeSpec:
        level = self.curriculum.level_for_transition(transition_count)
        sampler = QuadrupleTrainingSampler(
            level=level.level_id,
            split=self.split,
            control_dt=self.control_dt,
            episode_steps=self.episode_steps,
        )
        return sampler.sample(
            seed,
            worker_index=worker_index,
            episode_index=episode_index,
        )


class QuadrupleReferenceGenerator:
    """Generate reachable holds, steps, multi-steps, and compiled ramps."""

    def generate(
        self,
        model,
        *,
        initial_reference,
        operating_action,
        episode_steps: int,
        specification: Mapping[str, Any],
        rng: np.random.Generator,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        patterns = tuple(str(item) for item in specification["patterns"])
        selected = pattern or patterns[int(rng.integers(0, len(patterns)))]
        if selected not in patterns:
            raise ValueError(
                f"reference pattern {selected!r} is not declared"
            )
        schedule = [
            {
                "at_step": 0,
                "values": [float(value) for value in initial_reference],
                "kind": "initial",
            }
        ]
        if selected == "steady_hold":
            return schedule

        event_min, event_max = (
            int(value) for value in specification["event_count"]
        )
        event_count = (
            1
            if selected in {"single_step", "ramp"}
            else int(rng.integers(event_min, event_max + 1))
        )
        dwell = int(specification["minimum_dwell_steps"])
        times = _event_times(
            event_count,
            episode_steps,
            dwell,
            rng,
        )
        action = np.asarray(operating_action, dtype=np.float64)
        half_width = float(specification["action_half_width"])
        if selected == "ramp":
            target_action, _ = _sample_reference_target(
                model,
                action,
                half_width,
                rng,
            )
            start = times[0]
            available_steps = max(1, episode_steps - 1 - start)
            segments = min(6, available_steps)
            spacing = max(1, available_steps // segments)
            for index in range(1, segments + 1):
                fraction = index / segments
                ramp_action = (
                    (1.0 - fraction) * action
                    + fraction * target_action
                )
                equilibrium = model.equilibrium_state(
                    model.physical_action_vector(ramp_action)
                )
                schedule.append(
                    {
                        "at_step": min(
                            episode_steps - 1,
                            start + index * spacing,
                        ),
                        "values": [
                            float(value)
                            for value in model.controlled_output(equilibrium)
                        ],
                        "kind": "ramp",
                    }
                )
            return schedule

        for at_step in times:
            if selected == "async_step":
                action, target = _sample_async_reference_target(
                    model,
                    action,
                    half_width,
                    rng,
                )
            else:
                action, target = _sample_reference_target(
                    model,
                    action,
                    half_width,
                    rng,
                )
            schedule.append(
                {
                    "at_step": at_step,
                    "values": target,
                    "kind": selected,
                }
            )
        return schedule


class QuadrupleDisturbanceGenerator:
    """Compile bounded disturbance processes into deterministic schedules."""

    def generate(
        self,
        model,
        *,
        episode_steps: int,
        specification: Mapping[str, Any],
        rng: np.random.Generator,
        force_shift: bool | None = None,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        defaults = model.runtime_env(model.disturbance_defaults())
        events = {
            (0, str(name)): {
                "at_step": 0,
                "name": str(name),
                "value": copy.deepcopy(value),
                "kind": "initial",
            }
            for name, value in defaults.items()
        }
        should_shift = (
            bool(force_shift)
            if force_shift is not None
            else float(rng.random()) < float(specification["probability"])
        )
        if not should_shift:
            return _ordered_disturbance_events(events)

        patterns = tuple(str(item) for item in specification["patterns"])
        selected = pattern or patterns[int(rng.integers(0, len(patterns)))]
        if selected not in patterns:
            raise ValueError(
                f"disturbance pattern {selected!r} is not declared"
            )
        dynamic = {
            row["name"]: row
            for row in model.disturbance_schema()
            if row.get("kind") != "setpoint" and row.get("event")
        }
        names = tuple(sorted(dynamic))
        selected_name = names[int(rng.integers(0, len(names)))]
        observability = dict(specification.get("observability", {}))
        maximum_shift = float(specification["max_shift"])
        start = int(
            rng.integers(
                max(1, int(0.15 * episode_steps)),
                max(2, int(0.65 * episode_steps)),
            )
        )

        def add(at_step, name, value, kind=selected):
            row = dynamic[name]
            bounded = _bounded_disturbance_value(value, row.get("bounds"))
            events[(int(at_step), name)] = {
                "at_step": int(at_step),
                "name": name,
                "value": bounded,
                "kind": kind,
                "observability": observability.get(name, "unmeasured"),
            }

        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        magnitude = float(
            rng.uniform(min(0.04, maximum_shift), maximum_shift)
        )
        shifted_value = 1.0 + direction * magnitude
        if selected == "step":
            add(start, selected_name, shifted_value)
        elif selected == "pulse":
            add(start, selected_name, shifted_value)
            restore = min(
                episode_steps - 1,
                start + max(10, int(0.15 * episode_steps)),
            )
            add(restore, selected_name, 1.0, kind="pulse_restore")
        elif selected == "ramp":
            segments = 6
            spacing = max(1, int(0.12 * episode_steps) // segments)
            for index in range(1, segments + 1):
                add(
                    min(episode_steps - 1, start + index * spacing),
                    selected_name,
                    1.0 + direction * magnitude * index / segments,
                )
        elif selected == "bounded_random_walk":
            value = 1.0
            spacing = max(1, int(0.08 * episode_steps))
            for index in range(6):
                value += float(
                    rng.uniform(
                        -maximum_shift / 3.0,
                        maximum_shift / 3.0,
                    )
                )
                add(
                    min(episode_steps - 1, start + index * spacing),
                    selected_name,
                    value,
                )
        elif selected == "colored_noise":
            deviation = 0.0
            spacing = max(1, int(0.05 * episode_steps))
            for index in range(8):
                deviation = (
                    0.8 * deviation
                    + float(rng.normal(0.0, maximum_shift / 3.0))
                )
                deviation = float(
                    np.clip(
                        deviation,
                        -maximum_shift,
                        maximum_shift,
                    )
                )
                add(
                    min(episode_steps - 1, start + index * spacing),
                    selected_name,
                    1.0 + deviation,
                )
        elif selected == "piecewise_regime_shift":
            other_name = next(name for name in names if name != selected_name)
            add(start, selected_name, shifted_value)
            add(
                min(episode_steps - 1, start + int(0.18 * episode_steps)),
                other_name,
                1.0 - direction * 0.75 * magnitude,
            )
        else:
            raise ValueError(
                f"unsupported disturbance pattern {selected!r}"
            )
        return _ordered_disturbance_events(events)


def validate_quadruple_parameters(
    parameters: Mapping[str, Any],
    *,
    expected_regime: str,
    phase_margin: float = 0.05,
) -> None:
    """Reject invalid parameter samples and phase-boundary leakage."""

    model = apply_model_params(make_model("quadruple"), parameters)
    if expected_regime not in {"minimum-phase", "nonminimum-phase"}:
        raise ValueError(
            "expected_regime must be minimum-phase or nonminimum-phase"
        )
    gamma_sum = float(sum(model.p["gamma"]))
    if expected_regime == "minimum-phase":
        valid_phase = gamma_sum >= 1.0 + float(phase_margin)
    else:
        valid_phase = gamma_sum <= 1.0 - float(phase_margin)
    if not valid_phase or model.phase_configuration != expected_regime:
        raise ValueError(
            f"sampled gamma sum {gamma_sum:.6f} violates "
            f"{expected_regime} phase margin {phase_margin}"
        )
    equilibrium = model.equilibrium_state(model.p["nominal_voltage"])
    if not _state_within_bounds(model, equilibrium, margin=0.0):
        raise ValueError("sampled parameters produce infeasible equilibrium")


def validate_quadruple_episode(episode: EpisodeSpec) -> dict[str, Any]:
    """Return physical feasibility checks for one resolved episode."""

    checks = []

    def record(name, passed, detail=""):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": str(detail),
            }
        )

    if episode.scenario_id != "quadruple":
        record("scenario", False, episode.scenario_id)
        return {"passed": False, "checks": checks}
    try:
        model = apply_model_params(
            make_model("quadruple"),
            episode.plant_parameters,
        )
        record("parameter_bounds", True)
    except Exception as exc:
        record("parameter_bounds", False, exc)
        return {"passed": False, "checks": checks}

    record(
        "initial_state_bounds",
        _state_within_bounds(model, episode.initial_state, margin=0.0),
    )
    references_ok = all(
        model.is_setpoint_reachable(event["values"])
        for event in episode.reference_schedule
    )
    record("reference_feasibility", references_ok)

    disturbance_rows = {
        row["name"]: row
        for row in model.disturbance_schema()
        if row.get("kind") != "setpoint"
    }
    disturbances_ok = True
    for event in episode.disturbance_schedule:
        row = disturbance_rows.get(event["name"])
        if row is None:
            continue
        value = float(event["value"])
        bounds = row.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lo, hi = bounds
            disturbances_ok = (
                disturbances_ok
                and (lo is None or value >= float(lo))
                and (hi is None or value <= float(hi))
            )
    record("disturbance_bounds", disturbances_ok)

    initial_reference = episode.reference_schedule[0]["values"]
    action = model.tracking_steady_state_action(initial_reference)
    if action is None:
        record("mass_balance_consistency", False, "singular steady inverse")
    else:
        initial_disturbance = {
            event["name"]: event["value"]
            for event in episode.disturbance_schedule
            if event["at_step"] == 0
        }
        residual = model.mass_balance_residual(
            episode.initial_state,
            action,
            initial_disturbance,
        )
        record(
            "mass_balance_consistency",
            abs(float(residual)) < 1e-9,
            f"residual={residual:.3e}",
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _operating_point_anchors(profile) -> dict[str, dict[str, Any]]:
    parameters = {
        name: copy.deepcopy(row["value"])
        for name, row in profile["parameters"].items()
    }
    anchors = {}
    for regime in ("minimum-phase", "nonminimum-phase"):
        operating = profile["operating_points"][regime]
        values = copy.deepcopy(parameters)
        values["pump_gain"] = copy.deepcopy(
            operating["pump_gain_cm3_per_Vs"]
        )
        values["gamma"] = copy.deepcopy(operating["gamma"])
        values["nominal_voltage"] = copy.deepcopy(
            operating["voltage_V"]
        )
        anchors[regime] = values
    return anchors


def _sample_correlated_parameters(specification, rng):
    anchors = specification["anchors"]
    weights = specification["regime_weights"]
    regimes = tuple(anchors)
    probabilities = np.asarray(
        [float(weights[name]) for name in regimes],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    relative_std = float(specification["relative_std"])
    phase_margin = float(specification["phase_margin"])
    for _ in range(128):
        regime = regimes[int(rng.choice(len(regimes), p=probabilities))]
        parameters = copy.deepcopy(anchors[regime])
        outlet_multiplier = max(
            0.5,
            float(rng.normal(1.0, relative_std)),
        )
        pump_multiplier = max(
            0.5,
            float(rng.normal(1.0, 0.75 * relative_std)),
        )
        gamma_shift = float(rng.normal(0.0, 0.25 * relative_std))
        parameters["outlet_area"] = [
            float(value) * outlet_multiplier
            for value in parameters["outlet_area"]
        ]
        parameters["pump_gain"] = [
            float(value) * pump_multiplier
            for value in parameters["pump_gain"]
        ]
        parameters["gamma"] = [
            float(value) + gamma_shift for value in parameters["gamma"]
        ]
        try:
            validate_quadruple_parameters(
                parameters,
                expected_regime=regime,
                phase_margin=phase_margin,
            )
        except (TypeError, ValueError):
            continue
        return parameters, regime
    raise RuntimeError(
        "could not sample feasible correlated quadruple parameters"
    )


def _sample_equilibrium_initial_state(model, specification, rng):
    maximum = float(model.p["max_level"])
    margin = float(specification["state_safety_margin"])
    nominal_action = np.asarray(
        model.default_action(),
        dtype=np.float64,
    )
    action_half_width = float(
        specification["operating_action_half_width"]
    )
    for _ in range(128):
        action = np.clip(
            nominal_action
            + rng.uniform(
                -action_half_width,
                action_half_width,
                size=nominal_action.shape,
            ),
            0.12,
            0.65,
        )
        equilibrium = model.equilibrium_state(
            model.physical_action_vector(action)
        )
        if _state_within_bounds(model, equilibrium, margin=margin):
            break
    else:
        raise RuntimeError("could not sample a safe operating equilibrium")

    relative_std = float(specification["relative_std"])
    correlation = np.asarray(
        specification["correlation"],
        dtype=np.float64,
    )
    correlated = np.linalg.cholesky(correlation) @ rng.normal(size=4)
    state = np.asarray(equilibrium, dtype=np.float64) * (
        1.0 + relative_std * correlated
    )
    state = np.clip(state, margin, maximum - margin)
    mode = "equilibrium-perturbation"
    if float(rng.random()) < float(specification["recovery_probability"]):
        index = int(rng.integers(0, len(state)))
        if int(rng.integers(0, 2)) == 0:
            state[index] = max(
                margin,
                float(equilibrium[index])
                * float(rng.uniform(0.25, 0.55)),
            )
            mode = "recovery-low"
        else:
            room = maximum - margin - float(equilibrium[index])
            state[index] = min(
                maximum - margin,
                float(equilibrium[index])
                + float(rng.uniform(0.4, 0.75)) * max(room, 0.0),
            )
            mode = "recovery-high"
    return (
        [float(value) for value in state],
        [float(value) for value in equilibrium],
        [float(value) for value in action],
        mode,
    )


def _sample_reference_target(model, anchor_action, half_width, rng):
    anchor = np.asarray(anchor_action, dtype=np.float64)
    maximum = float(model.p["max_level"])
    for _ in range(128):
        candidate = np.clip(
            anchor
            + rng.uniform(
                -half_width,
                half_width,
                size=anchor.shape,
            ),
            0.12,
            0.65,
        )
        equilibrium = model.equilibrium_state(
            model.physical_action_vector(candidate)
        )
        target = [
            float(value) for value in model.controlled_output(equilibrium)
        ]
        change = max(
            abs(value - previous)
            for value, previous in zip(
                target,
                model.controlled_output(
                    model.equilibrium_state(
                        model.physical_action_vector(anchor)
                    )
                ),
            )
        )
        if (
            change >= 0.1
            and max(equilibrium) <= maximum - 0.25
            and min(equilibrium) >= 0.25
            and model.is_setpoint_reachable(target)
        ):
            return candidate, target
    equilibrium = model.equilibrium_state(
        model.physical_action_vector(anchor)
    )
    return anchor, [
        float(value) for value in model.controlled_output(equilibrium)
    ]


def _sample_async_reference_target(model, anchor_action, half_width, rng):
    anchor = np.asarray(anchor_action, dtype=np.float64)
    equilibrium = model.equilibrium_state(
        model.physical_action_vector(anchor)
    )
    current = [
        float(value) for value in model.controlled_output(equilibrium)
    ]
    output_index = int(rng.integers(0, len(current)))
    maximum_change = max(0.2, 20.0 * float(half_width))
    for _ in range(128):
        candidate = list(current)
        delta = float(
            rng.uniform(-maximum_change, maximum_change)
        )
        if abs(delta) < 0.1:
            continue
        candidate[output_index] = float(
            np.clip(candidate[output_index] + delta, 0.5, 19.5)
        )
        if not model.is_setpoint_reachable(candidate):
            continue
        action = model.tracking_steady_state_action(candidate)
        if action is not None:
            return np.asarray(action, dtype=np.float64), candidate
    return _sample_reference_target(
        model,
        anchor,
        half_width,
        rng,
    )


def _event_times(count, episode_steps, minimum_dwell, rng):
    count = max(1, int(count))
    minimum_dwell = max(1, int(minimum_dwell))
    maximum_dwell = max(1, (episode_steps - 1) // (count + 1))
    minimum_dwell = min(minimum_dwell, maximum_dwell)
    earliest_first = minimum_dwell
    latest_first = (
        episode_steps - 1 - count * minimum_dwell
    )
    if latest_first > earliest_first:
        first = int(
            rng.integers(earliest_first, latest_first + 1)
        )
    else:
        first = earliest_first
    times = [first]
    for index in range(1, count):
        minimum = times[-1] + minimum_dwell
        remaining = count - index - 1
        maximum = (
            episode_steps
            - 1
            - minimum_dwell
            - remaining * minimum_dwell
        )
        times.append(
            int(rng.integers(minimum, maximum + 1))
            if maximum > minimum
            else minimum
        )
    return times


def _bounded_disturbance_value(value, bounds):
    number = float(value)
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lo, hi = bounds
        if lo is not None:
            number = max(number, float(lo))
        if hi is not None:
            number = min(number, float(hi))
    return float(number)


def _ordered_disturbance_events(events):
    return [
        events[key]
        for key in sorted(events, key=lambda item: (item[0], item[1]))
    ]


def _state_within_bounds(model, state, *, margin):
    for value, row in zip(state, model.state_schema()):
        number = float(value)
        if not math.isfinite(number):
            return False
        bounds = row.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        lo, hi = bounds
        if lo is not None and number < float(lo) + float(margin):
            return False
        if hi is not None and number > float(hi) - float(margin):
            return False
    return True


__all__ = [
    "QuadrupleCurriculumSampler",
    "QuadrupleDisturbanceGenerator",
    "QuadrupleReferenceGenerator",
    "QuadrupleTrainingSampler",
    "quadruple_training_distribution",
    "validate_quadruple_episode",
    "validate_quadruple_parameters",
]
