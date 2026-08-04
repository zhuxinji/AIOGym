"""Internal Gymnasium-native process-control environment implementation.

Fast, synchronous, seedable, and vectorizable for benchmark evaluation,
offline-data generation, and online RL training.

Contract:
  generic obs    = [x, y_sp, disturbances]
  generic action = u in [0, 1]

reward_spec:
  "regulation-v1"      (default) provides the canonical time-consistent
                       regulation cost.
  "economic-v1"        provides canonical value/cost decomposition.

``auto_events`` controls generic automatically generated within-episode events; it
does not enable or disable the process model's physical dynamics. The model is
integrated on every step for both values. ``auto_events=True`` can inject setpoint
steps, cold-inlet steps, ambient drift, or demand surges on top of domain-
randomised start points. Named cases normally set it to ``False`` and declare
their own deterministic event schedules. The policy observes changed conditions
(t_cold / t_amb / setpoints are all in obs).
"""
from __future__ import annotations
import copy
from time import perf_counter

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aiogym.models.integration import Integrator
from aiogym.models.registry import apply_model_params, make_model

from .config import (
    validated_range as _validated_range,
)
from .spec import ENV_SPEC_HASH_SCHEMA_VERSION, ResolvedEnvSpec
from .disturbances import DisturbanceRuntimeMixin
from .observations import ObservationRuntimeMixin
from .realism import (
    ActuatorModelRuntime,
    SensorModelRuntime,
)
from .transitions import TransitionRuntimeMixin


def _build_configured_model(spec, model_params, case_profile):
    model = apply_model_params(make_model(spec.scenario), model_params)
    if case_profile is not None:
        from aiogym.models.cases import configure_model_for_case

        configure_model_for_case(model, case_profile)
    return model


def _resolve_setpoint_events(case_profile, initial_setpoint, schedule):
    case_profile = case_profile or {}
    initial_state = copy.deepcopy(
        case_profile.get("initialization", {}).get("state")
    )
    case_setpoints = case_profile.get("setpoints", {})
    initial = (
        case_setpoints.get("initial")
        if initial_setpoint is None
        else initial_setpoint
    )
    if initial is not None:
        if isinstance(initial, (str, bytes)) or not isinstance(initial, (list, tuple)):
            raise TypeError("initial_setpoint must be a numeric vector")
        initial = [float(value) for value in initial]
        if not all(np.isfinite(value) for value in initial):
            raise ValueError("initial_setpoint values must be finite")
    raw_schedule = case_setpoints.get("schedule", []) if schedule is None else schedule
    if not isinstance(raw_schedule, (list, tuple)):
        raise TypeError("setpoint_schedule must be a list of event mappings")
    events = {}
    for event in raw_schedule:
        if not isinstance(event, dict):
            raise TypeError("each setpoint_schedule event must be a mapping")
        at_step = event.get("at_step")
        if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
            raise ValueError("setpoint_schedule at_step must be a non-negative integer")
        values = event.get("values")
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise TypeError("setpoint_schedule values must be a numeric vector")
        values = [float(value) for value in values]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("setpoint_schedule values must be finite")
        if at_step in events:
            raise ValueError("setpoint_schedule cannot contain duplicate at_step values")
        events[at_step] = values
    return initial_state, copy.deepcopy(initial), events


def _resolve_disturbance_events(case_profile):
    resolved = {}
    for event in (case_profile or {}).get("disturbances", []):
        resolved.setdefault(int(event["at_step"]), []).append(
            {"name": str(event["name"]), "value": copy.deepcopy(event["value"])}
        )
    return resolved


def _validated_integral_limits(model, output_dimension):
    limits = tuple(float(value) for value in model.integral_observation_limits())
    if len(limits) != output_dimension:
        raise ValueError(
            "integral observation limits must match the controlled-output "
            f"dimension: {len(limits)} != {output_dimension}"
        )
    if not all(np.isfinite(value) and value > 0.0 for value in limits):
        raise ValueError("integral observation limits must be finite and positive")
    return limits


def _normalized_space_bounds(schema, *, delta):
    lows = []
    highs = []
    for row in schema:
        bounds = row.get("bounds") if isinstance(row, dict) else None
        valid = (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and bounds[0] is not None
            and bounds[1] is not None
            and np.isfinite(float(bounds[0]))
            and np.isfinite(float(bounds[1]))
            and float(bounds[1]) > float(bounds[0])
        )
        if valid:
            lows.append(-1.0 if delta else 0.0)
            highs.append(1.0)
        else:
            lows.append(-np.inf)
            highs.append(np.inf)
    return lows, highs


# ---- Environment wrapper ----
class _AIOGymEnv(
    DisturbanceRuntimeMixin,
    ObservationRuntimeMixin,
    TransitionRuntimeMixin,
    gym.Env,
):
    metadata = {"render_modes": []}

    def __init__(self, spec: ResolvedEnvSpec):
        super().__init__()
        if not isinstance(spec, ResolvedEnvSpec):
            raise TypeError("_AIOGymEnv requires a ResolvedEnvSpec")
        self.env_spec = spec
        runtime = spec.runtime()
        self.case_profile = runtime["case_profile"]
        observation = runtime["observation"]
        realism = runtime["realism"]
        events = runtime["events"]
        environment_options = {
            **observation,
            **realism,
            "control_dt": spec.control_dt,
            "episode_steps": spec.episode_steps,
            "action_mode": spec.action_mode,
            "model_params": runtime["model_params"],
        }
        initial_setpoint = events["initial_setpoint"]
        setpoint_schedule = events["setpoint_schedule"]
        profile_timing = spec.profile_timing
        info_level = spec.info_level
        reward_spec = spec.reward_spec
        crystal_ln_sp = realism["crystal_ln_sp"]
        crystal_cv_sp = realism["crystal_cv_sp"]
        crystal_random_targets = realism["crystal_random_targets"]
        crystal_ln_range = realism["crystal_ln_range"]
        crystal_cv_range = realism["crystal_cv_range"]
        self.scenario = spec.scenario
        self.model = _build_configured_model(
            spec, environment_options["model_params"], self.case_profile
        )
        (
            self._case_initial_state,
            self._case_initial_setpoint,
            self._case_setpoint_events,
        ) = _resolve_setpoint_events(
            self.case_profile, initial_setpoint, setpoint_schedule
        )
        self._episode_setpoint_events = copy.deepcopy(self._case_setpoint_events)
        self._case_disturbance_events = _resolve_disturbance_events(self.case_profile)
        auto_events = environment_options["auto_events"]
        randomize = environment_options["randomize"]
        randomize_setpoints = environment_options["randomize_setpoints"]
        randomize_plant = environment_options["randomize_plant"]
        plant_drift = environment_options["plant_drift"]
        integral_obs = environment_options["integral_obs"]
        disturbance_obs = environment_options["disturbance_obs"]
        previous_action_obs = environment_options["previous_action_obs"]
        normalize_observations = environment_options["normalize_observations"]
        tracking_error_obs = environment_options["tracking_error_obs"]
        observation_mode = environment_options["observation_mode"]
        action_mode = environment_options["action_mode"]
        noise = environment_options["noise"]
        terminate_on_runaway = environment_options["terminate_on_runaway"]
        self.control_dt = environment_options["control_dt"]
        self.episode_steps = environment_options["episode_steps"]
        self.reward_spec = reward_spec
        if action_mode not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be one of: actuator, setpoint")
        self.noise_pct = environment_options["noise_pct"]
        self.reward_spec_id = self.reward_spec.id
        self.reward_spec_hash = self.reward_spec.spec_hash
        self.env_spec_hash = spec.spec_hash
        self.env_mdp_hash = spec.mdp_hash
        self.env_runtime_hash = spec.runtime_hash
        self.env_spec_hash_schema = ENV_SPEC_HASH_SCHEMA_VERSION
        self.goal = self.reward_spec.goal
        if info_level not in {"minimal", "full"}:
            raise ValueError("info_level must be one of: minimal, full")
        self.info_level = str(info_level)
        self.profile_timing = bool(profile_timing)
        self.last_step_timings = {}
        self.auto_events = auto_events
        self.randomize_plant = randomize_plant    # per-episode operating-regime variation
        self.plant_drift = plant_drift            # slow within-episode parameter drift
        self.randomize = randomize
        self.randomize_setpoints = randomize_setpoints
        self.noise = noise                        # measurement noise on observed levels/temps
        # Custom stage reward hooks are intentionally outside the stable env API.
        self.custom_stage_reward = None
        self.reward_spec_is_canonical = bool(self.reward_spec.canonical)
        self.terminate_on_runaway = terminate_on_runaway
        self.crystal_ln_sp = crystal_ln_sp
        self.crystal_cv_sp = crystal_cv_sp
        self.crystal_random_targets = bool(crystal_random_targets)
        self.crystal_ln_range = _validated_range("crystal_ln_range", crystal_ln_range)
        self.crystal_cv_range = _validated_range("crystal_cv_range", crystal_cv_range)
        self._model_env_options = {
            "randomize_setpoints": bool(randomize_setpoints),
            "crystal_ln_sp": crystal_ln_sp,
            "crystal_cv_sp": crystal_cv_sp,
            "crystal_random_targets": bool(crystal_random_targets),
            "crystal_ln_range": self.crystal_ln_range,
            "crystal_cv_range": self.crystal_cv_range,
        }

        self._p_nominal = {k: (list(v) if isinstance(v, list) else v) for k, v in self.model.p.items()}
        self._regime = copy.deepcopy(getattr(self.model, "plant_regime", {}))
        self._econ = copy.deepcopy(getattr(self.model, "economic_config", {}))
        self._econ_nominal = copy.deepcopy(self._econ)
        self._disturbance_defaults = self.model.runtime_env(self.model.disturbance_defaults())
        self._disturbance_attrs = self.model.disturbance_attribute_map()
        self._disturbance_by_event = {
            row["event"]: row
            for row in self.model.disturbance_schema()
            if row.get("event")
        }
        self._disturbance_schema_by_name = {
            row["name"]: row for row in self.model.disturbance_schema() if row.get("name")
        }
        for events in self._case_disturbance_events.values():
            for event in events:
                self._validate_case_disturbance(event["name"], event["value"])
        self._reset_disturbance_values()
        self.integ = Integrator(self.model)
        self.nu = self.model.action_dim()
        y_sp = list(
            self._case_initial_setpoint
            if self._case_initial_setpoint is not None
            else self.model.env_setpoint_vector(self._model_env_options)
        )
        if len(y_sp) != len(self.model.controlled_output(self.model.initial_state())):
            raise ValueError(
                f"case setpoint length {len(y_sp)} does not match controlled-output length "
                f"{len(self.model.controlled_output(self.model.initial_state()))}"
            )
        if self._case_initial_state is not None and len(self._case_initial_state) != len(self.model.initial_state()):
            raise ValueError(
                f"case initial-state length {len(self._case_initial_state)} does not match state length "
                f"{len(self.model.initial_state())}"
            )
        for at_step, values in self._case_setpoint_events.items():
            if len(values) != len(y_sp):
                raise ValueError(
                    f"case setpoint event at step {at_step} has {len(values)} values; expected {len(y_sp)}"
                )
        self._ysp0 = list(y_sp)

        self._integral_limits = _validated_integral_limits(
            self.model, len(self._ysp0)
        )

        self._configure_spaces(
            integral_obs=integral_obs,
            disturbance_obs=disturbance_obs,
            previous_action_obs=previous_action_obs,
            normalize_observations=normalize_observations,
            tracking_error_obs=tracking_error_obs,
            observation_mode=observation_mode,
            action_mode=action_mode,
        )
        self._k = 0
        self.last_act = self.model.action_vector(self.model.default_action())
        self.last_commanded_act = copy.deepcopy(self.last_act)
        self.previous_act = copy.deepcopy(self.last_act)
        self._initialize_realism_runtimes()

    # ---- helpers ----
    def _configure_spaces(
        self,
        *,
        integral_obs,
        disturbance_obs,
        previous_action_obs,
        normalize_observations,
        tracking_error_obs,
        observation_mode,
        action_mode,
    ):
        self.integral_obs = integral_obs
        observation_flags = {
            "disturbance_obs": disturbance_obs,
            "previous_action_obs": previous_action_obs,
            "normalize_observations": normalize_observations,
            "tracking_error_obs": tracking_error_obs,
        }
        for name, value in observation_flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
            setattr(self, name, value)
        self.observation_mode = str(observation_mode)
        obs_dim = (
            len(self.model.initial_state())
            if self.observation_mode == "full_state"
            else len(self._ysp0)
        ) + len(self._ysp0)
        if self.disturbance_obs:
            obs_dim += len(self.model.dynamics_disturbance_names())
        if self.previous_action_obs:
            obs_dim += self.nu
        if integral_obs and self.model.supports_integral_observation:
            obs_dim += len(self._ysp0)
        self.action_mode = action_mode
        self.layout = (
            list(getattr(self.model, "supervisory_layout", ()))
            if action_mode == "setpoint"
            else None
        )
        if action_mode == "setpoint" and not self.layout:
            raise ValueError(
                f"setpoint action_mode has no supervisory layout for '{self.scenario}'"
            )
        if self.layout is not None:
            from aiogym.controllers.pid import PIDAgent

            self.pid = PIDAgent(self.model)
            act_dim = len(self.layout)
        else:
            self.pid = None
            act_dim = self.nu
        self.action_space = spaces.Box(0.0, 1.0, (act_dim,), dtype=np.float32)
        observation_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        observation_high = np.full(obs_dim, np.inf, dtype=np.float32)
        # Cascade v2 freezes fixed-bounds normalization as part of its policy
        # contract. Other v1 scenarios retain their historical unbounded Box
        # metadata even when their observation values are normalized.
        if self.normalize_observations and self.scenario == "cascade":
            state_schema = (
                self.model.state_schema()
                if self.observation_mode == "full_state"
                else self.model.setpoint_schema()
            )
            lows, highs = _normalized_space_bounds(
                state_schema,
                delta=False,
            )
            reference_low, reference_high = _normalized_space_bounds(
                self.model.setpoint_schema(),
                delta=self.tracking_error_obs,
            )
            lows.extend(reference_low)
            highs.extend(reference_high)
            if self.disturbance_obs:
                disturbance_schema = {
                    row.get("name"): row
                    for row in self.model.disturbance_schema()
                }
                names = self.model.dynamics_disturbance_names()
                extra_low, extra_high = _normalized_space_bounds(
                    [disturbance_schema.get(name, {}) for name in names],
                    delta=False,
                )
                lows.extend(extra_low)
                highs.extend(extra_high)
            if self.previous_action_obs:
                extra_low, extra_high = _normalized_space_bounds(
                    self.model.action_schema(),
                    delta=False,
                )
                lows.extend(extra_low)
                highs.extend(extra_high)
            observation_low[:] = lows
            observation_high[:] = highs
        if integral_obs and self.model.supports_integral_observation:
            observation_low[-len(self._ysp0) :] = -1.0
            observation_high[-len(self._ysp0) :] = 1.0
        self.observation_space = spaces.Box(
            observation_low,
            observation_high,
            dtype=np.float32,
        )

    def _initialize_realism_runtimes(self):
        self._episode_spec = None
        self._episode_noise_enabled = self.noise
        self._episode_noise_pct = self.noise_pct
        self._episode_plant_drift_enabled = self.plant_drift
        self._episode_disturbance_events = copy.deepcopy(
            self._case_disturbance_events
        )
        self._sensor_runtime = None
        self._actuator_runtime = None
        self._active_sensor_model = {"kind": "identity"}
        self._active_actuator_model = {"kind": "identity"}

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        reset_options = dict(options or {})
        episode_spec = reset_options.get("episode_spec")
        if episode_spec is not None:
            return self._reset_from_episode_spec(
                episode_spec,
                seed=seed,
            )
        super().reset(seed=seed)
        self._episode_spec = None
        self._episode_noise_enabled = self.noise
        self._episode_noise_pct = self.noise_pct
        self._active_sensor_model = (
            {
                "kind": "additive_gaussian",
                "noise_pct": self.noise_pct,
            }
            if self.noise
            else {"kind": "identity"}
        )
        self._active_actuator_model = {"kind": "identity"}
        self._episode_plant_drift_enabled = self.plant_drift
        self._episode_disturbance_events = copy.deepcopy(
            self._case_disturbance_events
        )
        self._econ = copy.deepcopy(self._econ_nominal)
        seed_bundle = dict(reset_options.get("seed_bundle") or {})
        if seed_bundle:
            required_seed_components = (
                "initial",
                "reference",
                "disturbance",
                "noise",
                "plant",
            )
            missing = [
                name
                for name in required_seed_components
                if name not in seed_bundle
            ]
            if missing:
                raise ValueError(
                    "seed_bundle is missing component seeds: "
                    + ", ".join(missing)
                )
            self._initial_rng = np.random.default_rng(
                int(seed_bundle["initial"])
            )
            self._reference_rng = np.random.default_rng(
                int(seed_bundle["reference"])
            )
            self._disturbance_rng = np.random.default_rng(
                int(seed_bundle["disturbance"])
            )
            self._noise_rng = np.random.default_rng(
                int(seed_bundle["noise"])
            )
            self._plant_rng = np.random.default_rng(
                int(seed_bundle["plant"])
            )
        else:
            self._initial_rng = self.np_random
            self._reference_rng = self.np_random
            self._disturbance_rng = self.np_random
            self._noise_rng = self.np_random
            self._plant_rng = self.np_random
        rng = self._initial_rng
        self._restore_nominal()
        self._init_regime_state()
        if self.randomize_plant:
            self._apply_regime()        # this episode's operating regime
            self._regime_target = self._sample_regime_mult()
        elif self.plant_drift:
            self._regime_target = self._sample_regime_mult()
        x0 = list(
            self._case_initial_state
            if self._case_initial_state is not None
            else self.model.initial_state()
        )
        self.y_sp = list(self._ysp0)
        self._episode_setpoint_events = copy.deepcopy(self._case_setpoint_events)
        self._reset_disturbance_values()
        if self.randomize:
            for j in range(len(x0)):
                x0[j] *= 1.0 + 0.08 * float(rng.uniform(-1, 1))
            if self.model.randomize_common_temperatures:
                if "t_cold" in self._disturbance_values:
                    base = float(self._disturbance_defaults["t_cold"])
                    self._set_disturbance_value("t_cold", float(np.clip(base + rng.uniform(-5, 5), 2, 35)))
                if "t_amb" in self._disturbance_values:
                    base = float(self._disturbance_defaults["t_amb"])
                    self._set_disturbance_value("t_amb", float(np.clip(base + rng.uniform(-5, 8), 0, 40)))
        self._sync_known_disturbances()
        initial_sampling_options = {
            **self._model_env_options,
            # A case with reference events keeps its initial equilibrium SP and
            # samples the episode's scheduled targets instead.
            "randomize_setpoints": (
                self.randomize_setpoints
                and not self._episode_setpoint_events
            ),
        }
        self.y_sp = self.model.sample_env_setpoints(
            self.y_sp,
            self._reference_rng,
            initial_sampling_options,
        )
        if self.randomize_setpoints and self._episode_setpoint_events:
            event_sampling_options = {
                **self._model_env_options,
                "randomize_setpoints": True,
            }
            self._episode_setpoint_events = {
                at_step: self.model.sample_env_setpoints(
                    values,
                    self._reference_rng,
                    event_sampling_options,
                )
                for at_step, values in self._episode_setpoint_events.items()
            }
        self.integ.reset(x0)
        if self.pid is not None:
            self.pid.reset()
        self._iy = [0.0] * len(self.y_sp)
        self._k = 0
        self.last_act = self.model.action_vector(self.model.default_action())
        self.last_commanded_act = copy.deepcopy(self.last_act)
        self.previous_act = copy.deepcopy(self.last_act)
        self._reset_realism(x0)
        self._schedule_disturbances()
        # A case event at t=0 is part of the initial controller context. Applying
        # it before the first observation avoids an artificial one-sample delay
        # in paper-style reference-step experiments.
        if 0 in self._episode_setpoint_events:
            self.y_sp = list(self._episode_setpoint_events[0])
        for event in self._episode_disturbance_events.get(0, []):
            self._set_disturbance_value(event["name"], event["value"])
        info = (
            {"seed_bundle": copy.deepcopy(seed_bundle)}
            if seed_bundle
            else {}
        )
        info.update(self._episode_provenance())
        return self._obs(), info

    def _reset_from_episode_spec(self, episode_spec, *, seed=None):
        from aiogym.generation.specs import EpisodeSpec
        from aiogym.generation.validation import validate_episode_for_env

        resolved = (
            episode_spec
            if isinstance(episode_spec, EpisodeSpec)
            else EpisodeSpec(episode_spec)
        )
        validate_episode_for_env(resolved, self)
        if seed is not None and int(seed) != resolved.base_seed:
            raise ValueError(
                "reset seed must match EpisodeSpec base_seed when an "
                "EpisodeSpec is injected"
            )
        super().reset(
            seed=resolved.base_seed if seed is None else int(seed)
        )
        seeds = resolved.component_seeds
        self._initial_rng = np.random.default_rng(seeds["initial_state"])
        self._reference_rng = np.random.default_rng(seeds["reference"])
        self._disturbance_rng = np.random.default_rng(seeds["disturbance"])
        self._noise_rng = np.random.default_rng(seeds["sensor"])
        self._plant_rng = np.random.default_rng(seeds["plant"])

        self._episode_spec = resolved
        self._restore_nominal()
        self._init_regime_state()
        for name, value in resolved.plant_parameters.items():
            self.model.p[name] = copy.deepcopy(value)
        self._episode_plant_drift_enabled = False

        sensor_model = resolved.sensor_model
        self._episode_noise_enabled = (
            sensor_model.get("kind") == "additive_gaussian"
        )
        self._episode_noise_pct = float(
            sensor_model.get("noise_pct", 0.0)
        )
        self._active_sensor_model = sensor_model
        self._active_actuator_model = resolved.actuator_model
        self._econ = (
            resolved.economic_context
            if resolved.economic_context
            else copy.deepcopy(self._econ_nominal)
        )

        self._episode_setpoint_events = {
            int(event["at_step"]): list(event["values"])
            for event in resolved.reference_schedule
        }
        self.y_sp = list(self._episode_setpoint_events[0])
        self._episode_disturbance_events = {}
        for event in resolved.disturbance_schedule:
            self._episode_disturbance_events.setdefault(
                int(event["at_step"]),
                [],
            ).append(
                {
                    "name": str(event["name"]),
                    "value": copy.deepcopy(event["value"]),
                }
            )

        self._reset_disturbance_values()
        for event in self._episode_disturbance_events.get(0, []):
            self._set_disturbance_value(event["name"], event["value"])
        self._sync_known_disturbances()
        self.integ.reset(resolved.initial_state)
        if self.pid is not None:
            self.pid.reset()
        self._iy = [0.0] * len(self.y_sp)
        self._k = 0
        self.last_act = self.model.action_vector(
            self.model.default_action()
        )
        self.last_commanded_act = copy.deepcopy(self.last_act)
        self.previous_act = copy.deepcopy(self.last_act)
        self._reset_realism(resolved.initial_state)
        self._dist_events = []
        return self._obs(), {
            **self._episode_provenance(),
            "episode_spec": resolved.as_dict(),
        }

    def _episode_provenance(self):
        provenance = {
            "reward_spec_id": self.reward_spec_id,
            "reward_spec_hash": self.reward_spec_hash,
            "env_spec_hash": self.env_spec_hash,
            "env_spec_hash_schema": self.env_spec_hash_schema,
        }
        if self._episode_spec is None:
            return provenance
        provenance.update({
            "episode_spec_id": self._episode_spec.episode_spec_id,
            "episode_spec_hash": self._episode_spec.resolved_hash,
            "distribution_id": self._episode_spec.distribution_id,
            "distribution_hash": self._episode_spec.distribution_hash,
            "episode_base_seed": self._episode_spec.base_seed,
            "episode_component_seeds": (
                self._episode_spec.component_seeds
            ),
        })
        return provenance

    def _reset_realism(self, initial_state):
        self._sensor_runtime = SensorModelRuntime(
            self._active_sensor_model,
            state_schema=self.model.state_schema(),
            control_dt=self.control_dt,
            rng=self._noise_rng,
        )
        self._sensor_runtime.reset(initial_state)
        self._actuator_runtime = ActuatorModelRuntime(
            self._active_actuator_model,
            dimension=self.nu,
            control_dt=self.control_dt,
            low=np.zeros(self.nu, dtype=np.float64),
            high=np.ones(self.nu, dtype=np.float64),
        )
        self.last_act = self.model.action_vector(
            self._actuator_runtime.reset(self.last_act)
        )

    def step(self, action):
        step_started = perf_counter() if self.profile_timing else None
        action = self._validated_action(action)
        commanded_act = (
            self._supervise(action)
            if self.pid is not None
            else self._split(action)
        )
        act, actuator_audit = self._actuator_runtime.apply(commanded_act)
        act = self.model.action_vector(act)
        state = list(self.integ.x)
        self.last_commanded_act = self.model.action_vector(commanded_act)
        self.last_act = act
        for (t, event) in self._dist_events:
            if t == self._k:
                self._apply_disturbance(event)
        for event in self._episode_disturbance_events.get(self._k, []):
            self._set_disturbance_value(event["name"], event["value"])
        self._apply_plant_drift()
        integration_started = perf_counter() if self.profile_timing else None
        self.integ.step(self.control_dt, act, self._env())
        if self.profile_timing:
            integration_seconds = perf_counter() - integration_started
        self._accumulate_integral()
        self._k += 1
        reward_started = perf_counter() if self.profile_timing else None
        reward, terminated, info = self._reward_done(state, act)
        info.update(actuator_audit)
        info["action_commanded_physical"] = [
            float(value)
            for value in self.model.physical_action_vector(commanded_act)
        ]
        info["action_applied_physical"] = [
            float(value)
            for value in self.model.physical_action_vector(act)
        ]
        if self.profile_timing:
            reward_seconds = perf_counter() - reward_started
        info_started = perf_counter() if self.profile_timing else None
        info.update(self._episode_provenance())
        if self.info_level == "minimal":
            info = self._minimal_step_info(info)
        if self.profile_timing:
            info_seconds = perf_counter() - info_started
        self.previous_act = copy.deepcopy(act)
        # Stage the next step's scheduled reference before returning its
        # observation. The controller therefore sees an event at step k before
        # selecting u_k, while the transition just completed is still scored
        # against the reference that was active when its action was selected.
        if self._k in self._episode_setpoint_events:
            self.y_sp = list(self._episode_setpoint_events[self._k])
        truncated = self._k >= self.episode_steps
        observation_started = perf_counter() if self.profile_timing else None
        observation = self._obs()
        if self.profile_timing:
            observation_seconds = perf_counter() - observation_started
            self.last_step_timings = {
                "ode_integration_seconds": integration_seconds,
                "reward_metric_seconds": reward_seconds,
                "info_construction_seconds": info_seconds,
                "observation_seconds": observation_seconds,
                "total_seconds": perf_counter() - step_started,
            }
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _minimal_step_info(info):
        """Keep trainer inputs and provenance without diagnostic payloads."""

        keep = {
            "reward_terms",
            "costs",
            "termination_reason",
            "goal",
            "reward_spec_id",
            "reward_spec_hash",
            "env_spec_hash",
            "env_spec_hash_schema",
            "episode_spec_id",
            "episode_spec_hash",
            "distribution_id",
            "distribution_hash",
            "episode_base_seed",
            "action_commanded_physical",
            "action_applied_physical",
            "actuator_model_kind",
            "actuator_intervened",
            "actuator_command_applied_delta",
            "actuator_command_applied_l1",
        }
        return {
            name: copy.deepcopy(value)
            for name, value in info.items()
            if name in keep
        }

    def render(self):
        pass
