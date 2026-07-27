# Physical-model infrastructure

This layer prepares AIO-Gym for physically grounded scenario additions and
replacements without silently changing existing benchmark models.

## Design boundary

The executable dynamics remain in `aiogym.models.scenarios`. Parameter profiles
and task profiles are versioned, read-only declarations. Loading either profile
does not mutate `model.p`, and the current equations and controller tuning remain
unchanged.

The runtime contract has one numerical path for each concept:

- `dynamics(x, u, d)` is the only public state-derivative entry point;
- `u` is always a flat numeric vector whose indices are defined by
  `action_schema()`;
- actuator names and physical kinds are metadata, not alternate action-dict
  containers;
- `outputs(x)["y"]` is the controlled-output result and should be reused when
  the full output mapping has already been computed.

Custom models should implement `_dynamics(x, u, d, ops)` when they need both
numeric and CasADi execution, or override `dynamics()` for a numeric-only model.

Profiles that have not been migrated are marked `legacy-unverified`. This status
is an explicit warning that their numerical values have not yet been accepted as
reference-quality physical parameters. `quadruple` is the first migrated model
and is marked `reference-parameterized`.

`cascade-recirculating` demonstrates the separate-scenario path for a partially
documented physical proposal: source-supported topology is retained, every
unknown numerical parameter receives an explicit provisional status, and the
original open `cascade` benchmark remains unchanged.

## Parameter profiles

Files under `aiogym/models/parameters/` reserve one profile per scenario for:

- parameter source and source locator;
- units, valid ranges, and uncertainty;
- nominal operating point and model validity domain;
- fidelity status and references;
- numerical solver method and maximum integration step.

Profiles enrich structured model metadata but do not change numerical values. Runtime parameter
changes still require the existing explicit `model_params` mechanism.

```python
import aiogym

profile = aiogym.load_parameter_profile("quadruple")
model = aiogym.make_model("quadruple")
assert profile["status"] == "reference-parameterized"
assert model.p == aiogym.make_model("quadruple").p
```

## Scenario, task, and objective

The three concepts are deliberately separate, but they do not duplicate one
another:

- **scenario** selects the process model and its state/action semantics;
- **task** selects the experiment on that process: parameters or operating
  point, initialization, setpoint/disturbance schedule, constraints, timing,
  noise, and randomization;
- **objective** selects how the same task rollout is scored and reported, such
  as tracking, KPI, robustness, or safety.

Therefore `minimum-phase` and `nonminimum-phase` are tasks, not
objectives. A task may list several `supported_objectives`; changing the
objective does not silently change its plant or disturbance conditions.

Files under `aiogym/models/tasks/builtin/<scenario>/` are the versioned task
declarations. Benchmark rows record `scenario`, `task`, `objective`, the task
status, and a task-profile hash. Leaderboard ranks restart for every
`(scenario, task, objective)` tuple.

```python
env = aiogym.make_env("quadruple", task="minimum-phase")
protocol = aiogym.BenchmarkProtocol.tracking(
    "quadruple",
    task="minimum-phase",
)
```

Objective priority is runtime/API override, case config, suite config, then
`task.default_objective`. If none is available, protocol construction fails with
a clear error. Resolved protocol metadata records
`objective_source`, reward mode, primary metric, and ranking direction.

With no task, direct environments remain at 0.5 s and 600 steps; benchmark
protocols remain at 0.5 s and 400 steps. No-task benchmark conditions are now
neutral and deterministic rather than changing with the selected objective.
Named suites declare any noise, randomization, or plant drift explicitly.

Suites can use a Cartesian matrix or declare task-aware cases:

```json
{
  "cases": [
    {
      "scenario": "quadruple",
      "task": "minimum-phase",
      "objective": "tracking",
      "controllers": ["pid", "mpc"]
    }
}
```

Internally, each expanded suite entry becomes a `BenchmarkCase` containing an
`EnvironmentSpec`, an `ObjectiveSpec`, controller configuration, and seeds.
Environment construction is injectable, so the runner no longer has to create
an `AIOGymNativeEnv` through the protocol itself.

Results distinguish `execution_status` (`passed`, `degraded`, or `failed`) from
`objective_status` (`met`, `not-met`, or `not-defined`).

Controller parameters remain in controller `profile` files. A task may bind
controller IDs to default profile names through its `controllers` mapping.
`BenchmarkCase` resolves that mapping for single-task, suite, and Python API
runs alike, so changing the entry point does not silently change controller
tuning. Explicit controller configuration can still override the task default.

Task setpoint and disturbance events are executable. Events at step zero are
applied before the initial controller observation; later disturbance events are
applied before process integration at their declared step and are recorded in
the result schedule. Event names must exist in the model disturbance schema and
values must respect its bounds.

The environment calls `model.sample_env_setpoints(...)` when seeded target
randomization is requested. `ProcessModelContract` supplies bounded relative
sampling and `is_setpoint_reachable(...)` validation. A custom model with an
analytic steady-state calculation should override `is_setpoint_reachable` to
reject targets that satisfy metadata bounds but cannot be maintained by valid
actuators. Models with a singular target inverse may override
`sample_env_setpoints` and sample a steady actuator/state before mapping it to
controlled outputs.

## Numerical readiness

`validate_model_readiness()` runs the model-agnostic checks that can be honest at
this stage: contract validity, finite dynamics, finite outputs, valid RK4
settings, and one finite integration step.

```python
report = aiogym.validate_model_readiness("cstr")
assert report["passed"]
```

The report explicitly lists physical checks that remain unimplemented. Migrated
models may contribute scenario-specific checks: `quadruple` now supplies its
reference-parameter, nonlinear-equilibrium, volume-balance, and phase checks,
and declares energy balance not applicable to the isothermal hydraulic model.

## Acceptance gate for a future physical model

Before replacing a legacy scenario, its new parameter and task profiles should:

1. replace `legacy-unverified` with a reviewed status;
2. provide traceable primary references and parameter-level source locators;
3. define units, validity ranges, nominal operating points, and solver settings;
4. add scenario-specific conservation and steady-state tests;
5. add at least one transient reference or published control task;
6. pass controller-independent numerical tests before controller retuning begins.

This gate separates model credibility from controller performance and prevents a
better-tuned controller from masking an invalid process model.
