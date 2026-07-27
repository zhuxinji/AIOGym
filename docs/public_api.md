# Public API guide

This page is the user-facing entry point for selecting a process model, creating
an environment, choosing a reproducible task, and running controller benchmarks.
Import these APIs from the top-level `aiogym` package unless noted otherwise.

For terminology and current built-in coverage, see [Concepts](concepts.md) and
the [Capability matrix](capabilities.md). The [Documentation index](index.md)
links canonical scenario documents, validation records, and internal historical
material.

## Choose the right layer

| Goal | Public entry point | What it selects |
| --- | --- | --- |
| Discover canonical IDs | `list_scenarios()`, `list_tasks()`, `list_suites()`, `list_controllers()` | Current registered and bundled resources |
| Inspect or simulate process equations | `aiogym.make_model(...)` | Scenario/model only |
| Train or interact through Gymnasium | `aiogym.make_env(...)` | Scenario plus optional task and objective |
| Inspect bundled experiments | `aiogym.list_tasks(...)`, `aiogym.load_task_profile(...)` | Task declarations |
| Compare controllers on one experiment | `aiogym.run_benchmark(...)` or `aiogym benchmark suite <scenario> <task>` | One scenario/task/objective protocol |
| Run all scenario tasks or a published matrix | `aiogym benchmark suite <target>` | Multiple resolved benchmark cases |

The concepts are intentionally separate:

- **scenario** selects the process equations, state/action contract, and default
  model parameters;
- **task** selects operating conditions such as initialization, setpoints,
  disturbances, constraints, timing, noise, and optional `model_params`;
- **objective** selects the reward/reporting semantics and primary ranking metric;
- **controller** selects the policy evaluated against that resolved experiment.

## Resource discovery

```python
import aiogym

scenarios = aiogym.list_scenarios()
tasks = aiogym.list_tasks()
cascade_tasks = aiogym.list_tasks("cascade")
suites = aiogym.list_suites()
controllers = aiogym.list_controllers()
```

These functions return sorted tuples of canonical IDs. Scenario and controller
results include runtime custom registrations. Task IDs use the complete
`scenario/name` form; suite IDs are the built-in JSON names without `.json`.

These four consistently named discovery functions are the supported catalog API.

The unified CLI exposes the same catalog and groups the remaining workflows:

```bash
aiogym list scenarios
aiogym list tasks --scenario cascade
aiogym list suites
aiogym list controllers
aiogym benchmark suite --help
aiogym train --help
aiogym artifacts --help
```

The unified `aiogym` executable is the only installed console entry point.

## Models

```python
model = aiogym.make_model(scenario="cascade")
```

`scenario` can be a registered scenario name, a model instance, or a zero-argument
model factory. A model is independent of task and objective selection.

```python
import aiogym

model = aiogym.make_model("cascade-recirculating")
equilibrium = model.nominal_steady_state()
```

Use `aiogym.list_scenarios()` to inspect registered scenario names. Custom
models can be managed with `register_model()` and `unregister_model()`.

## Environments

```python
env = aiogym.make_env(
    scenario="cascade",
    objective=None,
    seed=None,
    config=None,
    *,
    protocol=None,
    **overrides,
)
```

For reproducible work, select a named task. If `objective` is omitted, the task's
`default_objective` is used.

```python
import aiogym

env = aiogym.make_env(
    "cascade-recirculating",
    task="temperature-step",
    seed=7,
)
obs, info = env.reset(seed=7)
```

A no-task `make_env()` call preserves the direct interactive environment defaults.
It does not acquire a benchmark task identity or task acceptance criteria.

### Automatic events

`auto_events=False` does **not** turn the physical process into a static model. The
model still advances through `model.dynamics(...)` on every integration step.
The environment's `auto_events` option only controls generic automatically generated
within-episode events. A named task can set it to `false` and still apply its own
deterministic setpoint and disturbance schedule.

`auto_events` is the only accepted field; removed names are rejected.

Use `model_params={...}` when the numerical plant parameters should differ from
the scenario defaults. Initial state, setpoints, disturbance schedules, and
`control_dt` change the experiment without changing the underlying equations.
Named-task setpoints can be replaced directly:

```python
env = aiogym.make_env(
    "quadruple",
    task="minimum-phase",
    episode_steps=360,
    initial_setpoint=[12.2629675195507, 12.783158403008972],
    setpoint_schedule=[
        {"at_step": 0, "values": [14.2629675195507, 12.783158403008972]}
    ],
)
```

Set `randomize_setpoints=True` to create one reproducible target per episode:

```python
env = aiogym.make_env(
    "quadruple",
    task="nonminimum-phase",
    randomize_setpoints=True,
)
obs, info = env.reset(seed=9000)
```

If the task has a setpoint schedule, its initial equilibrium is preserved and
the scheduled values are sampled. Otherwise the initial target is sampled.
Sampling stays within the model's SP bounds; models with an analytic
steady-state feasibility check reject unreachable candidates.

## Tasks

```python
tasks = aiogym.list_tasks("cascade")
task = aiogym.load_task_profile(
    "cascade/temperature-step"
)
```

The short form is accepted when the scenario is supplied by the caller:

```python
env = aiogym.make_env(
    "cascade",
    task="temperature-step",
)
```

Task files live under `aiogym/models/tasks/builtin/<scenario>/`. They are
validated, versioned experiment declarations bound to model scenarios rather
than executable model implementations.

## Objective resolution

One objective is resolved before an environment or benchmark case is created:

```text
explicit API/CLI objective
-> case objective
-> suite objective
-> task.default_objective
```

If all four are absent, benchmark protocol construction raises an error.

Resolved benchmark metadata records `objective_source` as `explicit`,
`case-config`, `suite-config`, or `task-default`.

Changing objective does not change task conditions. `robustness` and `safety`
use the environment's KPI reward implementation but retain distinct benchmark
metrics and result identities.

## One-protocol benchmarks

```python
payload = aiogym.run_benchmark(config, *, objective=None)
```

`config` can be a mapping or a JSON path. One resolved
scenario/task/objective protocol is evaluated with every listed controller and
seed.

The canonical top-level fields are `scenario`, `task`, `objective`,
`controllers`, `controller_configs`, `seeds`, `environment`, `output_dir`,
`overwrite`, and `strict`. Environment settings may also be written directly
at the top level. Removed aliases are rejected instead of being interpreted
silently.

```python
import aiogym

payload = aiogym.run_benchmark({
    "scenario": "cascade-recirculating",
    "task": "temperature-step",
    "controllers": ["pid", "mpc"],
    "seeds": [9000, 9001, 9002],
    "output_dir": "runs/recirculating-temperature-step",
})
```

The equivalent command-line workflow is:

```bash
aiogym benchmark suite cascade-recirculating temperature-step \
  --controllers pid,mpc \
  --episodes 3
```

Use `aiogym benchmark suite <scenario> <task> --help` for named-task runtime
overrides. `--setpoint-step STEP:VALUE1,VALUE2` is repeatable; providing it
replaces the task's default schedule.

For a multi-seed generalization benchmark:

```bash
aiogym benchmark suite quadruple nonminimum-phase \
  --controllers pid,mpc,oracle \
  --episodes 10 \
  --randomize-setpoints
```

Seeds `9000` through `9009` generate ten target schedules at reset time. The
same seed reproduces the same schedule, and the realized schedules are stored
with the evaluation and rollout metadata. Without `--randomize-setpoints`,
multiple episodes retain the task's deterministic references.

The command writes the same standard artifact directory as `run_benchmark()`,
including `benchmark.json`, structured config, metadata,
summary, results, figures, and `report.md`. Single-task CLI and API benchmarks
default to `runs/<scenario>/<task>` relative to the current working directory.
Suite benchmarks default to `runs/<suite>_suite`. Runs replace the standard
managed artifacts in their base directory by default; unrelated user files are
preserved. Use CLI `--no-overwrite` or API `"overwrite": false` to create a
timestamped sibling instead. Single-task tracking benchmarks automatically save
one rollout per controller and generate state/setpoint/control figures. Set
`AIOGYM_RUNS_DIR` to change the shared default root. Explicit API and CLI base
paths such as `output_dir` and `--artifact-dir` take precedence and obey the
same overwrite setting.

Automation should consume the standard artifact directory so it can use
`aiogym artifacts check` and `aiogym artifacts report` directly.

## Benchmark suites

A suite is a JSON source declaration that expands to one or more canonical
benchmark cases. The loader currently accepts two source shapes: a Cartesian
matrix and an explicit `cases` list. Both shapes resolve to the same
`BenchmarkCase` contract before execution.

The same command also accepts a scenario target. With no task it runs every
registered task for that scenario; with a task positional argument it runs only
that task:

```bash
aiogym benchmark suite cascade
aiogym benchmark suite cascade temperature-step
```

Scenarios without registered tasks run one default tracking case; use
`--objective` to select another supported objective.

```bash
aiogym benchmark suite cascade-control \
  --episodes 3
```

The suite runner supports per-case tasks, objectives, controller profiles, and
environment settings. It writes a standard artifact directory and keeps rankings
separate for every `(scenario, task, objective)` identity. Use
`aiogym benchmark suite --help` to list built-in suites and filtering options.

### Matrix source schema

When `cases` is absent, the top-level `scenarios`, `objectives`, and
`controllers` fields are Cartesian axes:

```json
{
  "name": "tracking-demo",
  "description": "Two-scenario controller comparison.",
  "scenarios": ["cstr", "hvac"],
  "objectives": ["tracking"],
  "controllers": ["pid", "mpc"],
  "action_mode": "actuator",
  "episode_steps": 80,
  "control_dt": 0.5,
  "environment": {
    "auto_events": false,
    "randomize": false
  }
}
```

This declaration expands to `2 scenarios × 1 objective × 2 controllers = 4`
resolved cases. `scenarios` may also be one built-in group string such as
`ALL_SCENARIOS`. Scenario groups are resolved from the current model registry,
so runtime registrations are included. A top-level `task` applies to every
expanded combination and must therefore be valid for every selected scenario.

A suite that tracks the model-owned task registry can use a task selector
instead of repeating task names:

```json
{
  "scenarios": ["cascade"],
  "controllers": ["pid", "mpc"],
  "tasks": {
    "exclude": ["continuous-benchmark"]
  }
}
```

Use `"tasks": "ALL_TASKS"` to include every registered task for the selected
scenarios. Task selectors and explicit `cases` are mutually exclusive.

### Explicit-cases source schema

When a non-empty `cases` list is present, the runner expands only those
declarations. The top-level axes become defaults; they do not produce an
additional matrix:

```json
{
  "name": "quadruple-demo",
  "controllers": ["pid", "mpc"],
  "action_mode": "actuator",
  "episode_steps": 600,
  "cases": [
    {
      "scenario": "quadruple",
      "task": "minimum-phase",
      "objective": "tracking"
    },
    {
      "scenario": "quadruple",
      "task": "disturbance-rejection",
      "objective": "robustness",
      "controllers": ["pid", "mpc", "oracle"],
      "controller_configs": {
        "oracle": {"profile": "quadruple-disturbance-rejection"}
      }
    }
  ]
}
```

The first declaration inherits two controllers and expands to two resolved
cases. The second overrides the controller list and expands to three. The suite
therefore contains five resolved cases, not a top-level matrix plus five cases.

`cases` must be a non-empty list, and every item must be a JSON object. Inside a
case, singular fields select one value while plural fields create a local axis:

| Concept | Suite or matrix field | Explicit-case fields |
| --- | --- | --- |
| Scenario | `scenarios` | `scenario` or `scenarios` |
| Objective | `objectives` | `objective` or `objectives` |
| Controller | `controllers` | `controller` or `controllers` |
| Task | `task` | `task` |
| Action mode | `action_mode` | `action_mode` |
| Environment | direct environment keys or `environment` | direct environment keys or `environment`, including `initial_setpoint` and `setpoint_schedule` |
| Controller tuning | `controller_configs` | `controller_configs` |
| Timing | `episode_steps`, `control_dt` | `episode_steps`, `control_dt` |

The source schema accepts singular and plural case fields for case expansion.
After expansion, each canonical case always has one `scenario`, one task
identity, one `objective`, one `controller`, one environment specification, and
one seed list.

### Source reuse

Suite source files can remove repeated declarations with three validated
reference fields:

```json
{
  "extends": "economic-actuator",
  "description": "Economic benchmark for supervisory policies.",
  "controllers": ["sb3"],
  "action_mode": "setpoint"
}
```

- `extends` accepts a suite ID, a relative JSON path, or an ordered list of
  either. Parent mappings are merged first and the child wins. Nested mappings
  merge recursively; lists and scalar values are replaced.
- `preset` accepts a reusable preset ID or ordered list of IDs. It is valid on
  the suite, an explicit case, or its `environment` mapping.
- `case_ref` is valid on an explicit case and imports one reusable canonical
  case declaration. Fields beside `case_ref` override the imported declaration.

Bundled preset and case IDs are declared under
`aiogym/evaluation/suites/presets/`. Missing IDs, duplicate IDs, invalid types,
and cycles fail during loading. References are fully expanded before suite
validation and execution, so `suite_source_config` artifacts remain
self-contained and contain no `extends`, `preset`, or `case_ref` fields.

### Inheritance and overrides

For fields that can appear at more than one level, resolution follows these
rules:

- objective: CLI `--objectives` override, then case `objective`/`objectives`,
  then suite `objectives`, then `task.default_objective`;
- task: case `task`, then suite `task`;
- action mode: case `action_mode`, then suite `action_mode`, then `actuator`;
- environment keys: case direct field, case `environment`, suite direct field,
  suite `environment`, then protocol/environment default;
- timing: CLI `--episode-steps` or `--control-dt`, then case, then suite, then
  task/environment default;
- controller configuration: runner/controller base, then task `controllers`
  defaults, suite `controller_configs`, and case `controller_configs`, with
  later nested `parameters` values taking precedence. Single-task and suite
  runs therefore use the same task-owned profile unless explicitly overridden.

Objective resolution records its winning source as `explicit`, `case-config`,
`suite-config`, or `task-default` in the resolved protocol and artifacts.

### CLI selection behavior

The same CLI flags intentionally operate differently on the two source shapes:

- matrix suite: `--scenarios`, `--objectives`, and `--controllers` replace the
  corresponding top-level axis;
- explicit-cases suite: `--scenarios` and `--controllers` filter declared
  expansions; they do not create undeclared cases;
- explicit-cases suite: `--objectives` overrides the objective candidate for
  each selected declaration rather than acting as a filter;
- both shapes: `--episode-steps` and `--control-dt` are global overrides.

An objective override that is not allowed by a selected task fails during
protocol resolution; it is not silently replaced. If filters select no cases,
the runner raises `suite filters selected no benchmark cases`.

See `aiogym/evaluation/suites/robustness-actuator.json` for matrix form and
`aiogym/evaluation/suites/quadruple.json` for explicit-cases form. Hybrid files
such as `standard-baselines.json` use top-level defaults plus explicit cases;
the presence of `cases` still selects explicit-cases mode.

## Training objectives

Training commands use the same five public objective names as benchmarks:

```bash
aiogym train sb3 --scenario cstr --objective tracking
aiogym train rlpd --scenario quadruple --task minimum-phase
```

`economic` and `tracking` select their matching environment rewards. `kpi`,
`robustness`, and `safety` use the KPI environment reward; artifacts retain the
selected objective and record the internal choice as `resolved_reward_mode`.
SB3 evaluation defaults to the training objective unless `--eval-objective` is
provided explicitly.

Both training commands accept named task profiles. When a task is selected,
both resolve its default objective, control interval, episode length, tracking
weights, and observation contract identically; explicit command-line overrides
still take precedence. The optimization parameters remain specific to the
selected algorithm.

Training commands accept `--objective`. The low-level
`AIOGymNativeEnv(reward_mode=...)` constructor uses `reward_mode` to select its
internal reward computation for framework integrations.

Oracle controller configuration likewise uses `objective`. The former
Oracle-only `mode` alias and duplicate `mode` metadata field have been removed;
use `objective` in controller configuration and artifact readers.

## Advanced resolved specifications

Most users should prefer `make_env()` and `run_benchmark()`. Framework and
integration code can construct:

- `ObjectiveSpec`: resolved scoring semantics and provenance;
- `EnvironmentSpec`: environment construction independent of ranking;
- `BenchmarkProtocol`: a convenient declaration that resolves both;
- `BenchmarkCase`: the canonical resolved environment/objective/controller/seed evaluation unit.

Benchmark result rows distinguish `execution_status` (whether the run completed)
from `objective_status` (whether task acceptance criteria were met).
