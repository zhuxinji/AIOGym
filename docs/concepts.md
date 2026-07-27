# AIO-Gym concepts

This page defines the user-facing concepts that connect a process model to a
reproducible benchmark result. It describes the repository state at Git revision
`7bf0b08`; use the executable registries and task files as the source of truth
after later changes.

## From a scenario to an artifact

```text
scenario ──> model equations and state/action contract
    │
    ├── task ──> operating point, timing, events, constraints, acceptance
    │
    └── objective ──> reward, metrics, primary metric, ranking direction
                         │
controller preset ──────┤
                         v
                  resolved benchmark case
                         │
                         v
              rollout rows and artifact directory
```

A benchmark identity is the resolved `(scenario, task, objective)` tuple. A
controller is evaluated against that identity; changing the controller does not
create a different experiment. Results from different identities should not be
placed in the same ranking.

## Scenario and model

A **scenario** is the stable external identifier for a process, such as
`quadruple` or `cascade-recirculating`. It selects the executable **model**:
equations, state vector, actuator vector, controlled outputs, disturbances,
default parameters, bounds, and integration metadata.

Use `aiogym.make_model(...)` to inspect equations and metadata without choosing
an experiment. Use `aiogym.make_env(...)` when Gymnasium interaction, task
conditions, or objective semantics are needed.

Model numeric backends and RK4 integration live in focused modules under
`aiogym.models`; `models.core` owns the process-model contract. Environment
configuration, disturbance scheduling, observation
construction, and transition evaluation are composed behind
`aiogym.env.AIOGymNativeEnv`, so the public environment API remains stable.

Public resource identifiers use one consistent style:

- scenario IDs use kebab case, for example `cascade-recirculating`;
- task IDs use `scenario/name`, for example
  `quadruple/minimum-phase`;
- suite IDs use kebab case, for example `cascade-recirculating`;
- Gymnasium IDs use names such as `AIOGym/CascadeRecirculating-v0`.

Public Python calls and configuration files must use canonical IDs. Internal
Python module names and Gymnasium IDs are independent implementation details.

## Parameter metadata and runtime parameters

Files under `aiogym/models/parameters/` are **parameter metadata** (called
parameter profiles by the current API). They record provenance, units,
validity ranges, uncertainty, solver settings, and fidelity status. Loading one
does not mutate a model or change `model.p`.

Use `model_params={...}` in an environment or task when numerical model
parameters must change at runtime. This distinction prevents descriptive
provenance from silently changing an experiment.

## Task

A **task** is a versioned experiment declaration. It may select:

- initial state and setpoint;
- deterministic setpoint or disturbance events;
- control interval and episode length;
- action mode, noise, randomization, and termination settings;
- runtime `model_params` and optional production-throughput targets;
- supported and default objectives;
- constraints and acceptance thresholds.

Task IDs are optional for interactive environments but recommended for
reproducible evaluation. A no-task environment has direct environment defaults;
it does not gain a formal task identity or task acceptance criteria.

Task schema, discovery, and bundled resources live under `aiogym.models.tasks`
because each task defines an experiment for one model scenario. Evaluation
consumes the resolved task and owns acceptance calculation; it does not own the
task's runtime environment definition.

The current code calls these declarations task profiles. In user-facing prose,
**task spec** is the less ambiguous term: unlike parameter metadata, a task can
change executable conditions.

## Objective and reward mode

An **objective** selects reporting and ranking semantics. A task may limit which
objectives are valid. Objective resolution follows this precedence:

```text
explicit API or CLI value
-> benchmark case
-> benchmark suite
-> task default
```

The five public objectives map to three internal environment reward modes:

| Objective | Internal reward mode | Primary metric | Direction |
| --- | --- | --- | --- |
| `tracking` | `tracking` | `tracking_cost` | minimize |
| `economic` | `economic` | `profit` | maximize |
| `kpi` | `kpi` | `normalized_score` | maximize |
| `robustness` | `kpi` | `normalized_score` | maximize |
| `safety` | `kpi` | `constraint_violation_count` | minimize |

Users should select an objective. `reward_mode` is the internal training signal
chosen by that objective; it is not a substitute identity for robustness or
safety benchmarks. High-level APIs and training CLIs accept only `objective`.
Resolved metadata records
`resolved_reward_mode` for diagnostics without replacing the objective identity.

Unless a task or runtime override declares different weights, tracking uses
`Q=0.7` for normalized squared output error and `R=0.3` for normalized squared
input movement:

```text
tracking_cost = 0.7 * sum(normalized_error²)
              + 0.3 * sum(normalized_input_move²)
```

The resolved Q/R values are also forwarded to MPC and Oracle so their internal
tracking objective matches the reported benchmark objective.

Evaluation internals separate metric catalog, objective resolution, benchmark
case models, execution, rollout recording, aggregation, and metadata. Public
workflows should use the top-level `aiogym` API.

## Controller and controller preset

A **controller** is an implementation registered under an ID such as `pid`,
`mpc`, `oracle`, `policy`, `sb3`, or `onnx`.

A **controller preset** (called a controller profile in the current API) is a
named set of tuning parameters for a scenario or operating point. It changes
controller behavior but must not redefine the task. PID, MPC, and Oracle are
built-in algorithmic controllers; policy, SB3, and ONNX adapters require a
policy object or model artifact supplied by the caller.

Named tasks may bind default profiles through their `controllers` mapping.
Single-task, suite, and Python API benchmarks all resolve that mapping when
constructing a `BenchmarkCase`; selecting a suite does not select a different
controller tuning. Explicit controller configuration remains a higher-priority
override.

Controller contracts, policy adapters, configuration loading, and registry
factories live in focused modules under `aiogym.controllers`. The package-level
`aiogym.controllers` API remains the stable import facade.

Controller construction proves interface compatibility, not control quality.
Performance claims require a named task, objective, seeds, and recorded
benchmark artifacts.

## Action mode

`action_mode="actuator"` exposes the model's normalized actuator vector
directly. It is available for all built-in scenarios.

`action_mode="setpoint"` exposes a smaller supervisory action defined by the
model's `supervisory_layout`. An inner PID converts those supervisory choices to
actuator commands. It is available only when the model declares such a layout;
the [capability matrix](capabilities.md) lists the current scenarios.

Action mode belongs to the environment/task contract. A controller or learned
policy must declare the same mode and output the corresponding dimension.

## Automatic and deterministic events

The `auto_events` option does not enable or disable the differential equations. The
physical model is integrated on every step.

`auto_events=True` enables generic automatically generated within-episode events.
A reproducible task can set `auto_events: false` and still declare deterministic
events under `setpoints.schedule` or `disturbances`. `auto_events` is the only
accepted configuration field.

### Seeded setpoint randomization

`randomize_setpoints=True` samples targets during `env.reset(seed=...)`; it is
independent of `auto_events`. The same seed reproduces the same target, while a
multi-seed evaluation receives one target schedule per episode.

For a task with declared setpoint events, the initial reference equilibrium is
preserved and the event values are sampled. For a task without setpoint events,
the initial reference is sampled. Sampling honors each model's setpoint bounds.
Models with analytic steady-state feasibility, including `quadruple` and
`cascade`, reject unreachable targets. At the quadruple zero boundary, the
sampler instead samples a safe steady actuator pair and maps its equilibrium
forward onto the reachable output manifold.

Reference-derived tasks keep this option disabled by default. Increasing
`episodes` never enables randomization implicitly; callers must request it so
changing the number of episodes does not silently change task semantics.

## Suite and resolved case

A **suite** is a versioned source declaration for benchmark cases. A file with
no `cases` field is a Cartesian matrix. A file with a non-empty `cases` list is
an explicit-cases suite: top-level fields supply defaults, but do not generate
an additional matrix. In both forms, the runner expands source declarations to
a canonical resolved case containing:

- one environment specification;
- one objective specification;
- controller configurations;
- seeds and artifact settings.

The resolved case always uses singular identities even when its source used
plural axes. This separates convenient authoring syntax from the stable
execution and artifact contract.

Suite authors can use `extends` for suite inheritance, `preset` for shared
configuration fragments, and `case_ref` for shared explicit cases. The loader
validates and fully expands those references before producing resolved cases or
artifact configuration, so reuse syntax does not become a runtime dependency.

Suite declaration loading, resolved-case expansion, and summary preparation
live in evaluation modules rather than the CLI entrypoint.

The [suite source schemas](public_api.md#matrix-source-schema) document accepted
fields, inheritance, and CLI selection behavior. The executable loader and JSON
files under `aiogym/evaluation/suites/` remain the final source of truth.

## Artifact

An **artifact** is the durable output of a benchmark or training run. A standard
benchmark artifact directory records configuration, metadata, summary rows,
rollouts, figures, and optional training outputs. It should contain enough
resolved identity and provenance to reproduce or audit a ranking.

`execution_status` answers whether a run completed. `objective_status` answers
whether declared task acceptance thresholds were met. A completed run can
therefore fail its objective without being an execution failure.

Artifact row preparation, file writing, plot orchestration, Markdown rendering,
and structural checks are separate output-layer responsibilities. The top-level
API and unified CLI expose supported artifact operations.

## Which page to use next

- Use the [capability matrix](capabilities.md) to choose a supported combination.
- Use the [public API guide](public_api.md) to construct or run it.
- Use the [physical-model infrastructure guide](model_infrastructure.md) for
  model provenance and extension contracts.
- Return to the [documentation index](index.md) for model documentation,
  validation snapshots, and internal records.
