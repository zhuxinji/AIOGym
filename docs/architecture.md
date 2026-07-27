# AIO-Gym architecture

This page describes the current source-code layout and the ownership boundaries
between models, environments, controllers, evaluation, and generated artifacts.
For user-facing vocabulary, read [Concepts](concepts.md). For supported API
entry points, read the [Public API guide](public_api.md).

## Layer overview

```text
scenario + task
      |
      v
ProcessModelContract
      |
      v
AIOGymNativeEnv <--- controller
      |
      v
BenchmarkProtocol + BenchmarkCase
      |
      v
evaluation + rollout
      |
      v
aggregation
      |
      v
artifacts + plots + report + checks
```

The ownership rule is:

- models own physical equations, parameter metadata, and model-specific tasks;
- environments own Gymnasium interaction and runtime state;
- controllers own control algorithms, policy adapters, and controller presets;
- evaluation owns objectives, metrics, benchmark cases, suites, and acceptance;
- the output layer owns artifact persistence, plots, reports, and validation.

## Source tree

```text
AIO-Gym/
|-- aiogym/
|   |-- __init__.py                 top-level public API and Gym registration
|   |-- catalog.py                  scenario/task/suite/controller discovery
|   |-- env.py                      AIOGymNativeEnv orchestration and Gym API
|   |-- env_factory.py              high-level make_env factory
|   |
|   |-- _environment/               internal environment runtime helpers
|   |   |-- config.py               defaults and validation
|   |   |-- disturbances.py         events, disturbances, and plant drift
|   |   |-- observations.py         observations, setpoints, supervisory actions
|   |   `-- transitions.py          reward and termination evaluation
|   |
|   |-- _internal/                  shared internal utilities
|   |   |-- config.py
|   |   |-- identifiers.py
|   |   |-- paths.py
|   |   |-- serialization.py
|   |   `-- validation.py
|   |
|   |-- models/                     physical models and model-owned resources
|   |   |-- core.py                 ProcessModelContract
|   |   |-- backends.py             numeric and CasADi operation adapters
|   |   |-- integration.py          fixed-step RK4 Integrator
|   |   |-- registry.py             model registry and construction
|   |   |-- declarative.py          declarative custom-model support
|   |   |-- validation.py           numerical readiness checks
|   |   |-- metadata.py             structured model metadata validation
|   |   |-- scenarios/              eight built-in process models
|   |   |-- parameters/             parameter metadata JSON files
|   |   |-- tasks/                  task schema, registry, and built-in task JSON
|   |   `-- examples/               custom-model examples
|   |
|   |-- controllers/                controller implementations and construction
|   |   |-- contracts.py            controller protocol and step context
|   |   |-- adapters.py             Python-policy and SB3 adapters
|   |   |-- configs.py              controller configuration loading
|   |   |-- registry.py             controller registry and factories
|   |   |-- pid.py / mpc.py
|   |   |-- oracle.py / onnx.py
|   |   |-- configs/                built-in controller preset JSON files
|   |   `-- tuning/                 tuning utilities
|   |
|   |-- evaluation/                 benchmark semantics and execution
|   |   |-- metric_catalog.py       metric definitions, direction, and schema
|   |   |-- objective_specs.py      objective and reward-mode resolution
|   |   |-- protocols.py            BenchmarkProtocol
|   |   |-- cases.py                EnvironmentSpec and BenchmarkCase
|   |   |-- results.py              schemas, rows, aggregation, and task acceptance
|   |   |-- suite.py                suite loading, expansion, summaries, and paths
|   |   |-- metrics/                tracking/economic/KPI/safety/robustness metrics
|   |   |-- execution/              benchmark, runner, evaluator, rollout, metadata
|   |   |-- artifact/               writers, tables, plots, reports, and checks
|   |   |-- suites/                 built-in suite JSON and shared presets
|   |
|   |-- cli/                        command-line adapters
|   |-- rl/                         shared training config, SB3/RLPD, and transition datasets
|   `-- tests/                      contract, regression, physics, and API tests
|
|-- docs/                           user and contributor documentation
|-- runs/                           default benchmark/training artifacts
|-- README.md
|-- pyproject.toml                  packaging, dependencies, and CLI entry points
`-- MANIFEST.in                     source-distribution files
```

## Public APIs

`aiogym` exposes the supported user API, while `aiogym.controllers` and
`aiogym.models` expose their documented extension contracts. Evaluation
implementation modules are organized by responsibility and are not compatibility
import surfaces. These package facades load implementation groups lazily, so
importing `aiogym` registers Gymnasium IDs without also importing reporting,
plotting, Oracle, ONNX, or reinforcement-learning integrations.

## Configuration and generated data

Runtime package data is stored alongside its owner:

- model parameter metadata: `aiogym/models/parameters/`;
- model task declarations: `aiogym/models/tasks/builtin/`;
- controller presets: `aiogym/controllers/configs/`;
- benchmark suites and shared presets: `aiogym/evaluation/suites/`.

Generated directories are not source code:

- `runs/` contains benchmark and training artifacts;
- `build/` and `dist/` are Python packaging outputs;
- `output/` is reserved for manually exported reports or documents;
- `tmp/` is reserved for disposable intermediate files.

These generated paths are ignored by Git. Deleting `build/`, `dist/`, or `tmp/`
does not affect source code; deleting `runs/` or `output/` may remove results that
cannot be recovered unless they are regenerated.

## Extension points

- Add a process by implementing `ProcessModelContract` and registering it.
- Add a formal experiment under `models/tasks` when it changes model operating
  conditions, setpoints, events, or constraints.
- Add a controller through the controller registry and declare its action mode.
- Add an objective metric under `evaluation/metrics` and register its direction
  and definition in the metric catalog.
- Add reusable benchmark coverage through a suite JSON declaration rather than
  embedding a matrix in CLI code.

See [Physical-model infrastructure](model_infrastructure.md) for model extension
requirements and the [Public API guide](public_api.md) for executable examples.
