# AIO-Gym core refactor report

Date: 2026-08-06
Branch: `refactor/core-scenario-workflows`
Baseline: `4c70006afb255d37f34d70fdff13d9550ad966c7`
Release target: `0.2.0`

## Outcome

The repository now has one Scenario -> PlantConfig -> Task -> Run path. The
runtime package contains exactly five top-level functional packages and no
parallel v1 runtime:

```text
aiogym/
├── core/
├── scenarios/
│   ├── quadruple/
│   ├── cascade/
│   ├── cascade_recirculating/
│   └── three_tank/
├── controllers/
├── workflows/
└── cli/
```

`quadruple`, `cascade`, and `cascade_recirculating` retain their numerical
models and verified short-horizon behavior. `three_tank` is a formal
six-action, parameterized scenario whose engineering checks, environment,
controller, collection, training, and evaluation use the common runtime.

## Size and structure

| Measurement | Before | After |
|---|---:|---:|
| Python files under `aiogym` | 269 | 50 |
| Python lines under `aiogym` | 57,498 | 8,983 |
| Non-test Python files | 192 | 50 |
| Non-test Python lines | 42,961 | 8,983 |
| Top-level functional packages | 14 | 5 |
| Long-term test Python files | in-package historical suite | 15 files / 1,350 lines |

The 50-file result is below the locked limit of 80. Tests now live under
`tests/core`, `tests/scenarios`, `tests/controllers`, `tests/workflows`, and
`tests/e2e`.

## Dependency graph

The baseline graph had a strongly connected runtime cluster across
`_environment`, `benchmarks`, `controllers`, `datasets`, `evaluation`,
`generation`, and `rl`. For example, benchmark evaluation imported the
environment, controllers, generation, models, rewards, and evaluation, while
RL imported benchmarks, controllers, datasets, evaluation, and generation.

The final inter-package edges are:

```text
scenarios   -> core
controllers -> core, scenarios
workflows   -> core, scenarios, controllers
cli         -> core, scenarios, workflows
```

`core` has no absolute `aiogym.*` imports, and the resulting graph is acyclic.
Optional Torch/SB3 imports occur only inside checkpoint loading or `train()`.

## Final concepts

- `ScenarioPlugin` vertically owns a numerical model factory, PlantConfig
  resolver, Tasks, controller defaults, and an optional study provider.
- `PlantConfig` is the strict, hashable `aiogym.plant.v1` root declaration.
  Scenario-specific fields stay inside `plant`, `operating_point`, and `study`.
- `TaskSpec` owns objective, reward callable, formal metrics, primary metric,
  direction, horizon, control cadence, references, and presets. IDs are
  `<scenario>/<objective>`.
- A Run is one explicit `design`, `collect`, `train`, or `evaluate` workflow.
  Results bind `plant_hash`, `task_hash`, seeds, policy metadata, and artifacts.

Physical state, policy observation, commanded/applied action, scalar reward,
reward terms, constraint costs, and formal evaluation metrics remain separate
channels. `control_dt` is the policy cadence; scenario models retain their
smaller RK4 solver step where declared.

## Migrated, removed, and deferred capabilities

Preserved and migrated:

- parameterized plant validation, study, sweep, strict JSON, reports, and
  no-overwrite behavior;
- stable quadruple, cascade, and cascade-recirculating physical models;
- three-tank six-action model with absent-heater mask and all six design gates;
- environment-bound PID, MPC, hold, random, and SB3 checkpoint Policies;
- episode-oriented Dataset v3 with checksums, atomic writes, resume, and direct
  reader;
- thin SAC/PPO/TD3/DDPG training with save, reload, and evaluation;
- task-owned regulation/economic/safety metrics and multi-seed aggregation;
- five-command CLI: `list`, `design`, `collect`, `train`, `evaluate`.

Removed:

- Track, Anchor, ranking, final-test locks, Claims, Scorecards, Goal and
  RewardSpec registries;
- Case/Distribution/Curriculum and duplicate registries/config systems;
- Dataset v2 runtime, Minari, HPO, ONNX, Oracle, constrained SAC, RLPD/hybrid
  replay, tuning/export stacks, custom SVG reporting, and old artifact readers;
- phase-named protocol tests, profile/config resources, and historical
  readiness reports.

Deferred rather than represented as thin legacy adapters:

- crystallization, CSTR, extraction, fired-heater, and HVAC scenarios. Their
  old horizontal model-only implementations were removed; each must return as
  a complete vertical ScenarioPlugin with a PlantConfig resolver and Task.
- MPC for the six-action three-tank plant remains explicitly unsupported until
  its linearization and default configuration are commissioned.

## API and CLI migration

| Removed 0.1 surface | 0.2 surface |
|---|---|
| `make_env(scenario, case=..., reward_spec=...)` | `make_env("scenario/objective", plant=..., preset=...)` |
| `load_case`, `load_track` | `load_plant`, `list_tasks` |
| `evaluate_controller`, benchmark/final-test | `evaluate(policy, task=..., seeds=...)` |
| Dataset v2 collector/reader | `collect` and `workflows.DatasetReader` |
| RL runner / experiment command | `train` or `train --config FILE` |
| dedicated design engine/wizard | `design new/validate/run/sweep` |

A narrow old `make_env(scenario, case=..., reward_spec=...)` adapter remains for
one release. It emits `DeprecationWarning` and maps only common scenario,
objective, and preset selectors; it does not import or emulate Track/Case
runtime code.

## Dataset v2 to v3

`scripts/migrate_dataset_v2_to_v3.py OLD_DATASET NEW_DATASET` is a standalone,
one-shot converter. It directly verifies and reads v2 manifest/shards, maps
normalized actions into the environment action contract, writes checksummed v3
episodes, and records old Track/distribution/reward metadata only under
`legacy_metadata`. Dataset v3 runtime code imports no v2 package.

## Golden and design comparisons

For fixed seed 1729, fixed actions, and five steps, Phase 0 versus final results
were:

| Scenario | Observation max abs diff | Reward max abs diff |
|---|---:|---:|
| quadruple | 0.0 | 0.0 |
| cascade | 0.0 | 0.0 |
| cascade_recirculating | 0.0 | 0.0 |

No golden file was edited to obtain these results.

The default three-tank design retained all six PASS gates: static engineering,
model readiness, steady-state feasibility, hardware interlocks, dynamic
commissioning, and robustness. Nominal heat-up remains 890 s. Baseline/final
energy was 1.656587583/1.656588045 kWh. The two seed-7 robustness samples remain
PASS with 100% pass rate; heat-up changed from 1056/1354 s to 1060/1339 s and
energy from 2.041076/2.303005 to 2.039672/2.299307 kWh. This bounded change is
from the common Policy path using float32 environment actions instead of the
old design engine's private float64 controller. The acceptance result was not
relaxed. A final-validation bug in which reset discarded robustness
disturbances was caught, fixed with `set_disturbances()`, and regression-tested.

## Controller and RL smoke evidence

Two-step PID and MPC smoke evaluations completed on quadruple, cascade, and
cascade-recirculating with zero constraint violations. Three-tank PID uses the
common matrix PID and its MPC request raises the documented unsupported error.

SAC trained for two steps on each stable scenario, saved a `.zip` checkpoint,
reloaded it, and completed a separate two-step evaluation. Checkpoint SHA-256,
`plant_hash`, and `task_hash` were present for every run. These are pipeline
smokes only, not policy-quality or research-scale performance evidence.

All four supported algorithm classes (SAC, PPO, TD3, DDPG) also have save,
reload, and evaluation coverage in the test suite.

## Verification commands and results

```text
python3 -m ruff check aiogym tests scripts/migrate_dataset_v2_to_v3.py
  -> passed

python3 -m pytest -q
  -> 56 passed in 13.89s

python3 -m build
  -> built aiogym-0.2.0.tar.gz and aiogym-0.2.0-py3-none-any.whl

fresh venv + pip install dist/aiogym-0.2.0-py3-none-any.whl
  -> version 0.2.0; 5 Tasks; PID evaluation 2 steps; design PASS
  -> torch, stable_baselines3, casadi, onnx, and optuna not loaded

train() in the core-only venv
  -> RuntimeError with install `aiogym[rl]` guidance

wheel/sdist inspection
  -> wheel 57 files, sdist 78 files, 0 tests, 0 retired runtime packages,
     packaged three-tank default design present
```

The Phase 0 full historical suite recorded 730 passes plus one restricted-
sandbox multiprocessing permission failure; that single test passed when
rerun with the required semaphore permission. The final 56-test suite is
smaller because deleted protocols no longer have runtime contracts; it covers
the retained core, scenarios, controllers, workflows, golden behavior, and
three full E2E paths.

## Known limits

- Design results are simulation screening, not safety certification or field
  commissioning evidence.
- Built-in equipment values and controller gains remain model assumptions
  unless their ScenarioPlugin provenance says otherwise.
- Three-tank MPC is unsupported.
- The common `make_env` compatibility adapter is temporary and intentionally
  incomplete.
- Formal long-horizon controller performance and research-scale RL training
  were not rerun; short smokes do not establish those claims.
- Five old experimental scenarios were deliberately removed rather than kept
  through hidden legacy dependencies.

## Adding a study-capable scenario

1. Create `aiogym/scenarios/<id>/model.py` and any scenario-local numerical
   helper; implement state/action/output schemas, dynamics, defaults, and
   constraints.
2. Create `plugin.py` with one strict PlantConfig resolver and TaskSpec entries.
3. Implement a small `StudyProvider` beside the model. Return `CheckResult`
   objects for scenario-specific static/steady checks; let `workflows.design`
   own generic sweep, dynamic rollout, robustness, artifacts, and reporting.
4. Put PID/MPC data in `ScenarioPlugin.controller_defaults`; do not add another
   controller registry.
5. Register the plugin in `scenarios/__init__.py`.
6. Add golden model tests, controller tests, Dataset round-trip, Task metric
   separation, SAC smoke, and one E2E path binding the same plant/task hashes.
7. Rebuild and install the wheel to prove all defaults/resources are packaged
   and optional dependencies remain lazy.
