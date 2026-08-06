# Architecture

AIO-Gym 0.2 has four domain concepts and one execution direction:

```text
ScenarioPlugin -> PlantConfig -> ResolvedPlant -> TaskSpec -> ProcessControlEnv
                                                         -> Policy -> Run result
```

- `ScenarioPlugin` is the vertical owner of a model factory, plant resolver,
  supported tasks, controller defaults, and optional design-study provider.
- `PlantConfig` describes equipment independently of an operating objective.
- `TaskSpec` owns objective, reward, metrics, horizon, control cadence, and
  presets. It replaces benchmark protocol composition.
- A run is an explicit design, collection, training, or evaluation workflow.

## Package boundaries

```text
aiogym/
├── core/          contracts, immutable specs, registry, environment, rollout, I/O
├── scenarios/     vertical scenario plugins and physical models
├── controllers/   Policy implementations and environment-bound construction
├── workflows/     design, Dataset v3, collection, training, evaluation, artifacts
└── cli/           list, design, collect, train, evaluate adapters
```

Dependency direction is one-way. `core` imports no scenario, controller,
workflow, or optional training dependency. Scenario plugins depend on `core`.
Controllers depend on an already-created environment. Workflows compose these
parts and own persistent artifacts. CLI modules only translate user input into
workflow calls.

## State, observation, action, reward, and metric

These channels remain separate:

- model state is the physical integration state;
- observation is the policy-facing measurement;
- the policy returns a commanded action in `env.action_space`;
- the environment records the applied action and constraint channels;
- reward is the Task's per-transition learning signal;
- formal metrics are accumulated by `workflows.evaluate`, independently of
  reward shaping.

`control_dt` is the policy cadence. A model may perform smaller internal solver
steps without changing the Task horizon or Dataset transition cadence.

## Identity and artifacts

`PlantConfig.plant_hash` binds equipment declarations. `TaskSpec.task_hash`
binds objective semantics. Dataset v3 manifests and run artifacts record both,
plus the preset, policy metadata, seeds, and file checksums. Writers refuse to
replace an existing artifact unless the caller explicitly enables overwrite or
resume behavior.

No Track, Anchor, Goal, RewardSpec, Case, Distribution, Curriculum, Claim,
artifact-compatibility, or ranking layer participates in the 0.2 runtime.
