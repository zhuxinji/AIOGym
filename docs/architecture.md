# Architecture

AIO-Gym 0.4 has one resolution and execution direction:

```text
ScenarioPlugin -> PlantConfig -> OperatingCondition -> TaskSpec -> ProcessControlEnv
                                                         -> Policy -> Run result
```

- `PlantConfig` contains physical equipment, topology, actuators, parameters,
  safety settings, study metadata, and named conditions.
- `OperatingCondition` contains the initial state, reference, disturbances,
  schedules, `control_dt`, horizon, and observation mode.
- `TaskSpec` contains reward and metric semantics, capabilities, revision, and
  immutable objective coefficients.
- `EnvironmentIdentity` combines `task_hash`, `plant_hash`, `condition_hash`,
  `interface_hash`, and the integrator ID.

All specification and resolved-plant mappings are recursively immutable.
Serialized derivative hashes are checked rather than trusted when a PlantConfig
is loaded.

## Package boundaries

```text
aiogym/
├── core/          contracts, specs, registry, environment, rollout, I/O
├── scenarios/     vertical plugins and physical models
├── controllers/   environment-bound PID, MPC, baseline, and learned policies
├── workflows/     design, Dataset v4, collect, train, evaluate, artifacts
└── cli/           list, design, collect, train, evaluate adapters
```

`core` imports no scenario or optional training dependency. Scenarios own
models and Task definitions. Controllers consume an already-resolved
environment. Workflows own persistent artifacts and share one policy-target
contract resolver.

## Runtime channels

State, observation, action, reward, and formal metrics are separate:

- the state is integrated by the physical model;
- the observation is the policy-facing measurement;
- a policy returns a public action in `env.action_space`;
- a model may expand that public action into private physical slots;
- reward is the Task's per-transition learning signal;
- evaluation accumulates formal episode metrics.

`control_dt` is the policy cadence. Smaller internal integration steps do not
change action cadence or Dataset transition cadence.

## Schedule semantics

At reset, event `schedule[0]` is applied before `observation_0` is returned.
For action `k`, the policy sees schedule state `k`; integration and reward use
that same transition context. The environment then applies `schedule[k+1]`
before returning the next observation. Step info distinguishes:

- `transition_reference` / `transition_disturbance`: context used by the action,
  dynamics, and reward;
- `reference` / `disturbance`: context represented by the returned observation.

## Policy target contracts

Bound PID/MPC policies must exactly match every explicitly requested selector.
Checkpoint Task and interface hashes always match strictly. Plant and condition
transfer are rejected by default and require explicit flags; those flags and
both contracts are written to evaluation results and Dataset manifests.

There is no runtime Track, Anchor, Goal registry, RewardSpec registry, Case
registry, Distribution registry, Claim, Scorecard, final-test lock, or ranking
layer.
