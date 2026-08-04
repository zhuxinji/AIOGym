# Architecture

## Runtime flow

```text
load_track() ──> TrackSpec ──> Case sampler ──┐
                                               v
load_case()  ──> CaseSpec ───────────────> make_env()
                                               |
DistributionSpec ──> EpisodeSpec ──reset()────┤
                                               |
RewardSpec ──> canonical reward engine <──────┤
                                               v
controller ──> evaluate_controller() ──> Scorecard + safety gate
                                               |
                                               v
                                  result / artifact / report
```

An explicit specialist run starts at `load_case()`. An official run starts at
`load_track()` and evaluates a single policy contract across its Case
population.

## Package boundaries

```text
aiogym/
├── _environment/               private environment implementation
│   ├── factory.py              implementation behind public make_env()
│   ├── spec.py                 immutable ResolvedEnvSpec + resolver
│   ├── env.py                  sole concrete Gymnasium environment
│   ├── builder.py              Track-specific resolved-spec builders
│   └── realism.py              EpisodeSpec sensor/actuator execution
├── models/
│   ├── core.py                 process-model contract
│   ├── integration.py          numerical integration
│   ├── scenarios/              built-in process models
│   ├── parameter_profiles.py   parameter-profile loading and validation
│   ├── parameters/             built-in parameter provenance JSON
│   └── cases/                  Case schema, registry, built-in JSON
├── rewards/
│   ├── specs.py                RewardSpec declarations
│   ├── registry.py             canonical registry
│   ├── canonical.py            reward engine
│   └── terms.py                stage terms and cost channels
├── _internal/
│   ├── control_math.py         shared action/tracking normalization
│   └── serialization.py        canonical JSON, hashes, atomic JSON
├── resources/profiles/         installed guided profile declarations
├── controllers/                contracts, adapters, PID, MPC, Oracle
├── evaluation/
│   ├── goal_specs.py           GoalSpec
│   ├── metric_catalog.py       Scorecard schema and directions
│   ├── scorecard.py            accumulation and grouping
│   ├── safety_gate.py          ranking eligibility
│   ├── statistics.py           IQM, bootstrap, final score matrices
│   ├── execution/              evaluator and rollout recorder
│   ├── artifact/               current writers, plots, checks, reports
├── benchmarks/
│   └── tracks/                 Track schema, registry, built-in JSON
├── generation/                 distributions, episodes, seeds, resolvers
│   ├── curriculum.py           fixed L0-L4 difficulty declarations
│   ├── quadruple.py            physical quadruple training sampler
│   └── registry.py             versioned distribution IDs
├── datasets/                   episode schema, atomic shards, readers
│   ├── collector.py            collector provenance and v2 collection
│   ├── quality.py              bounded-memory coverage reports
│   └── minari_adapter.py       optional lossless sidecar conversion
├── rl/                         unified config, preprocessing, replay, trainers
│   ├── backends/               direct BC, SB3, and RLPD training kernels
│   ├── coordinator.py          global episode assignment
│   ├── checkpoints.py          atomic restart-episode resume contract
│   ├── dataset_replay.py       immutable Dataset v2 prior replay
│   ├── hybrid_replay.py        canonical RLPD source mixing
│   ├── behavior_cloning.py     offline data-alignment sanity check
│   ├── online_collection.py    batched normalized-action collection
│   ├── validation.py           fixed plans and eligible checkpoint selection
│   ├── lifecycle.py            shared validation/artifact finalization
│   ├── hpo.py                  benchmark-immutable Optuna studies
│   ├── final_test.py           one-shot locked test evaluation
│   ├── observations.py         history and recurrent policy contracts
│   ├── safety.py               auditable action projection shields
│   ├── constrained.py          reward/cost-separated Lagrangian SAC
│   ├── statistics.py           shared action/observation preprocessing
│   └── vector.py               autoreset-safe transition extraction
└── cli/                        discovery, benchmark, training, artifacts
```

## Dependency direction

- Models do not import evaluation or controllers.
- Models and rewards share neutral control math without importing evaluation:

  ```text
  models ───────────────┐
                        v
  _internal.control_math
         │              │
         v              v
      rewards       evaluation
  ```
- Cases configure a model/environment but do not select Goal or RewardSpec.
- Rewards consume model transition context but do not define benchmark ranking.
- Evaluation consumes environment and controller contracts.
- Tracks compose Case, Goal, RewardSpec, policy, seed, ranking, and safety
  declarations.
- Generation resolves all stochastic episode conditions before environment
  execution and records stable provenance hashes.
- A Track may name a programmatic training distribution. Fixed Track cases
  remain the validation/test population rather than the training generator.
- Dataset persistence accepts only complete validated episode objects and
  keeps train/validation/test splits atomic at episode boundaries.
- RL workers consume coordinator assignments; worker placement is not part of
  an episode identity. UTD is counted per environment transition.
- Training observation statistics are shared read-only with validation/test
  environments, and policy actions use one `[-1, 1]` contract.
- Hybrid trainers keep offline and online replay separate. Offline sampling is
  source-stratified, timeouts preserve bootstrap semantics, and every artifact
  binds the prior dataset ID and manifest hash.
- Validation checkpoint selection cannot accept test results. Algorithms share
  one resolved EpisodeSpec plan; only eligible checkpoints can become best.
- The runner's resolved config is the unique training identity consumed by
  private backend kernels. A backend returns only a serializable selected
  checkpoint result; it does not write final benchmark artifacts or return a
  live controller. The shared lifecycle verifies the checkpoint digest, loads
  it through the canonical policy loader, runs fixed validation, and is the
  sole writer of standard artifacts.
- Final test evaluation is one-shot and emits robust per-seed/per-case
  statistics through `evaluation.statistics`.
- Sensor bias, noise, drift, delay, dropout, and quantization and actuator
  efficiency, delay, deadband, lag, and slew limits are resolved by
  EpisodeSpec. Repeated observation reads within one control step are cached,
  so instrumentation cannot change the sensor realization.
- Tracks distinguish full-state and measured-output sensing from single-step,
  fixed-history, and recurrent policy memory. Learned-policy contexts never
  contain the environment object or diagnostic `info`.
- Safety shields preserve policy proposals, shielded commands, and final
  actuator actions as separate audit channels. Constrained RL reads explicit
  cost channels; it never rewrites the scalar benchmark reward.
- Artifact writers consume current result dictionaries and never normalize old
  schemas.
- Live execution accepts current schemas only; legacy compatibility readers are
  not shipped.

## Identity and release boundaries

- `ResolvedEnvSpec.runtime_hash` includes instrumentation such as `info_level`
  and profiling; the v1-compatible `spec_hash` is an alias for that digest.
  `mdp_hash` excludes instrumentation while retaining transition,
  observation, action, reward, event, and model semantics. Existing Dataset
  and policy schemas still bind `env_spec_hash`; migrating compatibility checks
  to `mdp_hash` requires a future schema version.
- Guided training and collection profiles have one source of truth under
  installed `aiogym/resources/profiles/`; `quick` remains tutorial-scale.
- `LagrangianSAC` is an experimental direct-call research component available
  from `aiogym.experimental.rl`. It is not in stable training config, CLI, or
  SB3 adapter discovery.
- Release wheels and sdists are built from clean source and checked for required
  runtime modules, installed profile resources, stale `runtime.py`, tests, and
  generated artifacts before wheel-only smoke tests run outside the source tree.

## Extension points

- Add a Scenario through `register_model()` or a declarative model.
- Add a Case under `models/cases/builtin/<scenario>/`.
- Add a RewardSpec through the reward registry when a new learning signal is
  genuinely required.
- Add Scorecard metrics independently of RewardSpec.
- Add an official Track only when policy contract, Case population, seeds,
  ranking, and safety rules are fully declared.
