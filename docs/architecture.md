# Architecture

## Runtime flow

```text
load_track() ──> TrackSpec ──> Case sampler ──┐
                                               v
load_case()  ──> CaseSpec ───────────────> AIOGymEnv
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
├── env.py                      AIOGymEnv composition
├── env_factory.py              public make_env
├── _environment/               observations, disturbances, transitions
├── models/
│   ├── core.py                 process-model contract
│   ├── integration.py          numerical integration
│   ├── scenarios/              built-in process models
│   ├── parameter_profiles/     model parameter provenance
│   └── cases/                  Case schema, registry, built-in JSON
├── rewards/
│   ├── specs.py                RewardSpec declarations
│   ├── registry.py             canonical registry
│   ├── canonical.py            reward engine
│   └── terms.py                stage terms and cost channels
├── controllers/                contracts, adapters, PID, MPC, Oracle
├── evaluation/
│   ├── goal_specs.py           GoalSpec
│   ├── metric_catalog.py       Scorecard schema and directions
│   ├── scorecard.py            accumulation and grouping
│   ├── safety_gate.py          ranking eligibility
│   ├── execution/              evaluator and rollout recorder
│   ├── artifact/               current writers, plots, checks, reports
│   └── legacy_artifacts.py     isolated offline historical reader
├── benchmarks/
│   └── tracks/                 Track schema, registry, built-in JSON
├── rl/                         SB3/RLPD training and Track configuration
└── cli/                        discovery, benchmark, training, artifacts
```

## Dependency direction

- Models do not import evaluation or controllers.
- Cases configure a model/environment but do not select Goal or RewardSpec.
- Rewards consume model transition context but do not define benchmark ranking.
- Evaluation consumes environment and controller contracts.
- Tracks compose Case, Goal, RewardSpec, policy, seed, ranking, and safety
  declarations.
- Artifact writers consume current result dictionaries and never normalize old
  schemas.
- The historical reader is leaf-only and is not imported by live execution.

## Extension points

- Add a Scenario through `register_model()` or a declarative model.
- Add a Case under `models/cases/builtin/<scenario>/`.
- Add a RewardSpec through the reward registry when a new learning signal is
  genuinely required.
- Add Scorecard metrics independently of RewardSpec.
- Add an official Track only when policy contract, Case population, seeds,
  ranking, and safety rules are fully declared.
