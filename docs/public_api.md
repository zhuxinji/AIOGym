# Public API

AIO-Gym v1 intentionally exposes a small product surface. Imports from modules
not listed here are implementation details and may change without a
compatibility shim.

## Top-level package

```python
import aiogym

aiogym.__version__
aiogym.make_env(...)
aiogym.make_controller(...)
aiogym.list_scenarios()
aiogym.list_cases()
aiogym.list_tracks()
aiogym.list_controllers()
aiogym.load_case(...)
aiogym.load_track(...)
aiogym.evaluate_controller(...)
```

Create environments only with `make_env` and seed them only with
`env.reset(seed=...)`. The concrete environment class and resolved environment
spec are private.

## Stable package facades

| Facade | Stable names |
|---|---|
| `aiogym.benchmarks` | `TrackSpec`, `list_tracks`, `load_track`, `evaluate_policy_on_track` |
| `aiogym.controllers` | `Controller`, `make_controller`, `register_controller`, `unregister_controller` |
| `aiogym.datasets` | `DatasetCollectionConfig`, `DatasetReader`, `validate_dataset`, `list_collectors` |
| `aiogym.design` | `prompt_design_spec`, `load_design_spec`, `validate_design_spec`, `compile_design_model`, `run_design_study`, `run_design_sweep`, `write_design_report_bundle` |
| `aiogym.evaluation` | `evaluate_controller`, `build_evaluation_report`, `build_final_statistical_report`, `render_benchmark_report` |
| `aiogym.generation` | `DistributionSpec`, `EpisodeSpec`, `list_distributions`, `load_distribution`, `make_episode_sampler` |
| `aiogym.models` | `ProcessModelContract`, `define_model`, `make_model`, `register_model`, `unregister_model`, `CaseSpec`, `list_cases`, `load_case` |
| `aiogym.rewards` | `RewardSpec`, `get_reward_spec`, `list_reward_specs`, `stage_reward` |
| `aiogym.rl` | `RLTrainingConfig`, `RunResult`, `run_experiment`, `run_seed_sweep`, `list_algorithms` |

Advanced helpers are imported from their responsibility-specific modules. For
example, checkpoint verification and loading live in
`aiogym.controllers.checkpoints`; Dataset v2 writing lives in
`aiogym.datasets.writer`. Those explicit modules are not facade promises.
`aiogym.benchmarks.evaluate_policy_on_track` always evaluates the official
validation split. Locked test evaluation is available only through
`aiogym final-test`.

## Environment contract

```python
env = aiogym.make_env(
    "quadruple",
    case="minimum-phase",
    reward_spec="regulation",
)
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(action)
```

Direct arguments and `config=` are mutually exclusive and resolve into the
same internal specification. No Gym registration side effect or alternate
constructor is provided.

## Controller and checkpoint contract

The default controller catalog is `hold`, `mpc`, and `pid`. Learned policies
are not registry IDs. Supply a checkpoint path, algorithm ID, expected SHA-256
digest, and Track policy contract to the canonical checkpoint loader.

Supported learned formats are SAC, TD3, PPO, BC, RLPD, and ONNX. The digest is
checked before importing or deserializing the backend. The returned controller
converts normalized actions to physical actuator commands exactly once.

## Dataset and training contract

Dataset v2 is the only live dataset schema. Collection writes complete episodes
with observations, true state, commanded and applied actions, scalar reward,
reward terms, cost channels, termination flags, bootstrap masks, identities,
and checksums. The optional Minari adapter is available only from
`aiogym.datasets.minari_adapter`.

All algorithms use `RLTrainingConfig` and `run_experiment`. Unknown
algorithm-specific, replay, evaluation, checkpointing, or output fields fail
validation. Training and tuning may use only training and validation splits.
Checkpoint continuation supports `resume_mode="restart_episode"`: optimizer,
replay, random-number, and episode-coordinator state are restored, while any
partial episode is discarded and restarted from the next assigned episode.

## CLI

The stable workflow commands support guided selectors and exact config-first
forms:

```text
aiogym list tracks
aiogym list rewards
aiogym list profiles
aiogym describe track TARGET
aiogym describe reward REWARD
aiogym collect TARGET --profile quick
aiogym train TARGET ALGORITHM --profile quick
aiogym collect --config FILE
aiogym train --config FILE
aiogym evaluate RUN
aiogym evaluate --checkpoint PATH --algorithm ID --track TRACK
aiogym benchmark --run RUN
aiogym benchmark --config FILE
aiogym final-test --config FILE
aiogym design
aiogym design --advanced
aiogym design new NAME [--advanced] [--run]
aiogym design run --interactive [--save-spec FILE]
aiogym design validate SPEC
aiogym design run SPEC --output DIR
aiogym design sweep SPEC --parameter PATH --values CSV --output DIR
aiogym design report RESULT --output DIR
```

`benchmark` accepts an official Track, the PID/MPC baselines, and at most one
learned checkpoint through `--checkpoint`, `--algorithm`, `--sha256`, and
`--name`, or one self-describing run manifest through `--run`. Backend-specific
checkpoint flags and direct scenario/Case/reward overrides are not supported.

Selectors are ergonomic inputs only. Resolved configs and artifacts contain
canonical Track and RewardSpec IDs. `quick-v1` is a tutorial/smoke profile;
formal experiments should use an audited config until a verified baseline
profile is registered. Guided profile declarations are installed package
resources; files retained under the repository's `configs/` tree are examples,
not the installed profile source.

`final-test` is the only ordinary command that can consume a test split. Its
configuration binds the Track hash, checkpoint digests, seed list, and
one-shot lock.

Design-study commands operate on user-supplied equipment and are deliberately
outside official Track ranking. Their reports carry exact DesignSpec hashes and
state `simulation-screening-only` evidence limits.

## Optional dependencies

| Extra | Capability |
|---|---|
| `rl` | Torch and Stable-Baselines3 training/loading |
| `onnx` | ONNX export and runtime loading |
| `oracle` | CasADi oracle controller |
| `hpo` | Optuna tuning |

The core import and environment path do not import any of these dependencies.
