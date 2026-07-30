# Public API

## Discovery

```python
import aiogym

aiogym.list_scenarios()
aiogym.list_cases()
aiogym.list_cases("cascade")
aiogym.list_tracks()
aiogym.list_controllers()
```

Advanced registries are imported from their subpackages:

```python
from aiogym.rewards import list_reward_specs
from aiogym.generation import DistributionSpec, EpisodeSpec
from aiogym.datasets import DatasetReader, DatasetWriter
```

## Cases

```python
case = aiogym.load_case("quadruple/minimum-phase")
env = aiogym.make_env(
    "quadruple",
    case=case,  # a mapping or the short name is accepted
    reward_spec="regulation-v1",
)
```

`load_case(source, scenario=None, overrides=None)` accepts a canonical
`scenario/name` ID, a path, or a mapping. Overrides are limited to Case-owned
sections and produce a new validated profile with a distinct hash.

## Environment

```python
env = aiogym.make_env(
    config={
        "scenario": "cstr",
        "reward_spec": "economic-v1",
        "environment": {
            "action_mode": "actuator",
            "control_dt": 0.5,
            "episode_steps": 400,
        },
    }
)
```

`make_env()` is the only public environment-construction API. It accepts
Scenario, Case, and RewardSpec in direct mode. Advanced action/observation,
timing, and realism settings use mutually exclusive config mode. The concrete
Gymnasium environment class only accepts an immutable `ResolvedEnvSpec`.
Seeding belongs to `env.reset(seed=...)`.

## Episode generation

```python
from aiogym.generation import FixedCaseEpisodeSampler

sampler = FixedCaseEpisodeSampler(
    "quadruple/minimum-phase",
    split="validation",
)
episode_spec = sampler.sample(23)
observation, info = env.reset(
    options={"episode_spec": episode_spec},
)
```

`DistributionSpec` describes a versioned probability distribution.
`EpisodeSpec` is a fully resolved immutable episode with stable hashes and
component seeds. `SeedTree` separates plant, initial-state, reference,
disturbance, sensor, actuator, exploration, and policy random streams, while
training, validation, and test use distinct namespaces.

Config migration may still resolve historical randomization fields, but the
runtime no longer exposes a legacy distribution. New training code samples a
versioned distribution before reset and injects the resolved `EpisodeSpec`.

The built-in quadruple training distribution supports fixed L0–L4 difficulty
levels:

```python
track = aiogym.load_track("quadruple-regulation-generalist-v1")
distribution = track.training_distribution()
from aiogym.generation import QuadrupleTrainingSampler

sampler = QuadrupleTrainingSampler(distribution)
episode_spec = sampler.sample(seed=1000, episode_index=17)
```

Its parameter sampler preserves declared outlet-area, pump-gain, and split
fraction correlations. Initial states are sampled around an equilibrium for
the resolved plant. Reference schedules are generated through the steady-state
action map so every target is reachable. Disturbance processes are compiled
into bounded deterministic schedules. `QuadrupleCurriculumSampler` selects
L0–L4 from cumulative environment transitions.

## Dataset v2

```python
from aiogym.datasets import DatasetReader, DatasetWriter, collect_episode

episode = collect_episode(
    env,
    episode_spec,
    collector_id="safe_excitation",
    track_id=track.id,
)
with DatasetWriter(
    "runs/datasets/quadruple-diverse-v1",
    dataset_id="quadruple-diverse-v1",
    split="training",
) as writer:
    writer.append_episode(episode)

reader = DatasetReader("runs/datasets/quadruple-diverse-v1")
batch = reader.random_batch(256)
```

Dataset v2 stores one validated episode per compressed atomic shard. The
manifest records checksums, episode boundaries, resolved distribution hashes,
collector provenance, termination semantics, reward terms, and cost channels.
The superseded in-memory transition API is not part of `aiogym.rl`. Existing
`aiogym.transition.v1` rows can be passed through the one-way migration adapter
in `aiogym.compat.transitions`; new collection and training code must use
Dataset v2 directly.

Dataset CLIs:

```bash
aiogym collect --config configs/datasets/quadruple-mixed-pilot-v1.json
python -m aiogym.datasets.inspect PATH --quality
python -m aiogym.datasets.validate PATH
```

Collection configs use schema `aiogym.dataset_collection.v1` and declare the
Track, training-transition target, worker count, output, and explicitly
weighted collectors. Episodes are assigned global indexes in the parent
process and committed in index order, so changing worker count does not change
episode or collector identities. The transition target permits one complete
final episode to cross the requested budget; both requested and actual counts
are recorded in the manifest.

The built-in collector IDs resolve to behavior adapters rather than metadata
labels: randomized and noisy PID use the EpisodeSpec policy seed, checkpoint
mixtures verify and record SHA256 hashes, and `safe_excitation` passes every
proposal through the versioned projection shield.

## Reinforcement-learning training

`aiogym.rl.RLTrainingConfig` v2 is the canonical, hashable configuration for
SAC, TD3, PPO, RLPD, and BC. It includes output identity, fixed validation
seeds, resume mode, dataset identity, and checkpoint cadence. Its transition
budget and explicit `utd_ratio` are expressed in environment transitions.

```bash
aiogym train --config configs/train/quadruple-sac-v1.json
aiogym tune --config configs/tune/quadruple-sac-v1.json
```

`EpisodeCoordinator` assigns monotonic global episode indexes independently of
worker placement. `NormalizedActionWrapper` exposes `[-1, 1]` to every
algorithm while retaining normalized, commanded, and applied actions.
Algorithm-specific execution is behind the stable `run_experiment()` adapter
boundary. Backend-specific trainer subcommands are not public.

`CheckpointManager` stores framework model/optimizer and replay references,
normalization statistics, counters, the real training coordinator state,
curriculum and validation state, configuration identity, and process RNG
state. SB3 and RLPD use the explicit `restart_episode` resume mode: active
partial episodes are recorded as discarded and resume starts at the next
unused EpisodeSpec. Multi-process backends do not claim bitwise continuation
of simulator state.

Training and tuning evaluate only the fixed validation plan. The benchmark
test split is available solely through the one-shot command:

```bash
aiogym final-test --config configs/final-test.json
```

Its config locks Track/config identity, base seeds, and the SHA256 of every
checkpoint. The lock is consumed even when evaluation fails, and a successful
artifact records both lock state and statistical-report hash.

Simulator baselines can be collected with:

```bash
python -m aiogym.tools.benchmark_env \
  --track quadruple-regulation-generalist-v1 \
  --n-envs 1,2,4,8,16 \
  --steps 100000
```

## Offline and hybrid RL

`DatasetReplay` samples Dataset v2 shards without loading the complete prior
into memory. Canonical sampling is uniform across
`collector_id × difficulty_tag` strata and transition-weighted inside each
stratum. Dataset ID and manifest hash are included in checkpoints and training
artifacts.

`BehaviorCloningTrainer` is a data-alignment sanity check: it verifies that
policy observations reproduce `action_policy_normalized`, rather than serving
as a final benchmark algorithm.

BC uses the same `aiogym train --config ...` entry point with
`"algorithm_id": "bc"` and a bound Dataset v2 path/hash.

RLPD v2 keeps immutable offline replay and mutable online replay separate.
Canonical batches are exactly 50% offline and 50% online once online data
exists; different ratios require an explicit variant flag. Time-limit
truncations retain `bootstrap_mask=1`, whereas true terminals use zero.

RLPD uses the same config-first command with `"algorithm_id": "rlpd"`,
`dataset_path`, and optional expected `dataset_id`/`dataset_hash`.

RLPD checkpoints contain networks, target networks, all optimizers, entropy
state, the online replay state, the immutable Dataset v2 reference,
Torch/NumPy RNG, and offline/online sample accounting.

## Validation, HPO, and final statistics

`ValidationEpisodePlan` resolves every validation Case/seed pair to a fixed
EpisodeSpec. The same plan hash and EpisodeSpec hashes are shared across
algorithms. `CompleteValidationCallback` always evaluates the complete
validation Track and `EligibilityAwareSelector` excludes unsafe checkpoints
before comparing IQM, worst-case performance, and intervention cost. These
selection APIs reject test-split input.

`OptunaStudyRunner` creates a persistent, resumable study and reports
intermediate validation utility for pruning. Suggestions may change only
algorithm hyperparameters; Track, RewardSpec, safety gates, splits,
normalization anchors, and distributions remain immutable.

Install its optional dependency with:

```bash
pip install 'aiogym[hpo]'
```

`build_final_statistical_report()` consumes locked test evaluations with
per-episode data and produces:

- a per-seed × per-case matrix for every algorithm;
- IQM with stratified bootstrap confidence intervals;
- mean, median, worst-case, and tail summaries;
- paired probability of improvement;
- performance profiles.

`FinalTestLock` binds the Track hash, training config hash, checkpoint hashes,
and test seeds before evaluation. Its one-shot `run()` method marks the lock as
consumed before reading test and cannot be rerun, including after a failed
attempt.

## Realism, partial observation, and safety

An EpisodeSpec sensor model may be `identity`, `additive_gaussian`, or
`sensor_dynamics_v1`. The latter supports resolved noise, bias, drift, delay,
quantization, and hold-last-value dropout. Actuator models may be `identity` or
`actuator_dynamics_v1`, with efficiency, bias, command delay, deadband,
first-order lag, and slew limits. The environment records commanded and applied
physical actions separately.

Tracks now explicitly identify their sensing and temporal policy contract:

```python
from aiogym.experimental.rl import (
    ObservationContract,
    wrap_observation_contract,
)

contract = ObservationContract(
    sensing="measured_output",
    temporal="history",
    history_length=4,
    include_action_history=True,
)
wrapped = wrap_observation_contract(env, contract)
```

Supported temporal contracts are `single_step`, `history`, and `recurrent`.
`RecurrentStateContract` defines batched hidden-state shape and mandatory reset
semantics at episode boundaries. Learned-policy contexts contain measured
information only and do not expose the underlying environment.

An auditable projection shield can be composed without changing the policy:

```python
from aiogym.experimental.rl import (
    ProjectionSafetyShield,
    SafetyShieldWrapper,
)

shield = ProjectionSafetyShield(
    low=0.05,
    high=0.95,
    max_delta_per_step=0.1,
)
safe_env = SafetyShieldWrapper(env, shield)
```

Every transition retains the raw policy proposal, shielded command, final
actuator action, intervention reasons, duration, and magnitude. A
`SafetyGateSpec(cost_budgets=...)` makes declared cost budgets part of
validation eligibility.

`LagrangianSAC` is the constrained baseline. It uses independent reward and
cost critics, a projected dual multiplier, a cost-aware replay buffer, and
complete resumable state. Cost channels remain separate from RewardSpec:

```python
from aiogym.experimental.rl import LagrangianSAC

agent = LagrangianSAC(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    cost_limit=0.02,
    cost_channels=("soft_safety", "hard_safety"),
)
```

Final statistical reports include an intervention report with per-seed ×
per-case matrices, intervention probability/rate, P90/P95 severity, and
CVaR-style upper-tail severity for protection, shield, and actuator channels.

## Controllers

```python
controller = aiogym.make_controller(
    "pid",
    scenario="quadruple",
    config={
        "profile": "quadruple-minimum-phase-benchmark",
        "case": "minimum-phase",
        "policy_scope": "specialist",
        "goal": "regulation",
        "reward_spec": "regulation-v1",
    },
)
```

Built-ins include PID, MPC, nonlinear MPC Oracle, SB3, ONNX, and generic policy
adapters where optional dependencies are installed.

## Evaluation

```python
result = aiogym.evaluate_controller(
    controller,
    env,
    episodes=3,
    seed_list=[11, 12, 13],
    goal_specification="regulation",
)
```

The result contains:

- `schema_version`, `goal`, `reward_spec_id`;
- controller, environment, model, Case, and reproducibility metadata;
- episode aggregates and optional per-episode rows;
- flat metrics plus grouped `scorecard`;
- `metric`, `metric_direction`, `official_score`;
- safety gate and ranking eligibility.

`rollout_controller()` records one scenario-neutral transition sequence with
Goal and RewardSpec metadata.

## Tracks

```python
track = aiogym.load_track("quadruple-regulation-generalist-v1")
payload = aiogym.evaluate_policy_on_track(
    controller,
    track,
    split="validation",
)
```

`load_track()` validates Case existence, RewardSpec/Goal consistency, policy
contract consistency, seed namespaces, ranking, and safety configuration.
Built-in Tracks bind their declared safety gate automatically and reject
runtime overrides. Fixed-anchor Tracks expose both raw rate metrics and a
normalized `aggregate["official_score"]`; checkpoint selection and HPO use the
latter.

## Artifacts

```python
aiogym.check_benchmark_artifacts(path)
aiogym.render_benchmark_report(path)
aiogym.plot_results(path)
```

New artifacts are written without compatibility normalization. Historical file
inspection is an explicit offline action:

```python
from aiogym.evaluation.legacy_artifacts import (
    load_legacy_evaluation_artifact,
)
```

The migration reader is deliberately absent from the top-level API.
