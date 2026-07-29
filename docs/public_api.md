# Public API

## Discovery

```python
import aiogym

aiogym.list_scenarios()
aiogym.list_cases()
aiogym.list_cases("cascade")
aiogym.list_tracks()
aiogym.list_reward_specs()
aiogym.list_controllers()
```

## Cases

```python
case = aiogym.load_case("quadruple/minimum-phase")
env = aiogym.AIOGymEnv(
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
    scenario="cstr",
    case=None,
    reward_spec="economic-v1",
    action_mode="actuator",
    control_dt=0.5,
    episode_steps=400,
)
```

`AIOGymEnv` is the single environment class. Its constructor accepts Scenario,
Case, RewardSpec, action/observation options, timing, randomization, and
scenario-specific parameters. Goal is derived from RewardSpec and cannot
conflict with evaluation.

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
