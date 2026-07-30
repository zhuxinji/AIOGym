# AIO-Gym

AIO-Gym is a native Gymnasium backend for process-control research. It provides
eight process models, versioned operating Cases, canonical RewardSpecs,
goal-based evaluation, reproducible benchmark Tracks, classical controllers,
and RL training entry points.

## Install

```bash
pip install -e .
```

Optional controller and RL dependencies are exposed through the project extras
defined in `pyproject.toml`.

## Core vocabulary

- **Scenario**: process dynamics, physical parameters, inputs, outputs,
  constraints, and economic declarations.
- **Case**: one scenario-bound experiment: initialization, timing, references,
  disturbances, operating conditions, and optional acceptance thresholds.
- **RewardSpec**: the scalar training signal returned by `env.step()`.
- **Goal**: evaluation intent and primary ranking metric. Supported values are
  `regulation` and `economic`.
- **Scorecard**: reward-independent regulation, economics, safety, service,
  robustness, and controller measurements.
- **Track**: an official benchmark contract combining one scenario, a Case
  distribution, Goal, RewardSpec, policy interface, seed namespaces, ranking,
  and safety rules.

Cases deliberately do not select rewards or ranking. RewardSpecs deliberately
do not define the benchmark population. Tracks join those independent pieces.

## Environment use

```python
import aiogym

env = aiogym.make_env("quadruple", case="minimum-phase")
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Advanced timing, observation, and realism settings use mutually exclusive
config mode:

```python
env = aiogym.make_env(
    config={
        "scenario": "cascade",
        "case": "continuous-benchmark",
        "reward_spec": "economic-v1",
        "environment": {"episode_steps": 400, "control_dt": 0.5},
    }
)
```

`make_env()` is the only public environment-construction API. The concrete
Gymnasium environment class consumes an immutable resolved spec and is an
implementation detail. Seed only through `env.reset(seed=...)`.

## Discovery

```python
aiogym.list_scenarios()
aiogym.list_cases()
aiogym.list_cases("quadruple")
aiogym.list_tracks()
aiogym.list_controllers()
```

Advanced registries are explicit submodule APIs, for example
`from aiogym.rewards import list_reward_specs`.

CLI equivalents:

```bash
aiogym list scenarios
aiogym list cases
aiogym list cases --scenario quadruple
aiogym list tracks
aiogym list controllers
```

## Evaluation

```python
controller = aiogym.make_controller("pid", scenario="cstr")
env = aiogym.make_env(
    config={
        "scenario": "cstr",
        "reward_spec": "regulation-v1",
        "environment": {"episode_steps": 100},
    }
)

result = aiogym.evaluate_controller(
    controller,
    env,
    episodes=3,
    seed=100,
)

print(result["goal"])
print(result["metric"], result[result["metric"]])
print(result["scorecard"])
env.close()
```

`return` is comparable only when `reward_spec_id` matches. Official ranking uses
the Track Goal utility, a committed fixed-anchor manifest, the declared
cross-case aggregation, and the Track-owned safety gate—not raw training
return. Any gate failure has official score `0`.

## Benchmarks

Run an official Track:

```bash
aiogym benchmark \
  --config configs/benchmark/quadruple-validation-v1.json
```

Run one explicit specialist Case:

```bash
aiogym benchmark quadruple \
  minimum-phase \
  --goal regulation \
  --reward-spec regulation-v1 \
  --controllers pid
```

Use `--output FILE` to write the JSON result. Official Tracks validate their
policy contract before evaluation and record Track/Case hashes and seed
namespaces for provenance.

Anchor updates are an explicit review workflow. Calibration writes a candidate;
official evaluation never recalculates anchors:

```bash
aiogym benchmark calibrate-anchors \
  --track quadruple-regulation-generalist-v1 \
  --bad-controller hold \
  --reference-controller pid \
  --output /tmp/quadruple-anchors-candidate.json
```

Bundled Tracks:

- `quadruple-regulation-generalist-v1`
- `cascade-regulation-generalist-v1`
- `cascade-economic-specialist-v1`
- `cascade-recovery-diagnostic-v1`
- `cascade-recirculating-regulation-generalist-v1`
- `cascade-recirculating-recovery-diagnostic-v1`

## RL training

```bash
aiogym train --config configs/train/quadruple-sac-v1.json
aiogym train --config configs/train/quadruple-sac-v1.json --seeds 0,1,2
aiogym tune --config configs/tune/quadruple-sac-v1.json
```

The config selects SAC, TD3, PPO, RLPD, or BC. Training resolves the Track and
fixed validation plan before constructing environments; formal test remains
available only through `aiogym final-test`.

## Artifacts

Current benchmark artifacts use `case`, `goal`, `reward_spec_id`, `track_id`,
`official_score`, and `scorecard`. Validate or render them with:

```bash
aiogym artifacts check RUN_DIR
aiogym artifacts report RUN_DIR
```

Archived pre-redesign files can be read only through the explicit offline
migration module:

```python
from aiogym.evaluation.legacy_artifacts import (
    load_legacy_evaluation_artifact,
)
```

That module is not imported by environment, evaluation, Track, artifact writer,
or RL execution paths.

## Development

```bash
python3 -m pytest -q
python3 -m compileall -q aiogym
```

See [Concepts](docs/concepts.md), [Public API](docs/public_api.md),
[Architecture](docs/architecture.md), and
[Benchmark semantics ADR](docs/adr/0001-benchmark-semantics-v2.md).
Breaking migrations are summarized in
[API compatibility](docs/api_compatibility.md).
