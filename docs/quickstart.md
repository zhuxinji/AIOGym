# Quickstart

This guide follows the common Python workflow from environment creation through
PID/MPC/RL comparison. Choose a built-in scenario first, then use the same
workflow throughout. Physical variables, supported parameter overrides,
disturbances, controller settings, and recommended training budgets belong to
the [scenario guides](../README.md#included-scenarios).

## 1. Install AIO-Gym

From the repository root:

```bash
pip install '.[rl]'
```

This installs simulation, PID, MPC, data collection, evaluation, and the
reinforcement-learning dependencies used later in this guide.

## 2. Choose a scenario

```python
import aiogym

scenarios = aiogym.list_scenarios()
scenario = scenarios[0]  # Replace this with any returned scenario id.

print(scenarios)
print(aiogym.list_algorithms())
print(aiogym.list_benchmarks(scenario))
print(aiogym.list_rewards(scenario))
print(aiogym.list_parameters(scenario))
```

The scenario guide explains its observation, action, physical units, accepted
initial state, parameters, disturbances, Benchmarks, and practical limits.

## 3. Run an environment

```python
import aiogym

scenario = aiogym.list_scenarios()[0]
env = aiogym.make_env(scenario)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

An ordinary environment may also accept `initial_state=` and `parameters=`.
Their values, order, and physical units are scenario-specific, so copy them
from the selected scenario guide. Randomized training environments choose their
own initial states, while fixed Benchmarks own their complete test cases.

## 4. Compare PID and MPC

```python
import aiogym

scenario = aiogym.list_scenarios()[0]
env = aiogym.make_env(scenario, benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)

comparison = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc},
    seeds=range(20),
)
env.close()

print(comparison["ordering"])
```

Both controllers receive the same 20 physical test cases. The comparison is
written to:

```text
runs/<scenario>/benchmarks/tracking/
├── comparison.json
├── comparison.svg
└── trajectories.npz
```

Use `comparison.svg` for a quick review, `comparison.json` for scores and
summaries, and `trajectories.npz` for complete numeric trajectories.

PID, MPC, and loaded learned policies receive the same environment observation.
At each action the policy context contains only `step_index`, `physical_time`,
and the current public `reference`. Diagnostic fields such as true state,
physical disturbance factors, applied action, safety margins, and the complete
episode schedule remain in `info` and saved artifacts, but are not policy
inputs.

## 5. Collect a Dataset

```python
from pathlib import Path

import aiogym

scenario = aiogym.list_scenarios()[0]
run_scenario = scenario.replace("_", "-")
dataset_path = Path("runs") / run_scenario / "datasets" / "pid-20" / "seed-0"

env = aiogym.make_env(scenario, randomize=True)
pid = aiogym.make_controller("pid", env=env)
dataset = aiogym.collect(
    env=env,
    policy=pid,
    episodes=20,
    seed=0,
    output=dataset_path,
)
env.close()

reader = aiogym.DatasetReader(dataset_path)
print(len(reader), reader.transition_count)
```

## 6. Train SAC

```python
from pathlib import Path

import aiogym

scenario = aiogym.list_scenarios()[0]
run_scenario = scenario.replace("_", "-")
run_path = (
    Path("runs")
    / run_scenario
    / "training"
    / "sac"
    / "sac-100k"
    / "seed-0"
)

training_env = aiogym.make_env(scenario, randomize=True)
evaluation_env = aiogym.make_env(scenario, randomize=True)
trained = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=100_000,
    seed=0,
    record_every=500,
    evaluation_env=evaluation_env,
    evaluate_every=5_000,
    output=run_path,
)
training_env.close()
evaluation_env.close()

print(trained["checkpoint"])
```

Training writes a final `model.zip`, a learning curve, metadata, and—because
periodic evaluation is enabled—a `best/model.zip` checkpoint. Selection follows
the shared [best-checkpoint protocol](architecture.md#select-the-best-checkpoint)
on a separate randomized validation environment. Formal Benchmark cases remain
held out from this selection.

Store runs as
`runs/<scenario>/training/<algorithm>/<experiment>/seed-<n>` so checkpoints
from different algorithms remain separate. Scenario guides may recommend a
different step budget or algorithm configuration for meaningful performance.

## 7. Compare the learned policy

```python
from pathlib import Path

import aiogym

scenario = aiogym.list_scenarios()[0]
run_scenario = scenario.replace("_", "-")
checkpoint = (
    Path("runs")
    / run_scenario
    / "training"
    / "sac"
    / "sac-100k"
    / "seed-0"
    / "best"
    / "model.zip"
)

env = aiogym.make_env(scenario, benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
sac = aiogym.load_policy(checkpoint, env=env)
comparison = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc, "sac": sac},
    seeds=range(20),
)
env.close()
```

## 8. Continue training

```python
from pathlib import Path

import aiogym

scenario = aiogym.list_scenarios()[0]
run_scenario = scenario.replace("_", "-")
source = (
    Path("runs")
    / run_scenario
    / "training"
    / "sac"
    / "sac-100k"
    / "seed-0"
    / "model.zip"
)
output = (
    Path("runs")
    / run_scenario
    / "training"
    / "sac"
    / "sac-150k"
    / "seed-0"
)

training_env = aiogym.make_env(scenario, randomize=True)
continued = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=50_000,
    resume_from=source,
    output=output,
)
training_env.close()
```

Here, `steps=50_000` adds 50,000 environment steps. The new run keeps the
source checkpoint unchanged. Use the same scenario, training settings, seed,
and—for RLPD—the same Dataset.

## Add training variation

Training environments can vary operating conditions and add common process or
measurement problems:

```python
import aiogym

scenario = aiogym.list_scenarios()[0]
env = aiogym.make_env(
    scenario,
    randomize=True,
    boundary_probability=0.30,
    disturbance=True,
    noise=True,
    delay=True,
    fault=True,
)
```

- `randomize` varies feasible interior initial conditions and targets;
- `boundary_probability` selects the fraction of randomized episodes that use
  the scenario's reachable boundary initializer (default `0.0`);
- `disturbance` adds a temporary physical disturbance;
- `noise` affects measured observation channels;
- `delay` adds observation or action delay;
- `fault` applies temporary actuator loss of effectiveness.

Pass a mapping instead of `True` when you need explicit settings. These options
are for training and ordinary simulation; fixed Benchmarks keep their own test
conditions so policies remain directly comparable. Available disturbance names,
schedule examples, event timing, and physical meaning are documented by each
scenario.

## Read the main metrics

- `unsafe_rate`: lower is better;
- `safe_completion`: higher is better;
- `settling_rate`: fraction of cases that enter and remain inside every settling
  band; higher is better;
- `return`: higher is better when policies use the same Reward;
- `tracking_iae` and `tracking_ise`: lower tracking error is better;
- `final_error`: lower endpoint error is better.

Formal comparisons rank mean safety metrics across all cases before performance
metrics. The report's `ranking_metrics` rows show the exact metric order and
aggregate used by the selected Benchmark. Scenario guides define their physical
settling bands, Reward terms, and any additional ranking rules.

Short training runs only show that the workflow executes. For performance
claims, use fixed Benchmarks, identical seeds, and enough training and test
cases to support the conclusion.

Keep these interpretation limits in mind:

- a good median return does not cancel unsafe episodes;
- results using different Rewards or model parameters are not directly
  comparable;
- learned-policy comparisons use the checkpoint selected by the shared
  validation protocol and still make final claims on held-out Benchmark cases;
- simulation safety does not establish safety on real equipment.

## Optional command-line use

The same workflows are available through the `aiogym` command for shell scripts
and batch jobs. Use `aiogym --help` and `aiogym <command> --help` when needed;
the documentation uses Python as the primary interface. In CLI automation,
`boundary_probability=0.30` is written as `--boundary-probability 0.30` and
requires `--randomize`.
