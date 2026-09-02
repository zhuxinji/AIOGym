# RLPD training

RLPD combines a fixed Dataset with new online experience. Use it when you have
controller demonstrations or logged simulation data and want those transitions
to remain part of reinforcement learning instead of serving only as a one-time
warm start.

AIO-Gym follows the method from [Ball et al. (2023)](https://arxiv.org/abs/2302.02948)
and the authors' [reference implementation](https://github.com/ikostrikov/rlpd),
adapted to the common AIO-Gym Dataset, checkpoint, and evaluation workflow.

Choose the scenario and output root once. Its scenario guide gives the suitable
controller, episode count, and training budget for that process.

```python
from pathlib import Path

import aiogym

scenario = aiogym.list_scenarios()[0]
run_scenario = scenario.replace("_", "-")
root = Path("runs") / run_scenario
dataset_path = root / "datasets" / "pid-100" / "seed-0"
run_path = root / "training" / "rlpd" / "rlpd-100k" / "seed-0"
```

## Collect the fixed Dataset

```python
env = aiogym.make_env(scenario, randomize=True)
pid = aiogym.make_controller("pid", env=env)
dataset = aiogym.collect(
    env=env,
    policy=pid,
    episodes=100,
    seed=0,
    output=dataset_path,
)
env.close()
```

The Dataset must match the training environment's scenario, reward, physical
parameters, training variation, control interval, and observation/action
shapes. Collecting more episodes improves coverage but does not replace
held-out evaluation.

## Train RLPD

```python
training_env = aiogym.make_env(scenario, randomize=True)
evaluation_env = aiogym.make_env(scenario, randomize=True)
trained = aiogym.train(
    env=training_env,
    algorithm="rlpd",
    dataset=dataset_path,
    steps=100_000,
    seed=0,
    record_every=500,
    evaluation_env=evaluation_env,
    evaluate_every=5_000,
    output=run_path,
)
training_env.close()
evaluation_env.close()
```

RLPD samples from both the fixed Dataset and the growing online replay during
training. The default split is 50/50. Set `offline_ratio` in
`algorithm_kwargs` to a value from 0 to 1 when another balance is needed.

The reference-style defaults are computationally demanding: batch size 256,
10 critics, update-to-data ratio 20, and 10,000 random online transitions before
gradient updates. Lower `utd_ratio` and `n_critics` for a local smoke test, but
do not treat that short run as performance evidence.

## Continue training

```python
continued_path = root / "training" / "rlpd" / "rlpd-150k" / "seed-0"
training_env = aiogym.make_env(scenario, randomize=True)
continued = aiogym.train(
    env=training_env,
    algorithm="rlpd",
    dataset=dataset_path,
    steps=50_000,
    resume_from=run_path / "model.zip",
    output=continued_path,
)
training_env.close()
```

Use the same Dataset and training settings. `steps` is the additional online
environment-step budget, and the source run remains unchanged. Continuation
restores the recorded seed, critics, optimizers, online replay, and fixed
offline data reference.

## Evaluate the result

Use the selected `best/model.zip` for formal comparison and retain the
top-level final checkpoint for continuation. The common selection rule is
defined in [Select the best checkpoint](architecture.md#select-the-best-checkpoint).

```python
env = aiogym.make_env(scenario, benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
rlpd = aiogym.load_policy(run_path / "best" / "model.zip", env=env)
comparison = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc, "rlpd": rlpd},
    seeds=range(20),
)
env.close()
```

Report the Dataset source and coverage, online environment-step budget, main
algorithm settings, and safety results alongside tracking performance.
