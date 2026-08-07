# Quickstart

## Discover and create

```bash
aiogym list scenarios
aiogym list tasks --scenario three_tank
aiogym list plants --scenario three_tank
```

```python
import aiogym

env = aiogym.make_env("quadruple/regulation", condition="minimum-phase")
observation, info = env.reset(seed=0)
next_observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

## Evaluate and collect

```python
env = aiogym.make_env(
    "three_tank/regulation",
    plant="open-cascade-v1",
    condition="continuous-benchmark",
)
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(pid, seeds=[0, 1, 2])
env.close()

dataset = aiogym.collect(
    task="quadruple/regulation",
    condition="minimum-phase",
    policy="pid",
    episodes=3,
    seed=0,
    output="runs/data/quadruple-pid-v4",
)
```

Dataset v4 stores checksummed episode `.npz` files. Its manifest records Task,
Plant, Condition, interface, policy contracts, transfer flags, schemas, seeds,
and `schedule_semantics = pre-action-v1`.

## Explicit checkpoint transfer

Checkpoint transfer is opt-in for Plant or Condition changes:

```python
result = aiogym.evaluate(
    checkpoint_policy,
    task="three_tank/regulation",
    plant="recirculating-h1-v1",
    condition=custom_condition,
    allow_condition_transfer=True,
)
```

Task and interface hashes must still match. A six-action checkpoint cannot run
on a four-action Plant. Bound PID/MPC policies never transfer implicitly; build
a new controller against the target environment.

## Train

```python
run = aiogym.train(
    task="quadruple/regulation",
    algorithm="sac",
    steps=10_000,
    condition="minimum-phase",
    seed=0,
    eval_seeds=[100, 101, 102],
    output="runs/train/quadruple-sac",
)
```

Short budgets prove pipeline execution only, not control performance.
