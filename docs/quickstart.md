# Quickstart

## Discover and create

```bash
aiogym list scenarios
aiogym list tasks --scenario quadruple
```

```python
import aiogym

env = aiogym.make_env("quadruple/regulation", preset="minimum-phase")
observation, info = env.reset(seed=0)
action = env.action_space.sample()
next_observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Task IDs use `<scenario>/<objective>`. `plant=` accepts a `PlantConfig`, a
mapping, or a JSON path. Omitting it uses the scenario's built-in plant.

## Evaluate a controller

```python
import aiogym

env = aiogym.make_env("cascade/regulation", preset="continuous-benchmark")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(pid, seeds=[0, 1, 2], output="runs/evaluate/cascade-pid")
env.close()
print(result["aggregate"])
```

## Collect Dataset v3

```python
dataset = aiogym.collect(
    task="quadruple/regulation",
    preset="minimum-phase",
    policy="pid",
    episodes=3,
    seed=0,
    output="runs/data/quadruple-pid-v3",
)
```

Each episode is one checksummed `.npz` file; `manifest.json` records Task,
Plant, policy, seeds, array schemas, and aggregate counts. Use
`aiogym.workflows.DatasetReader` for direct reading.

## Train and evaluate

```python
run = aiogym.train(
    task="quadruple/regulation",
    algorithm="sac",
    steps=10_000,
    preset="minimum-phase",
    seed=0,
    eval_seeds=[100, 101, 102],
    output="runs/train/quadruple-sac",
)
```

This saves an SB3 checkpoint and evaluates the reloaded policy. A short budget
only validates the pipeline; it is not credible research-scale evidence.
