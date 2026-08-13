# Quickstart

## Inspect and run an environment

```python
import aiogym

print(aiogym.list_scenarios())
print(aiogym.list_benchmarks("quadruple"))
print(aiogym.list_parameters("quadruple"))

env = aiogym.make_env(
    "quadruple",
    parameters={"pump_gain": [3.2, 3.25]},
)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

## Collect varied training data

```python
env = aiogym.make_env(
    "quadruple",
    randomize=True,
    noise=True,
    delay=True,
    fault=True,
)
pid = aiogym.make_controller("pid", env=env)
aiogym.collect(env=env, policy=pid, episodes=2, seed=0, output="runs/data")
env.close()

reader = aiogym.DatasetReader("runs/data")
print(len(reader), reader.transition_count)
```

## Train, load, evaluate, and compare

Install `aiogym[rl]`, then run:

```python
training_env = aiogym.make_env("quadruple", randomize=True, noise=True)
trained = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=64,
    seed=0,
    record_every=500,
    output="runs/sac",
)
aiogym.plot_training_curve(
    trained["training_curve"],
    output="runs/replotted-training-curve.svg",
)
sac = aiogym.load_policy(
    trained["checkpoint"], algorithm="sac", env=training_env
)
training_env.close()

benchmark_env = aiogym.make_env("quadruple", benchmark="tracking")
pid = aiogym.make_controller("pid", env=benchmark_env)
mpc = aiogym.make_controller("mpc", env=benchmark_env)
evaluation = aiogym.evaluate(
    env=benchmark_env, policy=sac, seeds=[0, 1, 2]
)
comparison = aiogym.compare_policies(
    env=benchmark_env,
    policies={"pid": pid, "mpc": mpc, "sac": sac},
    seeds=[0, 1, 2],
)
benchmark_env.close()
```

Short RL runs verify the pipeline; they are not performance evidence. The
default comparison directory is `runs/<scenario>/<benchmark>/`; rerunning the
same benchmark replaces its machine-readable `comparison.json` and trajectory
figure `comparison.svg`. Pass an explicit empty `output` directory when those
artifacts must not be overwritten.

## CLI

```bash
aiogym list benchmarks --scenario quadruple
aiogym list parameters --scenario quadruple
aiogym collect quadruple --randomize --noise on --delay on --fault on \
  --controller pid --episodes 2 --output runs/data
aiogym train quadruple sac --randomize --noise on --steps 64 \
  --record-every 500 --output runs/sac
aiogym evaluate quadruple --benchmark tracking --controller pid --seeds 0 1 2 \
  --output runs/pid-evaluation.json
aiogym compare quadruple --benchmark tracking --controllers pid mpc \
  --seeds 0 1 2
```

Use `--parameters parameters.json` on any workflow command to apply one JSON
object of model parameter overrides.
