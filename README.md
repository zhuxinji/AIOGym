# AIO-Gym

AIO-Gym is a compact process-control pipeline for simulation, classical
controllers, reinforcement learning, datasets, and policy comparison.

```python
import aiogym

env = aiogym.make_env("three_tank")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=[0, 1, 2])
env.close()
```

The built-in scenarios are `quadruple` and `three_tank`. A Reward supplies the
training signal. Every scenario owns exactly three fixed evaluation Benchmarks:
`tracking`, `disturbance-rejection`, and `boundary-safety`.

## Install

```bash
pip install .
pip install 'aiogym[rl]'        # Stable-Baselines3 and Torch
pip install 'aiogym[symbolic]'  # CasADi symbolic dynamics
```

CasADi is optional. Numerical simulation, PID, MPC, data collection, and
evaluation do not require it.

## One environment entry point

Use the deterministic scenario default for a simple rollout:

```python
env = aiogym.make_env("three_tank", reward="regulation")
```

Use randomized episodes and policy-channel variation for RL or data collection:

```python
env = aiogym.make_env(
    "three_tank",
    reward="tank3-regulation",
    randomize=True,
    noise={"std": 0.01, "bias_std": 0.005},
    delay={"observation_steps": [0, 2], "action_steps": [0, 1]},
    fault={
        "probability": 0.05,
        "severity": [0.2, 0.5],
        "duration_steps": [20, 100],
    },
)
```

`randomize=True` samples tracking, disturbance-rejection, or boundary-safety
training episodes on every reset. These samples have the same kind of challenge
as the benchmarks but do not reuse their fixed protocol values. `noise`,
`delay`, and `fault` also accept `True` for documented defaults or `False` to
disable them. Faults are loss-of-effectiveness faults on one actuator.

Use a fixed Benchmark only for evaluation:

```python
env = aiogym.make_env("three_tank", benchmark="tracking")
```

A Benchmark always uses the Scenario's default model parameters and default
Reward. It cannot be combined with `parameters`, `reward`, `randomize`,
`noise`, `delay`, or `fault`, and `train()` rejects benchmark environments. Each
concrete Benchmark declares its own fixed measurement-noise configuration; the
caller cannot change it. This keeps the plant, score, and evaluation protocol
fixed.

## Parameter overrides and units

Parameter overrides create a model instance with different resolved values;
they do not edit source code. Unknown, non-finite, or physically invalid values
fail immediately.

```bash
aiogym list parameters --scenario three_tank
```

```python
env = aiogym.make_env(
    "three_tank",
    parameters={"heater_power": 1500.0},
)
print(env.model.resolved_parameters["heater_power"])
```

Units are model-native, not universally SI. Three-Tank primarily uses SI
(`m`, `m^2`, `m^3/s`, `W`, `W/K`) with temperature in `degC`. Quadruple-Tank
retains its source laboratory units (`cm`, `cm^2`, `cm^3/s`, `V`).

## Collect, train, and compare

```python
training_env = aiogym.make_env(
    "quadruple", randomize=True, noise=True, delay=True, fault=True
)
dataset = aiogym.collect(
    env=training_env,
    policy="pid",
    episodes=2,
    output="runs/quadruple-data",
)
training = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=64,
    record_every=500,
    output="runs/quadruple-sac",
)
aiogym.plot_training_curve(
    training["training_curve"],
    output="runs/quadruple-sac/replotted-training-curve.svg",
)
training_env.close()

benchmark_env = aiogym.make_env("quadruple", benchmark="tracking")
pid = aiogym.make_controller("pid", env=benchmark_env)
mpc = aiogym.make_controller("mpc", env=benchmark_env)
comparison = aiogym.compare_policies(
    env=benchmark_env,
    policies={"pid": pid, "mpc": mpc},
    seeds=[0, 1, 2],
)
benchmark_env.close()
```

All policies use the same controller configuration across the three benchmarks.
Comparison orders policies lexicographically by the selected benchmark's
declared metric medians. With no `output`, each benchmark comparison replaces
the two managed artifacts in `runs/<scenario>/<benchmark>/`: `comparison.json`
and a paper-style `comparison.svg` with output/reference trajectories, applied
actions, disturbances, safety margin, and cumulative-return distributions. An
explicit `output` must be an empty directory and is never overwritten.
Evaluation JSON also contains every seed trajectory plus median, MAD, minimum,
and maximum summaries. These are descriptive results, not statistical
significance.

A Dataset contains `metadata.json` plus one compressed NumPy file per episode.
Training writes `metadata.json`, `model.zip`, `training_curve.json`, and
`training_curve.svg`. The curve records fixed windows of mean step reward plus
complete-episode return, length, and termination outcome. Metadata records the
full resolved environment configuration; each collected or evaluated episode
records its actual episode specification and sampled channel variation.

Three-Tank residual control remains an optional scenario-owned wrapper:

```python
from aiogym.scenarios.three_tank import Tank3ResidualWrapper

env = Tank3ResidualWrapper(
    aiogym.make_env("three_tank", reward="tank3-regulation", randomize=True)
)
```

Hardware, calibration, and real-log tools are opt-in under
`aiogym.experimental.three_tank_hardware`.

## CLI and documentation

```bash
aiogym list scenarios
aiogym list benchmarks --scenario three_tank
aiogym list rewards --scenario three_tank
aiogym list parameters --scenario three_tank
aiogym collect three_tank --randomize --noise on --delay on --fault on \
  --output runs/data
aiogym evaluate three_tank --benchmark tracking --controller pid --seeds 0 1 2 \
  --output runs/tracking.json
aiogym compare three_tank --benchmark tracking --controllers pid mpc \
  --seeds 0 1 2
```

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Quadruple-Tank model](docs/scenarios/quadruple.md)
- [Three-Tank model](docs/scenarios/three_tank.md)
