# AIO-Gym

AIO-Gym is a compact process-control pipeline for simulation, classical
controllers, reinforcement learning, datasets, and policy comparison. Rather
than treating reinforcement learning as a replacement for every conventional
control structure, it provides a common environment for testing whether the
same continuous-control policy can sustain stable, long-horizon, multivariable
process control under model uncertainty, measurement noise, disturbances, and
safety constraints.

It includes the `crystallization`, `cstr`, `extraction`, `heater`, `hvac`,
`quadruple`, and `three_tank` scenarios. Each scenario provides training
episodes, Rewards, and three fixed evaluation Benchmarks: `tracking`,
`disturbance-rejection`, and `boundary-safety`.

## Install

```bash
pip install .
pip install 'aiogym[rl]'  # Stable-Baselines3 and Torch
```

Numerical simulation, PID, successive-linearization MPC, collection, and
evaluation use the base numerical dependencies. Reinforcement learning requires
the optional `rl` dependencies.

## Python API

Create one environment, controller, and evaluation through the public entry
points:

```python
import aiogym

env = aiogym.make_env("three_tank", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=range(20))
env.close()
```

Use `reset()` and `step()` directly for Gymnasium interaction:

```python
env = aiogym.make_env("quadruple")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

Use `randomize=True` for sampled training episodes and `benchmark=` only for a
fixed, reproducible evaluation protocol. Benchmark seeds identify resolved
evaluation cases; comparison gives every policy the same ordered cases.
Physical disturbances, observation noise, delay, and actuator faults are
independent optional training variations:

```python
training_env = aiogym.make_env(
    "quadruple",
    randomize=True,
    disturbance=True,
    noise=True,
    delay=True,
    fault=True,
)
benchmark_env = aiogym.make_env("quadruple", benchmark="tracking")
training_env.close()
benchmark_env.close()
```

An existing Dataset can supervise a supported actor before normal online
training. AIO-Gym stores the backend payload plus algorithm and environment
identity in the common `model.zip` format. Loading uses
`load_policy(checkpoint, env=...)`:

```python
env = aiogym.make_env("quadruple", randomize=True)
trained = aiogym.train(
    env=env,
    algorithm="sac",
    steps=10_000,
    dataset="runs/data",
    behavior_cloning_epochs=10,
    output="runs/quadruple/training/sac-bc/seed-0",
)
continued = aiogym.train(
    env=env,
    algorithm="sac",
    steps=10_000,
    resume_from=trained["checkpoint"],
    output="runs/quadruple/training/sac-bc/seed-0-continued",
)
env.close()
```

`steps` is the additional budget when `resume_from=` is set. Continuation
restores the model, optimizer, and replay state, uses the recorded seed and
algorithm configuration, and writes a new output directory.

Behavior cloning supports DDPG, SAC, and TD3. It learns
`observation -> commanded_action`; delayed, faulted, slew-limited
`applied_action` values are not imitation targets.

RLPD requires a Dataset and keeps its complete transitions active throughout
training. By default every update batch is 50% fixed Dataset transitions and
50% newly collected online replay; set `algorithm_kwargs.offline_ratio` between
`0` and `1` to change that split. The native implementation cites the
[paper](https://arxiv.org/abs/2302.02948) and
[reference code](https://github.com/ikostrikov/rlpd), while using AIO-Gym's
Dataset v2 and common `model.zip` format.

## CLI

Inspect the available resources:

```bash
aiogym list scenarios
aiogym list controllers
aiogym list algorithms
aiogym list benchmarks --scenario quadruple
aiogym list rewards --scenario quadruple
aiogym list parameters --scenario quadruple
```

Run the end-to-end workflows:

```bash
aiogym collect quadruple --randomize --disturbance on --noise on \
  --delay on --fault on \
  --controller pid --episodes 2 --output runs/data

aiogym train quadruple sac --randomize --disturbance on --noise on \
  --steps 10000 \
  --dataset runs/data --behavior-cloning-epochs 10 \
  --record-every 500 --evaluate-every 5000 \
  --output runs/quadruple/training/sac/seed-0

aiogym train quadruple rlpd --randomize --dataset runs/data --steps 100000 \
  --record-every 500 --output runs/quadruple/training/rlpd/seed-0

aiogym train quadruple sac --randomize --disturbance on --noise on \
  --steps 10000 \
  --resume-from runs/quadruple/training/sac/seed-0/model.zip \
  --output runs/quadruple/training/sac/seed-0-continued

aiogym evaluate quadruple --benchmark tracking --controller pid \
  --seeds {0..19} --output runs/pid-tracking.json

aiogym evaluate quadruple --benchmark tracking \
  --checkpoint runs/quadruple/training/sac/seed-0/best/model.zip \
  --seeds {0..19} --output runs/sac-tracking.json

aiogym compare quadruple --benchmark tracking --controllers pid mpc \
  --checkpoint sac-best runs/quadruple/training/sac/seed-0/best/model.zip \
  --seeds {0..19}
```

Commands print compact JSON summaries; complete artifacts remain under their
declared output paths. Short training runs verify the pipeline but are not
performance evidence.

## Documentation

- [Quickstart](docs/quickstart.md): environment interaction and complete
  collect/train/load/evaluate examples.
- [Architecture](docs/architecture.md): Scenario, Reward, Benchmark, controller,
  and workflow boundaries.
- [External algorithms](docs/external_algorithms.md): the backend contract,
  registration, package discovery, and optional behavior cloning.
- [RLPD](docs/rlpd.md): persistent Dataset plus online-replay training,
  reference sources, defaults, and implementation differences.
- [Reproducibility](docs/reproducibility.md): recorded configuration, artifacts,
  and evaluation summaries.
- [Two-input, two-output CSTR](docs/scenarios/cstr.md): reactor model,
  concentration/temperature targets, feed and cooling actions, disturbances,
  Benchmarks, and baseline controllers.
- [Batch crystallization](docs/scenarios/crystallization.md): moment model,
  reachable endpoint targets, disturbances, and batch controller baselines.
- [Multistage extraction](docs/scenarios/extraction.md): five-stage mass-transfer
  model, flow actions, disturbances, Benchmarks, and controller baselines.
- [Fired heater](docs/scenarios/heater.md): combustion and heat-transfer model,
  air/fuel actions, disturbances, Benchmarks, and controller baselines.
- [Two-zone HVAC](docs/scenarios/hvac.md): thermal model, disturbances,
  Benchmarks, and controller baselines.
- [Quadruple-Tank](docs/scenarios/quadruple.md): model, units, Benchmarks, and
  training commands.
- [Three-Tank](docs/scenarios/three_tank.md): hydraulic model, units, Benchmarks,
  direct four-action control, and experimental hardware boundary.
