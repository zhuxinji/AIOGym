# AIO-Gym

AIO-Gym is a compact process-control pipeline for simulation, classical
controllers, reinforcement learning, datasets, and policy comparison. Rather
than treating reinforcement learning as a replacement for every conventional
control structure, it provides a common environment for testing whether the
same continuous-control policy can sustain stable, long-horizon, multivariable
process control under model uncertainty, measurement noise, disturbances, and
safety constraints.

It includes the `quadruple` and `three_tank` scenarios. Each scenario provides
training episodes, Rewards, and three fixed evaluation Benchmarks: `tracking`,
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

env = aiogym.make_env("three_tank")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=[0, 1, 2])
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
    demonstrations="runs/data",
    behavior_cloning_epochs=10,
    output="runs/sac-bc",
)
env.close()
```

Behavior cloning supports DDPG, SAC, and TD3. It learns
`observation -> commanded_action`; delayed, faulted, slew-limited
`applied_action` values are not imitation targets.

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
  --demonstrations runs/data --behavior-cloning-epochs 10 \
  --record-every 500 --evaluate-every 5000 --output runs/sac

aiogym evaluate quadruple --benchmark tracking --controller pid \
  --seeds 0 1 2 --output runs/pid-tracking.json

aiogym evaluate quadruple --benchmark tracking \
  --checkpoint runs/sac/best/model.zip \
  --seeds 0 1 2 --output runs/sac-tracking.json

aiogym compare quadruple --benchmark tracking --controllers pid mpc \
  --checkpoint sac-best runs/sac/best/model.zip \
  --seeds 0 1 2
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
- [Reproducibility](docs/reproducibility.md): recorded configuration, artifacts,
  and evaluation summaries.
- [Quadruple-Tank](docs/scenarios/quadruple.md): model, units, Benchmarks, and
  training commands.
- [Three-Tank](docs/scenarios/three_tank.md): model, units, Benchmarks, direct
  five-action control, and experimental hardware boundary.
