# AIO-Gym

AIO-Gym is a process-control toolkit for simulation, classical control,
reinforcement learning, and reproducible comparison. It lets PID, MPC, and
learned policies run on the same environments and test cases, so users can
compare stability, tracking quality, disturbance rejection, and safety.

## Highlights

- Gymnasium-compatible `reset()` and `step()` environments;
- built-in PID, MPC, hold, and random controllers;
- DDPG, PPO, SAC, TD3, and RLPD training;
- randomized operating conditions, physical disturbances, measurement noise,
  delays, and actuator faults;
- Dataset collection, behavior cloning, and offline-to-online learning;
- trainable checkpoints that can be loaded, evaluated, or continued;
- tracking, disturbance-rejection, and boundary-safety Benchmarks;
- JSON summaries, SVG comparisons, and complete numeric trajectories.

## Included scenarios

| Scenario id | Process |
|---|---|
| [`cascade`](docs/scenarios/cascade.md) | Heated three-tank cascade with configurable heaters and a dynamic closed reservoir |
| [`crystallization`](docs/scenarios/crystallization.md) | Batch crystallization |
| [`cstr`](docs/scenarios/cstr.md) | Two-input, two-output stirred-tank reactor |
| [`extraction`](docs/scenarios/extraction.md) | Five-stage counter-current extraction |
| [`heater`](docs/scenarios/heater.md) | Fired heater |
| [`hvac`](docs/scenarios/hvac.md) | Two-zone HVAC |
| [`quadruple`](docs/scenarios/quadruple.md) | Quadruple-tank laboratory process |
| [`three_tank`](docs/scenarios/three_tank.md) | Hydraulic three-tank cascade |

Each scenario guide documents its physical variables, actions, safety limits,
training variation, controller settings, and fixed Benchmarks.

## Install

From a downloaded or cloned AIO-Gym repository, run:

```bash
pip install .
```

To include reinforcement-learning training, use this instead:

```bash
pip install '.[rl]'
```

## Five-minute start

Choose a scenario, reset it, and take one action:

```python
import aiogym

scenarios = aiogym.list_scenarios()
scenario = scenarios[0]  # Replace this with any returned scenario id.

env = aiogym.make_env(scenario)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

Continue with the [Quickstart](docs/quickstart.md) to compare PID and MPC,
collect a Dataset, train a learned policy, continue from a checkpoint, and read
the generated results.

## Documentation

- [Quickstart](docs/quickstart.md): the complete Python workflow from
  environment creation through collection, training, continuation, and
  comparison;
- [Features and workflow](docs/architecture.md): what ordinary environments,
  randomized training, fixed Benchmarks, checkpoints, and result files mean;
- [RLPD training](docs/rlpd.md): Dataset requirements and offline-to-online
  training;
- the scenario table above links directly to every process guide.

Short training runs verify that the workflow executes; they are not performance
evidence. For a meaningful comparison, use a fixed Benchmark, identical seeds,
and report safety separately from tracking quality.

The same workflows are also available through the optional `aiogym` command.
Use `aiogym --help` when shell automation is more convenient.
