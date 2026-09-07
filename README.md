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

## Start with heater

Run a small PID/MPC comparison from the repository root:

```bash
aiogym compare heater --benchmark tracking --controllers pid mpc --seeds 0 1 2 --output runs/heater/demos/pid-mpc
```

Open `runs/heater/demos/pid-mpc/comparison.svg`. The same directory contains
`comparison.json` for metrics and `trajectories.npz` for numeric trajectories.
Choose a new output directory when repeating this example.

With the RL dependencies installed, check the training workflow with:

```bash
aiogym train heater sac --steps 1000 --record-every 250 --no-evaluation
aiogym status
```

This is a workflow smoke run. Use the [Quickstart](docs/quickstart.md) to move
on to validation, best-checkpoint comparison, and a longer training budget.
Online SAC, DDPG, PPO, and TD3 training does not need a Dataset.

## Documentation

- [Quickstart](docs/quickstart.md): install, see a heater comparison, run a
  short training job, then start a validated experiment;
- [Task guide](docs/workflows.md): parallel training, status, continuation,
  Python examples, optional Dataset/BC/RLPD workflows, and output paths;
- [Concepts and reference](docs/architecture.md): environments, fixed Benchmarks,
  checkpoint selection, metrics, chart interpretation, and artifact contracts;
- [RLPD training](docs/rlpd.md): Dataset requirements and offline-to-online
  training;
- the scenario table above links directly to every process guide.

Short training runs verify that the workflow executes; they are not performance
evidence. For a meaningful comparison, use a fixed Benchmark, identical seeds,
and report safety separately from tracking quality.

Use `aiogym --help` to discover commands. The same workflows are available
through the Python API for integration with your own code.
