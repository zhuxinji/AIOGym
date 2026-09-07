# Quickstart

Start with `heater` to see a complete result before choosing a more demanding
process. Run commands from the repository root. Other processes use the same
commands; their variables and recommended budgets are in the
[scenario guides](../README.md#included-scenarios).

## 1. Install

```bash
pip install '.[rl]'
aiogym list scenarios
```

Use `pip install .` if you only need simulation and classical controllers.

## 2. Compare PID and MPC

```bash
aiogym compare heater --benchmark tracking --controllers pid mpc --seeds 0 1 2 --output runs/heater/demos/pid-mpc
```

Both controllers run the same three fixed test cases. Open
`runs/heater/demos/pid-mpc/comparison.svg` to see outputs, targets, actions, and
scores. `comparison.json` contains the metrics; `trajectories.npz` contains
the numeric trajectories. Three cases are an introduction, not a full study.

The explicit output must be a new or empty directory. To repeat the example,
choose another name such as `runs/heater/demos/pid-mpc-2`. Omitting `--output`
on a Benchmark comparison replaces the managed results in
`runs/heater/benchmarks/tracking/`.

## 3. Check that training runs

```bash
aiogym train heater sac --steps 1000 --record-every 250 --no-evaluation
```

This short run checks environment interaction and checkpoint saving. It skips
validation to finish sooner; it does not select a best policy or establish
control performance. SAC, DDPG, PPO, and TD3 can train directly online, without
collecting a Dataset first.

The command prints its output directory before starting. In another terminal,
from the same working directory, inspect recent CLI runs with:

```bash
aiogym status
```

The table shows state, phase, recorded steps, elapsed time, and the last update.
A rough ETA appears after enough progress has been recorded. Completed jobs
remain in the list. Use `aiogym status --json` for machine-readable output.

## 4. Train with validation

```bash
aiogym train heater sac --steps 500000
```

Training randomizes the operating conditions by default. Statistics are saved
every 500 steps and validation runs every 5,000 steps, including an initial
validation before training. This takes longer than the smoke run. The step
budget counts environment transitions, not seconds of wall time.

Each invocation creates a new directory:

```text
runs/heater/training/sac/<experiment>/seed-0/
```

When training finishes, the terminal prints actual paths and commands for:

- **Comparison:** `best/model.zip`, selected on separate validation cases.
- **Continuation:** `model.zip`, saved at the final training step.
- **Learning curve:** `training_curve.svg`; raw records remain in JSON.

Copy the printed comparison command to evaluate PID and the learned policy
on the held-out tracking Benchmark. Keep the same evaluation seeds when adding
other algorithms. The comparison command uses a separate output directory.

For independent algorithms or seeds, use the same `train` command:

```bash
aiogym train heater sac ppo td3 --seeds 0 1 --workers 3
```

## Continue by task

See the [task guide](workflows.md) for continuation, output rules, Python usage,
and the optional Dataset/BC/RLPD branch. Consult
[concepts and reference](architecture.md) for checkpoint selection and metrics,
or `aiogym train --help` for grouped options and examples.
