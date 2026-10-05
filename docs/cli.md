# CLI guide

Run AIO-Gym workflows from a terminal, with job scheduling and status for training runs. Complete the [installation](user-guide.md#1-install-aio-gym), then run commands from the repository root with the Python environment active. Commands below are task alternatives.

```bash
aiogym --help
aiogym train --help
aiogym list --help
```

The [User guide](user-guide.md) walks through the same operations in Python.

| Command    | Purpose                                          | Python entry                              |
| ---------- | ------------------------------------------------ | ----------------------------------------- |
| `list`     | Discover resources and inspect definitions       | `aiogym.list_*()`, environment properties |
| `collect`  | Collect a Dataset                                | `aiogym.collect()`                        |
| `train`    | Train one or several independent policies        | One `aiogym.train()` call per task        |
| `status`   | Inspect CLI-managed training jobs                | CLI job records                           |
| `evaluate` | Evaluate one or more policies on identical cases | `aiogym.evaluate()`                       |

## List resources and environment information

```bash
aiogym list scenarios
aiogym list algorithms
aiogym list controllers
aiogym list parameters --scenario heater
aiogym list states --scenario heater
aiogym list actions --scenario heater
aiogym list observations --scenario heater
aiogym list outputs --scenario heater
aiogym list rewards --scenario heater
aiogym list benchmarks --scenario heater
aiogym list safety_rules --scenario heater
aiogym list info --scenario heater --benchmark tracking --json
```

All `list` commands support `--json`. `info` shows a configuration template; `--benchmark` selects the fixed Benchmark. See [information fields](user-guide.md#inspect-and-configure-an-environment) for variable ranges, physical units, normalization, and safety rules.

## Run your first comparison

Start with built-in controllers:

```bash
aiogym evaluate heater --benchmark tracking --controllers pid mpc --seeds 0 1 2 --output runs/heater/demos/cli-pid-mpc
```

Open `comparison.svg` in the chosen directory. See the [result guide](user-guide.md#4-read-the-results) for what to expect. Use a new output directory when repeating this command.

## Train one policy

Train with validation and best-checkpoint selection. The 20,000 steps below demonstrate the workflow, not control performance:

```bash
aiogym train heater sac --steps 20000
```

| Training flag      | Default / effect                                                |
| ------------------ | --------------------------------------------------------------- |
| `--steps`          | `500000` additional environment transitions per task            |
| `--record-every`   | `500` steps                                                     |
| `--evaluate-every` | `5000` steps; [selection rule](user-guide.md#training-settings) |
| `--seeds`          | `0` for new training                                            |
| `--no-randomize`   | Fixed default episode; otherwise randomized training            |
| `--no-evaluation`  | Disable separate randomized validation and best selection       |

### Algorithm and environment settings

Save parameter overrides as a JSON object, for example `heater-parameters.json`:

```json
{"outlet_temperature_trip": 425.0}
```

```bash
aiogym train heater sac --parameters heater-parameters.json
aiogym train heater sac --boundary-probability 0.3 --disturbance on --noise on --delay on --fault on
```

`--algorithm-kwargs FILE.json` similarly supplies algorithm settings and supports one algorithm, optionally with multiple seeds. The CLI's on/off variation flags use the built-in defaults; use Python mappings for explicit channel settings. See [training variation](user-guide.md#inspect-and-configure-an-environment) for their meanings. These CLI flags configure training. CLI validation uses a separate randomized environment with variation disabled; use the Python `evaluation_env` argument to select checkpoints under custom validation conditions.

### Run directories

Automatic training output uses:

```text
runs/<scenario>/training/<algorithm>/<experiment>/seed-<n>/
```

| Run detail      | Behavior                                                                         |
| --------------- | -------------------------------------------------------------------------------- |
| Experiment name | Unique per invocation, shared across its algorithms/seeds                        |
| Scenario path   | Hyphenated, e.g. `three_tank` → `three-tank`                                     |
| `--output DIR`  | Exact new/empty directory for a single task; batches use automatic paths         |
| Extra CLI files | `train.log`, `status.json`, supplied `parameters.json` / `algorithm-kwargs.json` |
| Completion      | Prints paths plus comparison/continuation commands                               |
| Streams         | Structured results on stdout; notices on stderr                                  |

[Common training artifacts](user-guide.md#inspect-training-progress).

## Check progress or diagnose a failed run

```bash
aiogym status
aiogym status --limit 20
aiogym status /path/to/training/seed-0
aiogym status --json
```

Without paths, `status` finds the ten most recently updated CLI status records under `runs/*/training/`. Explicit paths also support runs outside `runs/`. Direct Python training does not create a CLI status record.

| Status field    | Interpretation                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------ |
| Steps           | Recorded in `training_curve.json`; may lag until the next write; includes resumed steps          |
| Phase           | Setup, behavior cloning, training, validation, saving; initial validation can stay at zero steps |
| Elapsed         | Includes setup and validation                                                                    |
| ETA             | Rough current-segment average; shown during training after 2 recording windows and 30 s          |
| `UPDATED`       | Age of last status/curve write                                                                   |
| `unknown` / `—` | Process identity could not be verified / progress unavailable                                    |

| Event                        | Scheduler behavior                                                            |
| ---------------------------- | ----------------------------------------------------------------------------- |
| Failed task                  | Stop new dispatch, let active tasks finish, exit nonzero; inspect `train.log` |
| Ctrl-C / SIGTERM             | Stop active workers                                                           |
| Scheduler exits unexpectedly | Active workers record final status; queued tasks need a scheduler             |

## Evaluate saved policies

Use the best checkpoint path printed at training completion:

```bash
aiogym evaluate heater --benchmark tracking --controllers pid mpc --checkpoint sac /path/to/run/best/model.zip --seeds 0 1 2 --output runs/heater/demos/cli-sac-comparison
aiogym evaluate heater --benchmark tracking --checkpoint sac /path/to/run/best/model.zip --seeds 0 1 2
```

Use `--controllers pid` for one controller, or list several controllers and repeat `--checkpoint LABEL MODEL_ZIP` for multiple learned policies. Omit `--output` to print the complete evaluation result without writing files. See [fair evaluation](user-guide.md#metrics-and-ranking) for case selection and [comparison charts](user-guide.md#4-read-the-results) for the results.

## Continue training

```bash
aiogym train heater sac --resume-from /path/to/run/model.zip --steps 50000
```

This adds 50,000 steps in a new run directory. Continuation supports one task. Reuse the original environment and validation settings; the seed and algorithm configuration are restored from the checkpoint. The continuation command printed at completion retains CLI settings and frozen parameter paths. RLPD also needs `--dataset`. See [continuation requirements](user-guide.md#continue-training).

## Train multiple algorithms or seeds

After a single run works, schedule independent models:

```bash
aiogym train heater sac ppo td3 --steps 500000 --workers 3
aiogym train heater sac ppo --steps 500000 --seeds 0 1 2 --workers 2
```

Each algorithm/seed pair gets its own model and step budget. The second command creates six training tasks. These are larger experiments; choose the algorithms, budgets, and concurrency for your available CPU and memory.

The default is at most two concurrent processes; `--workers 1` runs sequentially. Batch training uses the automatic [run directories](#run-directories); `--output` and `--resume-from` support a single task.

## Dataset, behavior cloning, and RLPD

```bash
aiogym collect heater --controller pid --randomize --episodes 20 --seed 0 --output runs/heater/datasets/cli-pid-20/seed-0
aiogym train heater sac --dataset runs/heater/datasets/cli-pid-20/seed-0 --behavior-cloning-epochs 10
aiogym train heater rlpd --dataset runs/heater/datasets/cli-pid-20/seed-0
aiogym train heater sac ddpg ppo td3 rlpd --dataset runs/heater/datasets/cli-pid-20/seed-0 --workers 3
```

In the last command only RLPD consumes the Dataset; the others train online. With `--behavior-cloning-epochs`, algorithms that support cloning also consume it. Compatibility is checked before dispatch. See the [Dataset workflow](user-guide.md#collect-and-use-a-dataset) and [RLPD tutorial](rlpd.md) for when to use these options.

## Output arguments

| Command    | Output argument                                                           |
| ---------- | ------------------------------------------------------------------------- |
| `collect`  | Required `--output DIR`                                                   |
| `train`    | Optional `--output DIR` for a single task; otherwise automatic            |
| `evaluate` | Optional `--output DIR`; omit to print full results without writing files |

Choose new or empty output directories. Evaluation exports the [same report files as Python](user-guide.md#4-read-the-results).

Use the [user guide](user-guide.md#5-common-operations) for Python operations and numeric data access, or the [scenario index](../README.md#included-scenarios) for process-specific settings.
