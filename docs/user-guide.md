# User guide

On this page: [Install](#1-install-aio-gym) · [Example](#2-understand-the-example) · [Train and compare](#3-train-and-compare) · [Results](#4-read-the-results) · [Common operations](#5-common-operations).

| Your task                                                | Start here                                    |
| -------------------------------------------------------- | --------------------------------------------- |
| Train with AIO-Gym's built-in algorithms                 | Follow this guide                             |
| Use your own training code or evaluate an external model | [External algorithms](external-algorithms.md) |
| Run terminal commands or manage training jobs            | [CLI guide](cli.md)                           |

## 1. Install AIO-Gym

You need **Python 3.10 or newer**. Download and extract the [repository](https://github.com/supcon-international/aiogym), or use Git:

```sh
git clone https://github.com/supcon-international/aiogym.git
cd aiogym
```

Run all commands from the **repository root**, the directory containing `pyproject.toml`. Create a virtual environment using your terminal's commands, or activate your existing one. Check the Python version before creating it.

**macOS / Linux terminal**

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

With the environment active, install RL support and check the import:

```sh
python -m pip install ".[rl]"
python -c "import aiogym; print(aiogym.__version__)"
```

The last command should print a version number. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` below. For projects using only simulation and classical controllers, `python -m pip install .` is sufficient.

## 2. Understand the example

Train SAC on the fired-heater simulation and compare it with PID and MPC. The policy adjusts air and fuel to bring oxygen and outlet temperature to their targets.

| Part        | In this example                                                             |
| ----------- | --------------------------------------------------------------------------- |
| Observation | Five normalized values: three process states and two target values          |
| Action      | Air damper and fuel valve commands, each in `[0, 1]`                        |
| Goal        | Reach the oxygen and temperature targets while staying within safety limits |
| One step    | One second of simulated process time                                        |

The [Heater guide](scenarios/heater.md) gives the physical variables, ranges, and success tolerances. Choose another process from the [scenario list](../README.md#included-scenarios).

## 3. Train and compare

Use a new `experiment` name for each fresh training run so its output directories do not contain results from an earlier run.

```python
from pathlib import Path

import aiogym

scenario = "heater"
algorithm = "sac"
experiment = "python-demo-1"
training_steps = 20_000
algorithm_settings = {}
record_interval = 500
validation_interval = 5_000

root = Path("runs") / scenario.replace("_", "-")
run = root / "training" / algorithm / experiment / "seed-0"
output = root / "demos" / experiment

print("Training; progress will be recorded under:", run.resolve(), flush=True)
with (
    aiogym.make_env(scenario, randomize=True) as training_env,
    aiogym.make_env(scenario, randomize=True) as validation_env,
):
    trained = aiogym.train(
        env=training_env,
        algorithm=algorithm,
        steps=training_steps,
        algorithm_kwargs=algorithm_settings,
        seed=0,
        record_every=record_interval,
        evaluation_env=validation_env,
        evaluate_every=validation_interval,
        output=run,
    )
print("Best checkpoint:", trained["best_checkpoint"])
print("Final checkpoint:", trained["checkpoint"])

print(f"Evaluating {algorithm.upper()}, PID, and MPC...", flush=True)
with aiogym.make_env(scenario, benchmark="tracking") as env:
    policy = aiogym.load_policy(trained["best_checkpoint"], env=env)
    comparison = aiogym.evaluate(
        env=env,
        policies={"pid": "pid", "mpc": "mpc", algorithm: policy},
        seeds=range(20),
        output=output,
    )
print("Policy ordering:", comparison["ordering"])
print("Comparison figure:", (output / "comparison.svg").resolve())
```

| Stage      | Purpose                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| Training   | `randomize=True` samples initial conditions and targets at each reset                 |
| Validation | The same 20 cases, seeds `1000–1019`, select the best saved model during training     |
| Comparison | `benchmark="tracking"` tests all policies on the same 20 held-out cases, seeds `0–19` |

Training and validation use separate environment instances: validation calls `reset()` and runs complete episodes, which would interrupt an ongoing training episode if they shared an instance.

A checkpoint is a saved model. This example evaluates the **best** checkpoint selected by validation; the **final** checkpoint also saves the training state needed for continuation.

## 4. Read the results

Evaluation saves these files in `output`:

| File               | Contents                                            |
| ------------------ | --------------------------------------------------- |
| `comparison.svg`   | Response curves and performance summary             |
| `comparison.json`  | Per-case metrics, aggregate statistics, and ranking |
| `trajectories.npz` | Numeric trajectories for every policy and test case |

Open `comparison.svg` in a browser: curves show the first test case; the summary covers all 20 cases.

| Question                     | Metric            | How to read it                                                         |
| ---------------------------- | ----------------- | ---------------------------------------------------------------------- |
| Did it finish safely?        | `safe_completion` | Fraction completing the planned episode without constraint violations  |
| Did it reach the target?     | `control_success` | Fraction completing safely and meeting the scenario's target criterion |
| How much reward did it earn? | `episode_return`  | Mean cumulative reward; higher is better                               |

Completion and success are percentages in the figure and fractions in `comparison.json`.

<a id="read-the-figure"></a>

| Cost boxes in multi-policy comparisons | Meaning                                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Ratio                                  | Policy cost / baseline cost per case; `0.5×` means half the cost; lower is better                     |
| Cost                                   | Tracking ISE for continuous control; terminal quality cost for crystallization                        |
| Baseline                               | First included PID, otherwise the first supplied policy                                               |
| Included cases                         | Safe, complete pairs with a valid ratio; usable counts are shown. Safety summaries include every case |

### Metrics and ranking

| Additional metric               | Meaning                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| `tracking_iae` / `tracking_ise` | Integrated absolute / squared normalized output error                |
| `final_error`                   | Largest absolute normalized output error at the endpoint             |
| `settling_time`                 | Time after which all outputs remain within their physical tolerances |
| `unsafe_rate`                   | Fraction of violating steps within a case                            |

Continuous-control success requires safe completion and all outputs within tolerance for the final 10% of the planned episode. Crystallization uses endpoint quality. See [scenario guides](../README.md#included-scenarios) for tolerances and error scales; normalized error is `(output - reference) / scale`, integrated in the model's physical time unit.

| Returned field                  | Meaning                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `comparison["ordering"]`        | Policy order from the benchmark's priorities, or the reward's primary metric without a benchmark |
| `comparison["ranking_metrics"]` | Ranking fields, directions, and aggregation; cost boxes do not determine ranking                 |

For a performance study, compare independently trained models on shared test cases held out from model selection. Record budgets, settings, Dataset sources, and software versions.

## 5. Common operations

Choose an operation as needed. Snippets reuse `aiogym` and `Path` imported in the main example; replace example paths for your run.

| Task                                                   | Section                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| Change algorithm, budget, or validation                | [Training settings](#training-settings)                       |
| Check progress or find checkpoints                     | [Training progress](#inspect-training-progress)               |
| Inspect variables, change parameters, or add variation | [Environment settings](#inspect-and-configure-an-environment) |
| Evaluate a saved model, with or without files          | [Evaluation](#evaluate-a-saved-model)                         |
| Resume a training run                                  | [Continuation](#continue-training)                            |
| Collect demonstrations for BC or RLPD                  | [Dataset collection](#collect-and-use-a-dataset)              |
| Adjust the MPC controller                              | [MPC settings](#configure-mpc)                                |

### Training settings

Change these values in the [main example](#3-train-and-compare):

| Example variable                          | Effect                                                                                              |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `algorithm`                               | `"sac"`, `"ddpg"`, `"ppo"`, or `"td3"` for online training; [RLPD](rlpd.md) also requires a Dataset |
| `training_steps`                          | Environment transitions to collect; PPO may exceed this count to finish a rollout batch             |
| `algorithm_settings`                      | Algorithm options, e.g. `{"learning_rate": 0.0003}`; `{}` uses defaults                             |
| `record_interval` / `validation_interval` | Steps between progress records / checkpoint evaluations                                             |
| `seed=0` in `train()`                     | Training randomness; use different seeds for independent models                                     |

Validation selects the best checkpoint by **safe completion → control success → mean return**, using each metric to break ties. It runs before training, at each interval, and at the end on the same 20 cases (seeds `1000–1019`).

The separate validation environment must use `randomize=True` and match the training scenario, parameters, reward, and observation/action interface. Noise, delay, faults, and disturbances may differ to test robustness. To disable validation, set `evaluate_every=None`, omit `evaluation_env` from `train()`, and evaluate `trained["checkpoint"]` instead of `trained["best_checkpoint"]`.

### Inspect training progress

| File in the training directory                  | Use                                                               |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| `model.zip` / `best/model.zip`                  | Final checkpoint for continuation / policy selected by validation |
| `training_curve.json` / `training_curve.svg`    | Recorded progress / validation learning curves                    |
| `evaluation_history.json` / `best/tracking.svg` | Validation records / selected policy's validation trajectories    |
| `metadata.json`                                 | Training settings and provenance                                  |

Progress records update at the configured intervals; figures are written at completion. While training runs, read recorded steps in a separate Python session once `training_curve.json` exists:

```python
import json

run = Path("runs/heater/training/sac/python-demo-1/seed-0")
curve = json.loads((run / "training_curve.json").read_text())
print("Recorded steps:", curve["actual_steps"])
```

For terminal-managed jobs, [CLI status](cli.md#check-progress-or-diagnose-a-failed-run) also shows logs, process state, and estimated time remaining.

### Inspect and configure an environment

```python
print(aiogym.list_scenarios())
print(aiogym.list_parameters("heater"))  # Available parameters, defaults, and units
with aiogym.make_env("heater", parameters={"outlet_temperature_trip": 425.0}) as env:
    print(env.parameters)     # Effective parameter values
    print(env.observations)   # Observation channels, units, and normalization
    print(env.actions)        # Action channels and bounds
    print(env.describe())     # Full environment description
```

For manual interaction or your own training code, use the [Gymnasium environment API](external-algorithms.md#1-use-the-simulation-environment).

Add these `make_env()` options to the main example's training and/or validation environment as needed:

| Option                                   | Effect                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `randomize=True`                         | Sample initial conditions and targets at reset                                        |
| `boundary_probability=0.3`               | Set the share of randomized starts near a reachable boundary                          |
| `disturbance=True`                       | Sample physical process changes                                                       |
| `noise=True`, `delay=True`, `fault=True` | Enable measurement noise, observation/action delay, or actuator loss of effectiveness |

`noise`, `delay`, and `fault` also accept mappings for explicit channel settings. See the [scenario guides](../README.md#included-scenarios) for physical variables and disturbance definitions.

### Evaluate a saved model

Evaluate the first example's best checkpoint without writing files:

```python
checkpoint = Path("runs/heater/training/sac/python-demo-1/seed-0/best/model.zip")
with aiogym.make_env("heater", benchmark="tracking") as env:
    policy = aiogym.load_policy(checkpoint, env=env)
    result = aiogym.evaluate(env=env, policies={"sac": policy}, seeds=range(20))
print(result["evaluations"]["sac"]["aggregate"])
```

| Change                          | How                                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| Compare several policies        | Add `"pid": "pid"`, `"mpc": "mpc"`, or another loaded policy to `policies`               |
| Save files as well              | Add `output="runs/heater/demos/saved-model-1"` to `evaluate()`                           |
| Evaluate your own trained model | Follow [external model evaluation](external-algorithms.md#2-evaluate-the-trained-policy) |

`result` contains metrics and all episode trajectories, whether or not `output` is supplied. If you set `max_steps` for a short diagnostic run, it does not establish full-task performance.

**Choose test conditions:** built-in benchmarks (`tracking`, `disturbance-rejection`, `boundary-safety`) fix parameters, reward, and test cases; they reject custom overrides. For a model trained with custom parameters, omit `benchmark` and create the test environment with the same parameters, reward, and policy interface. Use `randomize=True` for varied test cases; changing seeds alone does not vary a fixed environment. `load_policy()` checks compatibility. Noise, delay, faults, and disturbances may differ for robustness testing.

### Continue training

Set the paths below to your source and destination. This matches the first example and adds 50,000 transitions:

```python
scenario = "heater"
algorithm = "sac"
run = Path("runs/heater/training/sac/python-demo-1/seed-0")
continued_run = Path("runs/heater/training/sac/python-demo-1-continued/seed-0")
with (
    aiogym.make_env(scenario, randomize=True) as training_env,
    aiogym.make_env(scenario, randomize=True) as validation_env,
):
    continued = aiogym.train(
        env=training_env, algorithm=algorithm, steps=50_000,
        evaluation_env=validation_env, resume_from=run / "model.zip",
        output=continued_run,
    )
print("Best checkpoint:", continued["best_checkpoint"])
```

Keep the original training and validation environment settings, and retain the source run's histories and `best/` directory. Algorithm settings, seed, optimizer, and replay are restored from the checkpoint. Evaluate the newly printed checkpoint using [saved-model evaluation](#evaluate-a-saved-model).

### Collect and use a Dataset

Collect PID demonstrations, then read one episode:

```python
dataset_path = Path("runs/heater/datasets/pid-20/python-demo-1/seed-0")
with aiogym.make_env("heater", randomize=True) as env:
    aiogym.collect(env=env, policy="pid", episodes=20, seed=0, output=dataset_path)
episode = aiogym.DatasetReader(dataset_path).load_episode(0)
observations = episode.array("observation")
actions = episode.array("action")
```

Collection saves `metadata.json` and one `episode-XXXXXX.npz` per episode. To train from this data, match its scenario, parameters, reward, control interval, and ordered observation/action interface; collection variations may differ.

| Use the Dataset for                                | Change to the main training example                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| Behavior cloning before online training            | Add `dataset=dataset_path` and `behavior_cloning_epochs=10` to `train()` |
| Offline data mixed into replay throughout training | Follow [RLPD training](rlpd.md)                                          |

Define `dataset_path` in the training script and choose a new experiment name. For collection with your own model, see [external-policy collection](external-algorithms.md#3-collect-a-dataset-optional).

### Configure MPC

For default settings, pass `"mpc"` directly in `policies`. To change settings, use this inside the evaluation environment's `with ... as env:` block:

```python
mpc = aiogym.make_controller(
    "mpc", env=env, config={"prediction_horizon": 10, "solve_every": 2},
)
result = aiogym.evaluate(env=env, policies={"mpc": mpc}, seeds=range(20))
```

`make_controller()` supports `pid`, `mpc`, `hold`, and `random`. For PID/MPC, it combines scenario defaults with your `config`; `env` supplies the model, control interval, and observation/action definitions.

| Setting                 | Effect                                                                     |
| ----------------------- | -------------------------------------------------------------------------- |
| `prediction_horizon=10` | Predict ten control intervals                                              |
| `solve_every=2`         | Execute two planned actions before replanning; must not exceed the horizon |

MPC uses locally linearized dynamics and RK4 prediction with action bounds, without predicted-state safety constraints. Scenario guides list default controller settings.
