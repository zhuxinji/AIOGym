# Task guide

For a first run, follow the [Quickstart](quickstart.md). This page covers common
tasks once you have seen a heater comparison. Run CLI commands from the same
working directory so automatic output discovery uses the same `runs/` tree.

## Train one or several policies

```bash
aiogym train heater sac
aiogym train heater sac ppo td3 --workers 3
aiogym train heater sac ppo --seeds 0 1 2 --workers 2
```

Defaults: 500,000 additional environment steps per task, training records every
500 steps, validation every 5,000 steps, randomized training, and at most two
concurrent processes. Each seed starts an independent model. The CLI creates
a separate randomized validation environment. `--no-randomize` selects the
fixed default training episode; `--no-evaluation` disables validation.

Multiple tasks run in separate processes. `--workers 1` runs them sequentially;
each batch worker limits OpenMP, MKL, and OpenBLAS to one computation thread.
Choose concurrency for available CPU and memory. `--algorithm-kwargs FILE.json`
accepts one algorithm, optionally with multiple seeds.

Outputs remain under each algorithm:

```text
runs/<scenario>/training/<algorithm>/<experiment>/seed-<n>/
├── model.zip                 # final state, for continuation
├── best/model.zip            # validation-selected policy, when enabled
├── best/tracking.svg         # best policy on validation cases
├── training_curve.json
├── training_curve.svg
├── evaluation_history.json   # when validation is enabled
├── metadata.json
├── train.log                 # CLI only
├── status.json               # CLI only
└── parameters.json / algorithm-kwargs.json  # supplied JSON inputs, if any
```

An invocation shares one unique experiment name across algorithms and seeds.
Scenario ids use hyphens in paths, e.g. `three_tank` becomes `three-tank`.
`--output DIR` specifies the exact directory for a single task. Batch training
requires automatic directories. Completion hints use real paths; structured
summaries stay on stdout and human-readable notices go to stderr.

## Check progress or diagnose a failed run

```bash
aiogym status
aiogym status --limit 20
aiogym status runs/heater/training/sac/<experiment>/seed-0
aiogym status --json
```

Without paths, `status` finds the ten most recently updated CLI status records
under `runs/*/training/`. Explicit paths inspect specific runs, including those
written with `--output` outside `runs/`. Python-only training does not create a
CLI status record.

Steps come from `training_curve.json` and can lag until the next statistics
window or validation interval. For continuation, the displayed cumulative
target includes the starting checkpoint's steps. Phase distinguishes setup,
behavior cloning, training, validation, and saving. Initial validation can take
time while steps are still zero. Elapsed time includes setup and validation.
ETA is a rough average for the current training segment, shown only during
training after at least two recording windows and 30 seconds; validation costs
and changing training speed can make it inaccurate. `UPDATED` is the age of
the last status or curve write, not a heartbeat.

Read the printed `train.log` path for errors. A failed task stops new dispatch;
active tasks finish and the batch command exits nonzero. Ctrl-C or SIGTERM
stops active workers. A worker records its final status even if its scheduler
exits unexpectedly; queued tasks need the scheduler to start them.

On POSIX, `status` checks the original process identity, so dead jobs are
reported as `interrupted` even if their PID was reused. A running job without
an available process identity shows `unknown`. Unavailable progress fields
display `—`; inspection does not rewrite saved records.

## Compare or evaluate a checkpoint

Replace the checkpoint path below with the **best checkpoint** printed at
completion. These are fixed evaluation seeds, independent of training seeds.

```bash
aiogym compare heater --benchmark tracking --controllers pid mpc --checkpoint sac /path/to/run/best/model.zip --seeds 0 1 2 --output runs/heater/demos/sac-comparison
aiogym evaluate heater --benchmark tracking --checkpoint /path/to/run/best/model.zip --seeds 0 1 2 --output runs/heater/demos/sac-evaluation.json
```

For formal reports, use the same larger seed set for all policies and report
variation across independently trained models. Read safety separately from
tracking quality. See [comparison charts](architecture.md#read-a-comparison-figure)
and [scenario guides](../README.md#included-scenarios) for metric definitions.
See [controller capabilities](architecture.md#run-classical-or-learned-controllers)
for the MPC baseline's assumptions and limits.

## Continue training

```bash
aiogym train heater sac --resume-from /path/to/run/model.zip --steps 50000
```

This adds 50,000 steps in a new run directory, retaining compatible history.
The source checkpoint stays unchanged. Use the same scenario, reward,
parameters, variation settings, validation setting, and training seed.
The command printed when training finishes retains those CLI settings and
points to frozen parameter inputs when supplied. Algorithm settings and seed
are restored from the checkpoint. Keep `best/` and saved histories when moving
a run; see [checkpoint compatibility](architecture.md#dataset-and-checkpoint-compatibility).

RLPD continuation also needs `--dataset`, following the same compatibility
rules. Behavior cloning is an initialization step and is not repeated on continuation.

## Optional: Dataset, behavior cloning, and RLPD

Ordinary online SAC, DDPG, PPO, and TD3 do not require this step. Collect a
Dataset when you want demonstrations for behavior cloning or offline data
for RLPD:

```bash
aiogym collect heater --controller pid --randomize --episodes 20 --seed 0 --output runs/heater/datasets/pid-20/seed-0
aiogym train heater sac --dataset runs/heater/datasets/pid-20/seed-0 --behavior-cloning-epochs 10
aiogym train heater rlpd --dataset runs/heater/datasets/pid-20/seed-0
aiogym train heater sac ddpg ppo td3 rlpd --dataset runs/heater/datasets/pid-20/seed-0 --workers 3
```

In the last command only RLPD consumes the Dataset; the other algorithms train
online. With `--behavior-cloning-epochs`, algorithms that support cloning also
consume it. Compatibility is checked before dispatch. See [RLPD](rlpd.md) for
data requirements and [Dataset contracts](architecture.md#dataset-and-checkpoint-compatibility)
for observation/action and training requirements.

## Add training variation

```bash
aiogym train heater sac --boundary-probability 0.3 --disturbance on --noise on --delay on --fault on
```

Randomization samples feasible initial conditions and targets; boundary
probability controls the share of reachable boundary initializations.
Disturbance changes the physical process, noise changes measured observations,
delay affects observation/action timing, and fault applies temporary actuator
loss of effectiveness. Use `aiogym.make_env()` with mappings for explicit
variation settings. Fixed Benchmarks own their cases and reject these runtime
overrides. Names, timings, units, and parameter overrides belong to each
[scenario guide](../README.md#included-scenarios).

For explicit disturbance timing, supply a scenario-supported piecewise-constant
schedule to `make_env()` instead of enabling automatic disturbance sampling.
On ordinary environments, `set_disturbances()` applies values immediately and
updates their reset overrides; scheduled events retain precedence. Saved
trajectories record the actual disturbances at each transition.

## Choose output paths

| Command | `--output` expects | Existing output |
|---|---|---|
| `collect` | Dataset directory, required | Must be new or empty |
| `train` | Exact directory for one task, or automatic per-run path | Existing training artifacts rejected |
| `evaluate` | JSON file, required | File must not exist |
| `compare --output DIR` | Comparison directory | Must be new or empty |
| `compare --benchmark NAME` without output | `runs/<scenario>/benchmarks/<name>/` | Replaces the three managed comparison artifacts |

The CLI prints the destination and write mode before running the workflow.
Comparison creates `comparison.json`, `comparison.svg`, and `trajectories.npz`
together. Choose an explicit new directory for experiments; omit it only when
you intend to replace the official Benchmark comparison. Python workflows use
the same output rules.

If a comparison is interrupted and reports `.comparison-pending/`, first confirm
no comparison is still running. Inspect its `previous/` backups and the three
result files before retrying or removing recovery data. A completed rollback
restores the previous files; incomplete recovery blocks another write. This
recovery mechanism does not guarantee a consistent three-file snapshot for
concurrent readers or after power loss.

## Use the Python API

The API supports a single model per `train()` call. Supply a separate randomized
validation environment or disable validation with `evaluate_every=None`:

```python
from pathlib import Path
import aiogym

run = Path("runs/heater/training/sac/python-500k/seed-0")
training_env = aiogym.make_env("heater", randomize=True)
evaluation_env = aiogym.make_env("heater", randomize=True)
try:
    trained = aiogym.train(
        env=training_env, algorithm="sac", steps=500_000, seed=0,
        evaluation_env=evaluation_env, output=run,
    )
finally:
    training_env.close()
    evaluation_env.close()

print("Compare:", trained["best_checkpoint"])
print("Continue:", trained["checkpoint"])

env = aiogym.make_env("heater", benchmark="tracking")
try:
    policy = aiogym.load_policy(trained["best_checkpoint"], env=env)
    comparison = aiogym.compare_policies(
        env=env, policies={"pid": "pid", "sac": policy}, seeds=range(20),
        output=run / "comparison",
    )
finally:
    env.close()
print(comparison["ordering"])
```

For continuation, supply `resume_from=run / "model.zip"`, an additional `steps`
budget, and a new output directory to `train()`. For Dataset collection use
`aiogym.collect(env=env, policy="pid", episodes=20, seed=0, output=dataset_path)`
with a randomized environment; `aiogym.DatasetReader(dataset_path)` exposes
episodes and `transition_count`. The caller owns and closes its environments.

## Replot an existing learning curve

```python
import json
from pathlib import Path
import aiogym

run = Path("/path/to/run")
metadata = json.loads((run / "metadata.json").read_text())
aiogym.plot_training_curve(
    run / "training_curve.json",
    evaluation_history=run / "evaluation_history.json",
    control_dt=metadata["environment"]["control_dt"],
    settling_fraction=0.1,
    output=run / "validation-preview.svg",
)
```

This needs saved validation history. Choose a new output filename. Changing
the plotted settling fraction changes the displayed success rate, not saved
checkpoint selection. See [training charts](architecture.md#read-a-training-curve)
for P10 and the return axis, and [evaluation metrics](architecture.md#understand-evaluation-metrics)
for the settling-window definition.
