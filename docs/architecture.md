# Core concepts and results

This page explains the concepts needed to choose a workflow and interpret its
results. Start with the [Quickstart](quickstart.md); use the
[task guide](workflows.md) for commands, output paths, and troubleshooting.

```text
Online RL:  Environment ----------------------> train -> checkpoint
BC / RLPD:  Environment + collected Dataset ---> train -> checkpoint
Comparison: Evaluation environment + policies -> evaluate / compare
```

Online SAC, DDPG, PPO, and TD3 need no Dataset. Collect data when using behavior
cloning or RLPD. Built-in controllers and loaded checkpoints share the same
evaluation interface.

## Choose a process-control scenario

A Scenario defines the physical model, observations, actions, targets, rewards,
safety limits, and fixed Benchmarks. Its [scenario guide](../README.md#included-scenarios)
provides physical units, parameter overrides, and controller settings.

| Environment mode | Purpose | Episode settings |
|---|---|---|
| Ordinary | Simulation, collection, or training | Default, explicit, or randomized episodes |
| Benchmark | Reproducible evaluation | Fixed plant settings and protocol; the seed selects a case |

For an ordinary environment, `randomize=True` samples a new feasible episode on
each reset. With `randomize=False`, `initial_state=` sets the starting state where
supported, without inferring a matching target or steady action.
Benchmarks own their complete cases and reject manual initial states and runtime
variation. See [training variation](workflows.md#add-training-variation) for
randomization, disturbances, noise, delay, and actuator faults.

## Run classical or learned controllers

PID, MPC, hold, random, and loaded policies receive the observation plus the
current step, physical time, and public reference. Noise and delay apply
consistently to every policy. Derived observation features use the policy-visible
measurements and references, without independent sensor noise.

`info` contains diagnostics such as true state, applied action, safety margins,
and the complete episode schedule. These are saved for analysis and are not
additional policy inputs. MPC uses its nominal model without access to hidden
runtime disturbances.

The MPC baseline linearizes that model and predicts `P` steps, but optimizes
only one action adjustment (`M=1`), held throughout the prediction. It solves an
unconstrained quadratic objective and then clips actions to `[0, 1]`; clipping
may be suboptimal for coupled inputs. It imposes no predicted state safety
constraints. Increasing `P` extends prediction, not the number of optimized moves.
Environment safety checks still apply during simulation.

## Dataset and checkpoint compatibility

Datasets retain policy commands, actions after delay/faults, and the physical
actions applied to the model. Before BC or RLPD training, their scenario, reward,
parameters, control interval, and observation/action shapes must match the
training environment. Observation and action arrays must be finite and within
its spaces. Source randomization and variation may differ; their settings remain
recorded as data provenance.

RLPD continuation requires the same Dataset content used for the original run;
the directory may change.

A trainable checkpoint retains the optimizer and algorithm-owned replay state.
Continuation requires compatible scenario, reward, parameters, variation,
control interval, and interface shapes, and reuses the recorded seed and
algorithm settings. Histories are retained through the resumed checkpoint;
resuming an earlier best checkpoint excludes later training. The best policy is
selected across retained and new validation records. Validation stays enabled
or disabled consistently across segments.

For commands and files to retain, see [continuation](workflows.md#continue-training).

## Understand evaluation metrics

| Metric | Meaning | Preferred direction |
|---|---|---|
| Unsafe rate | Fraction of evaluated steps with a constraint violation, averaged across cases | Lower |
| Safe completion | Full planned horizon, no safety termination, and zero constraint violations | Higher |
| Control success | Safe completion plus continuous tracking within the settling window below | Higher |
| Settling rate | Fraction of cases that enter and remain in the settling band before the evaluated episode ends | Higher |
| Episode return | Sum of rewards; comparable under the same Reward | Higher |
| Tracking IAE / ISE | Time-integrated absolute / squared normalized tracking error | Lower |
| Final error | Largest absolute normalized tracking error at the episode endpoint | Lower |

The settling window is the **final 10% of planned control steps**, rounded up:
`ceil(episode_spec.horizon * 0.1)`. Every controlled output must stay inside its
scenario's settling tolerance at every sample in that window. Early termination
does not shorten the planned horizon. The same definition applies to training
validation and Benchmark reports; it does not change rewards or episode endings.

Rates include all evaluated cases. JSON stores fractions from 0 to 1; figures
show percentages. Scenario guides define physical tolerances independently of
observation normalization and reward weights. Unsupported control-success
calculations display `N/A`.

Time axes use each model's physical unit. Tracking integrals respect that unit;
energy metrics integrate kW after converting elapsed time to hours. Built-in
disturbance Benchmarks and default training samplers restore disturbances before
the settling window. Explicit user schedules retain their requested timings.

## Select the best checkpoint

Training validation uses a separate `randomize=True` environment and fixed
seeds `1000–1019`. Disturbance, noise, delay, and actuator faults are disabled.
This selection set is separate from the formal Benchmark cases.

Candidates are compared in this order, using the [metrics above](#understand-evaluation-metrics):

1. Higher safe completion rate.
2. Higher control success rate.
3. Higher mean episode return across all validation cases.

A later metric only breaks an exact tie; a complete tie retains the existing
best candidate. P10 return, median return, and episode lengths are diagnostics.
The selection order and selected step are recorded in validation history.

| Checkpoint | Role |
|---|---|
| `best/model.zip` | Validation-selected policy for formal comparison |
| `model.zip` | Final learner for continuation and late-training diagnosis |

Disabling validation leaves no selected best checkpoint. Formal Benchmark
ranking follows its own metric priorities, described next.

## Evaluate fairly

Benchmarks cover target tracking (`tracking`), recovery after process changes
(`disturbance-rejection`), and safety near operating limits (`boundary-safety`).
Each seed identifies a complete physical case; boundary cases start from
reachable conditions. The standard formal comparison uses seeds `0–19`.

Use `evaluate()` for one policy and `compare_policies()` for two or more.
Compare policies on the same Scenario, Benchmark, Reward, parameters, and ordered
seeds. These evaluation seeds are separate from independent training seeds.

Formal ranking prioritizes safety before performance. `unsafe_rate`,
`safe_completion`, `settling_rate`, and `return` use the mean across equally
weighted cases; other metrics use the median. The exact priority, direction,
and aggregation are stored
in `comparison.json["ranking_metrics"]`. Tracking-cost plots do not change this
ranking or include other reward costs.

For reproducibility, retain the comparison JSON and numeric trajectories along
with training settings, Dataset information, library versions, and source revision.
Report safety separately from tracking quality and compare multiple training
seeds when assessing RL performance. A recorded Git commit does not capture
uncommitted source changes.

## Read a training curve

`training_curve.svg` shows fixed-validation mean return and P10, followed by safe
completion and control success, with the selected checkpoint marked. P10 is the
lower return percentile across cases, not uncertainty across independently
trained models. The return axis is symmetric-logarithmic, linear within
`[-1, 1]`, with labels showing raw return values.

Training rewards and episode lengths remain in `training_curve.json`; individual
validation cases remain in `evaluation_history.json`. With validation disabled,
the figure indicates that validation data is unavailable. Replotting with a
different settling fraction changes the displayed success rate without
reselecting checkpoints; [replotting instructions](workflows.md#replot-an-existing-learning-curve)
use the run's recorded control interval.

## Read a comparison figure

The figure shows output/reference and action traces for the first shared seed,
a summary across all cases, and paired tracking-ISE boxplots. Each ratio divides
the policy's ISE by the baseline's on the **same seed**: `0.5` is half its error,
`1` is equal, and `2` is twice the error. The first included PID controller is
the baseline; otherwise the first supplied policy is used and named.

Boxes show the middle 50% of ratios and the median. Whiskers reach observed
values within 1.5 interquartile ranges; outliers and the largest ratio's seed
remain visible. Positive ratios use a log axis; a valid zero policy cost uses a
linear axis. Small baseline errors can amplify ratios, so also read absolute
tracking metrics and the safety summary.

Unsafe, incomplete, unsupported, or zero-baseline-cost pairs are excluded from
box statistics. Affected rows show valid/total counts and exclusion reasons;
no valid pairs produces `N/A`. Exclusion from a box does not remove those cases
from aggregate safety rates. A single-policy training report has no paired boxplot.

Simulation safety results apply to the documented models and protocols. Hardware
integration and experimental support are covered by their owning scenario guides.
