# Reproducibility

AIO-Gym records the inputs needed to repeat a local run without adding a second
experiment-management system.

## Dataset metadata

A Dataset records:

- scenario, reward, benchmark or training-randomization mode;
- complete resolved model parameters and model-native units discoverable via
  `list_parameters()`;
- resolved physical-disturbance, noise, delay, and fault configuration;
- observation and action shapes and control time step;
- policy description, base seed, and per-episode seed;
- each episode's actual `EpisodeSpec`, including its initial applied action,
  sampled family, physical-disturbance schedule, and channel variation.

Transition arrays distinguish policy action, delayed/faulted channel action,
and slew-limited applied action. They also include constraint costs and safety
margins.

## Training metadata

Training records:

- algorithm, requested added steps, initial checkpoint steps, cumulative actual
  environment steps, and training seed;
- the source checkpoint path for continued training and the fixed
  training-curve recording interval;
- fully resolved algorithm keyword arguments, including workflow defaults and
  caller overrides;
- periodic deterministic evaluation cadence, seed, primary metric, and best
  checkpoint step when enabled;
- the complete environment configuration listed above;
- algorithm-backend runtime versions (the built-in backends record Python,
  Stable-Baselines3, Torch, and AIO-Gym);
- the Git commit when the checkout provides one.

The Git commit does not encode uncommitted source changes. Commit the training
protocol before producing results that must be independently reproduced.

`training_curve.json` stores its initial and cumulative actual step, mean reward
over fixed environment-step windows, and complete-episode return, length, and
termination outcome. A fresh run starts at zero; a continued run starts at the
source checkpoint's completed step. The accompanying SVG can be regenerated
with `aiogym.plot_training_curve()`.

When behavior-cloning pretraining is enabled, `behavior_cloning.json` records
the Dataset path and schema, source policy, transition count, supervised
configuration, action-field semantics, and per-epoch mean loss. The training
metadata links this report. The final checkpoint is one AIO-Gym `model.zip`
containing `manifest.json` and the selected backend's `payload.zip`; the
manifest records the checkpoint schema, algorithm, runtime, environment, and
policy metadata. Its training section records completed steps, seed, resolved
algorithm arguments, and required Dataset identity. The backend payload retains
the parameters, optimizer state, and replay state needed for another `train()`
invocation. `load_policy()`
requires the target environment and rejects incompatible Scenario, Reward,
model parameters, control interval, or observation/action shapes before the
backend payload is loaded.

Reward identity denotes one fixed scoring meaning. In AIO-Gym 0.15,
Three-Tank `regulation` scores only `[h1, h2, h3]`, with a `0.1 m` error scale
for every level. Its state/observation/reference/action shapes are `(3,)`,
`(6,)`, `(3,)`, and `(4,)`. Artifacts must be regenerated
whenever Reward or Episode semantics change; an artifact is never relabeled to
represent a different scoring contract. Three-Tank default, randomized
training, and formal tracking Episodes all expose one target at reset and run
for 600 seconds. Its deterministic default resolves to formal tracking case
seed 0. Interior sampled tracking requires at least a `0.05 m` change in every
level but imposes no additional move cap inside the `0.125--0.4 m` range.
Datasets and checkpoints collected under
the earlier thermal-state or five-action contracts are incompatible with this
hydraulic-only task.

All built-in Scenarios align default and randomized training timing with their
fixed tracking Benchmark: one target is active at reset and the horizon is
Crystallization `50 s`, CSTR `225 s`, Extraction `10 h`, Heater `300 s`, HVAC
`300 s`, Quadruple-Tank `180 s`, or Three-Tank `600 s`. Optional training
disturbance events are sampled inside these horizons. Disturbance-rejection and
boundary-safety Benchmarks retain their separate evaluation-only protocols.

The fixed disturbance-rejection/boundary-safety horizons are Crystallization
`100/100 s`, CSTR `400/100 s`, Extraction `58/10 h`, Heater `700/450 s`, HVAC
`1050/300 s`, Quadruple-Tank `500/180 s`, and Three-Tank `1800/600 s`.
Shortened windows retain every declared event and an evidence-based recovery
or settling buffer. Crystallization's finite-batch windows and Heater
disturbance rejection remain longer where the target definition or observed
controller behavior does not support shortening.

Training metadata also records the validated Dataset path, schema, collector
policy, episode count, and transition count. RLPD retains that Dataset for all
optimizer updates; DDPG/SAC/TD3 behavior cloning consumes it only during the
supervised warm start.

Continued training passes the source `model.zip` through `resume_from=` and
uses `steps` as the additional environment-step budget. The algorithm,
environment, seed, and resolved algorithm arguments must match the checkpoint.
RLPD also requires the same validated Dataset. Behavior cloning is a fresh-run
warm start and is not repeated during continuation. Each invocation writes a
new empty output directory, leaving its source checkpoint unchanged.
`load_policy()` accepts only the self-describing AIO-Gym `model.zip` format and
validates its current checkpoint schema before loading the backend payload.

Periodic training evaluation uses a separate non-Benchmark environment. Its
compact `evaluation_history.json` records the Reward's primary metric at the
invocation's initial step, at each configured cumulative interval, and at the
final step. `best/model.zip` is
selected lexicographically by safe completion, episode length, and that primary
metric. `best/tracking.svg` plots that selected checkpoint on the same fixed
training case, while `model.zip` always remains the final learner.

Benchmark environments are rejected by `train()`. Benchmark creation also
rejects model-parameter and Reward overrides, so one Benchmark id cannot refer
to different plants or scoring signals. Benchmarks use exact observations;
caller-supplied physical disturbances and channel variations remain invalid.

## Evaluation and comparison

Evaluation records the ordered seed list, maximum step limit, policy metadata,
per-episode resolved protocol, reset parameters, metrics, and trajectories. On
a Benchmark, each non-negative seed is a case seed: reset deterministically
constructs the complete `EpisodeSpec` once, and `episode_parameters.case_seed`
records the mapping. Comparison resolves the same ordered cases for every
policy while holding the policy random stream at seed zero. Scalar aggregates
include mean, standard deviation, median, median absolute deviation, minimum,
and maximum. Trajectory summaries contain per-step median/minimum/maximum bands
for true state, output, reference, commanded/channel/applied action, reward,
disturbance, constraint cost, and minimum safety margin. Benchmark evaluation
also records its ordered ranking metrics. Comparison verifies the same ordered
seeds for every policy, ranks metric medians lexicographically, and writes
a compact JSON report, an SVG figure, and a compressed NPZ trajectory archive.
The SVG trajectory panels show only the first ordered seed, while the return
panel plots every seed and its median. The trajectory schema records the
Scenario's time unit, and every SVG time axis uses that declared unit rather
than assuming seconds. Exact returns, metrics, and resolved
Episodes remain in JSON; complete numeric trajectories remain in NPZ, with
array ids and column names declared by the JSON `trajectory_archive` section.
A benchmark comparison with no explicit output replaces those three managed
files under `runs/<scenario>/benchmarks/<benchmark>/`; an explicit output must
be an empty directory and is not overwritten.

Training output paths follow
`runs/<scenario>/training/<run-name>/seed-<seed>/`. Training and formal
Benchmark results therefore remain separate artifact types under the owning
Scenario.

The built-in formal comparison artifacts use the ordered seeds `0--19`. For
every built-in Scenario, all three Benchmark factories resolve those seeds to
20 distinct `EpisodeSpec` values. A comparison retains the same resolved list
for every policy, so variation is in the physical case rather than the policy's
random stream.

These fields make runs inspectable and repeatable. They do not turn short runs
into credible performance evidence.
