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

- algorithm, requested steps, and training seed;
- actual environment steps and the fixed training-curve recording interval;
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

`training_curve.json` stores mean reward over fixed environment-step windows
and complete-episode return, length, and termination outcome. The accompanying
SVG uses environment steps on an x-axis that starts at zero and can be
regenerated with `aiogym.plot_training_curve()`.

When behavior-cloning pretraining is enabled, `behavior_cloning.json` records
the Dataset path and schema, source policy, transition count, supervised
configuration, action-field semantics, and per-epoch mean loss. The training
metadata links this report. The final checkpoint is one AIO-Gym `model.zip`
containing `manifest.json` and the selected backend's `payload.zip`; the
manifest records the checkpoint schema, algorithm, runtime, environment, and
policy metadata, so loading does not repeat the algorithm id. `load_policy()`
requires the target environment and rejects incompatible Scenario, Reward,
model parameters, control interval, or observation/action shapes before the
backend payload is loaded.
Raw SB3 archives and earlier checkpoint layouts are not this format and are
rejected instead of being guessed or migrated implicitly.

Periodic training evaluation uses a separate non-Benchmark environment. Its
compact `evaluation_history.json` records the Reward's primary metric at step
zero, at each configured interval, and at the final step. `best/model.zip` is
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
seeds for every policy, ranks metric medians lexicographically, and writes both
JSON and SVG artifacts. The SVG trajectory panels show only the first ordered
seed, while the return panel plots every seed and its median; exact trajectories,
return values, and aggregate summaries remain in JSON. A benchmark comparison
with no explicit output replaces those two managed files under
`runs/<scenario>/<benchmark>/`; an explicit output must be an empty directory
and is not overwritten.

These fields make runs inspectable and repeatable. They do not turn short runs
into credible performance evidence.
