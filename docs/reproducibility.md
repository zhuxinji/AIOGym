# Reproducibility

AIO-Gym records the inputs needed to repeat a local run without adding a second
experiment-management system.

## Dataset metadata

A Dataset records:

- scenario, reward, benchmark or training-randomization mode;
- complete resolved model parameters and model-native units discoverable via
  `list_parameters()`;
- resolved noise, delay, and fault configuration;
- observation and action shapes and control time step;
- policy description, base seed, and per-episode seed;
- each episode's actual `EpisodeSpec`, sampled family, and channel variation.

Transition arrays distinguish policy action, delayed/faulted channel action,
and slew-limited applied action. They also include constraint costs and safety
margins.

## Training metadata

Training records:

- algorithm, requested steps, and training seed;
- actual environment steps and the fixed training-curve recording interval;
- fully resolved algorithm keyword arguments, including workflow defaults and
  caller overrides;
- the complete environment configuration listed above;
- Python, Stable-Baselines3, Torch, and AIO-Gym versions;
- the Git commit when the checkout provides one.

The Git commit does not encode uncommitted source changes. Commit the training
protocol before producing results that must be independently reproduced.

`training_curve.json` stores mean reward over fixed environment-step windows
and complete-episode return, length, and termination outcome. The accompanying
SVG uses environment steps on an x-axis that starts at zero and can be
regenerated with `aiogym.plot_training_curve()`.

Benchmark environments are rejected by `train()`. Benchmark creation also
rejects model-parameter and Reward overrides, so one Benchmark id cannot refer
to different plants or scoring signals. Every concrete Benchmark fixes its
measurement-noise parameters, and caller-supplied channel variation remains
invalid for Benchmark construction.

## Evaluation and comparison

Evaluation records the ordered seed list, maximum step limit, policy metadata,
per-episode resolved protocol, metrics, and trajectories. Scalar aggregates
include mean, standard deviation, median, median absolute deviation, minimum,
and maximum. Trajectory summaries contain per-step median/minimum/maximum bands
for true state, output, reference, commanded/channel/applied action, reward,
disturbance, constraint cost, and minimum safety margin. Benchmark evaluation
also records its ordered ranking metrics. Comparison uses the same ordered
seeds for every policy, ranks metric medians lexicographically, and writes both
JSON and SVG artifacts. A benchmark comparison with no explicit output replaces
those two managed files under `runs/<scenario>/<benchmark>/`; an explicit output
must be an empty directory and is not overwritten.

These fields make runs inspectable and repeatable. They do not turn short runs
into credible performance evidence.
