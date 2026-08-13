# Architecture

AIO-Gym has one small dependency direction:

```text
scenarios -> core <- controllers
                  <- workflows <- CLI
experimental (opt-in)
```

Users need six concepts:

- Scenario: one physical process model, default episode, and training sampler;
- Benchmark: one fixed evaluation protocol and its ranking metrics;
- Reward: the step reward and non-benchmark episode metrics;
- Env: the Gymnasium `reset()` and `step()` interface;
- Policy: PID, MPC, hold, random, or a loaded SB3 model;
- Workflow: collect, train, evaluate, or compare.

`EpisodeSpec` is an internal resolved value containing initial state, reference,
horizon, disturbances, and schedules. It is not another public selector.

## Environment boundary

`make_env()` is the only public environment-construction entry point:

```text
resolved model + default/benchmark/training episode
    -> process environment
    -> episode sampling (randomize=True)
    -> action delay and actuator fault
    -> observation delay and noise
```

Every mode of one Scenario keeps the same observation and physical action
meaning. `quadruple` uses an 8-dimensional observation and two pump actions.
`three_tank` uses a 17-dimensional observation and five physical actions.

Policy actions pass through action delay and loss-of-effectiveness before the
base environment applies physical actuator slew limits. Step info distinguishes
the original `commanded_action`, the delivered `channel_action`, and the final
`applied_action`. Observation noise is normalized policy-channel noise, not a
claim about a particular physical sensor. Controllers consume the same noisy
observation exposed to every other policy.

Episode, observation, delay, and fault sampling use independent random streams.
For the same reset seed, enabling observation noise does not change the sampled
episode.

## Benchmark and training boundary

Each Scenario owns exactly:

- `tracking`;
- `disturbance-rejection`;
- `boundary-safety`.

Benchmarks are fixed and evaluation-only. They always use the Scenario's
default resolved model parameters, default Reward, and Benchmark-declared
measurement noise. `make_env()` rejects caller overrides of parameters, Reward,
randomization, or channel variation, and `train()` rejects a benchmark
environment. Randomized training episodes cover the same three challenge
families without copying the fixed benchmark timelines and values.

Safety constraints are evaluated before the Reward result is finalized. A
terminal safety violation adds the Reward's explicit safety penalty. Formal
benchmark metrics include unsafe rate, safety margin, time to violation,
tracking error, and disturbance recovery where applicable.

## Workflow boundary

Python workflows receive an environment constructed by the caller. They never
reconstruct it from names and never close it. The CLI parses arguments, creates
one environment, calls one workflow, closes the environment, and prints JSON.

Artifacts are deliberately small:

- Dataset: `metadata.json` plus one `episode-*.npz` per episode;
- training: `model.zip`, `metadata.json`, `training_curve.json`, and a
  dependency-free `training_curve.svg`;
- evaluation: one JSON file with raw per-seed trajectories and summaries;
- comparison: `comparison.json` and `comparison.svg`; benchmark comparisons
  default to `runs/<scenario>/<benchmark>/` and replace those two managed files.

There is one evaluation path. Without `benchmark=`, metrics come from Reward.
With `benchmark=`, metrics and lexicographic ranking come from Benchmark.
Comparison reuses those evaluation results, ranks metric medians, and overlays
trajectory medians with seed-wise minimum/maximum bands.

## Experimental hardware

Hardware transport, calibration, and real-log code lives under
`aiogym.experimental.three_tank_hardware`. The hardware environment uses the
Three-Tank tracking benchmark protocol, including its complete six-output
steady references, and is not imported by the default package.
