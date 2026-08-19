# Architecture

AIO-Gym has one small dependency direction:

```text
scenarios -> core <- controllers
                  <- workflows <- CLI
experimental (opt-in)
```

Users need six concepts:

- Scenario: one physical process model, default episode, and training sampler;
- Benchmark: one fixed evaluation protocol, Reward, and ranking metrics;
- Reward: the step reward and non-benchmark episode metrics;
- Env: the Gymnasium `reset()` and `step()` interface;
- Policy: PID, successive-linearization MPC, hold, random, or a loaded
  registered-algorithm model;
- Workflow: collect, train, evaluate, or compare.

`EpisodeSpec` is an internal resolved value containing initial state, initial
applied action, reference, horizon, disturbances, and schedules. It is
not another public selector.

## Environment boundary

`make_env()` is the only public environment-construction entry point:

```text
resolved model + default/benchmark/training episode
    -> process environment
    -> operating-condition and physical-disturbance episode sampling
    -> action delay and actuator fault
    -> observation delay and noise
```

Every mode of one Scenario keeps the same observation and physical action
meaning. `quadruple` uses a 6-dimensional observation and two pump actions.
`three_tank` uses a 13-dimensional observation (six process-tank measurements,
the dynamic reservoir temperature, and six controlled-output references) and
five physical actions. Its controlled output remains the six process-tank
level/temperature values in state order; reservoir temperature is a measured
internal state without a setpoint.

Policy actions pass through optional action delay and loss-of-effectiveness.
The base environment then applies an actuator slew limit only when the Scenario
model explicitly declares one; the current built-in simulation models do not.
Step info distinguishes the original `commanded_action`, the delivered
`channel_action`, and the final `applied_action`. Observation noise is
normalized policy-channel noise, not a claim about a particular physical
sensor. It applies to measured observation channels but never to known
setpoint/reference channels. Controllers consume the same noisy observation
exposed to every other policy. Observation delay, when enabled, still applies
to the complete policy observation. Observation schema rows declare this
distinction with `kind` set to `measurement`, `reference`, or `action`.

Training-condition, physical-disturbance, Benchmark-case, observation, delay,
and fault sampling use independent random streams. For the same reset seed,
enabling a physical disturbance or observation noise does not change the
sampled tracking condition. Physical disturbances enter the model dynamics but
are not added as direct policy-observation channels.

## Benchmark and training boundary

Each Scenario owns one or more scenario-specific fixed Benchmarks. A Benchmark
declares its Reward, ranking metrics, and deterministic case distribution. A
non-negative case seed is resolved once at `reset()` into one complete
`EpisodeSpec`; no case table or run-time event-time equilibrium solve is used.
Benchmarks are evaluation-only, use exact observations, and always use the
Scenario's default resolved model parameters. `make_env()` rejects caller
overrides of parameters, Reward, randomization, physical disturbances, or
channel variation, and `train()` rejects a benchmark environment. Randomized
training episodes use a separate random stream from Benchmark cases.

Safety constraints are evaluated before the Reward result is finalized. A
terminal safety violation adds the Reward's explicit safety penalty. Formal
Benchmark ranking first minimizes unsafe rate and then maximizes cumulative
Reward return. Safety margin, time to violation, tracking error, and disturbance
recovery remain diagnostic metrics.

## Workflow boundary

Python workflows receive an environment constructed by the caller. They never
reconstruct it from names and never close it. The CLI parses arguments, creates
one environment, calls one workflow, closes the environment, and prints a
compact JSON summary while the workflow preserves complete artifacts.

Artifacts are deliberately small:

- Dataset: `metadata.json` plus one `episode-*.npz` per episode;
- training: the self-describing AIO-Gym `model.zip`, `metadata.json`,
  `training_curve.json`, and a
  dependency-free `training_curve.svg`; periodic evaluation also writes
  `evaluation_history.json`, `best/model.zip`, and `best/tracking.svg`;
  behavior-cloning pretraining additionally writes `behavior_cloning.json`;
- evaluation: one JSON file with raw per-seed trajectories and summaries;
- comparison: `comparison.json` and `comparison.svg`; benchmark comparisons
  default to `runs/<scenario>/<benchmark>/` and replace those two managed files.

There is one evaluation path. Without `benchmark=`, metrics come from Reward.
With `benchmark=`, metrics and lexicographic ranking come from Benchmark.
Comparison reuses those evaluation results, ranks metric medians, and overlays
the complete trajectory from the first ordered seed. Its return panel retains
one point for every evaluated seed plus the median, while the JSON keeps all
trajectories and per-step summaries. Evaluation seeds are Benchmark case seeds;
policy randomness is held at seed zero so a case seed does not simultaneously
change both the plant case and controller random stream. Fixed episode factories
may intentionally resolve several case seeds to the same episode.

Training has one algorithm-backend boundary. A backend constructs and advances
its model, reports one `TrainingStep` per environment transition, saves and
loads its checkpoint payload, and exposes the live or loaded model as the
existing public `Policy`. AIO-Gym wraps that payload in one `model.zip` format
containing `manifest.json` and `payload.zip`; the manifest identifies the
algorithm and training environment. Loading requires the target environment and
validates its Scenario, Reward, parameters, control interval, and interface
shapes before the backend payload is extracted. AIO-Gym remains responsible for
the training curve, periodic
evaluation, best-checkpoint selection, artifacts, evaluation, and comparison.
The built-in DDPG/PPO/SAC/TD3 implementations use this same public contract as
external backends. Any additional SB3 `BaseAlgorithm` subclass can use the
shared adapter through `register_sb3_algorithm()`.

Behavior cloning is an optional backend capability inside that same training
path. It validates a Dataset against the target environment before invoking the
backend hook. The built-in DDPG/SAC/TD3 hook supervises
`observation -> commanded_action` and then hands the same model to online
learning. It does not define a second checkpoint or policy loader.

## Experimental hardware

Hardware transport, calibration, and real-log code lives under
`aiogym.experimental.three_tank_hardware`. The hardware environment uses the
Three-Tank tracking benchmark protocol, including its six-output steady
references and measured reservoir temperature, and is not imported by the
default package.
