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
`three_tank` uses a 6-dimensional observation (three level measurements and
three level references) and four physical actions. Its state and controlled
output are both `[h1, h2, h3]`; temperature is absent from the model contract.
Three-Tank exposes one `regulation`
Reward, with the same `0.1 m` scale applied to all three levels in default
training, randomized training, and every Benchmark.
Its sampled tracking transitions preserve broad operating-point coverage. Each
interior level move is at least `0.05 m`, with no additional move cap inside the
declared `0.125--0.4 m` operating range; these Episode limits do not change
observation or action meaning.
Every Scenario's default training, randomized training, and fixed tracking
Benchmark use one target from reset and the same Scenario-owned tracking
horizon. Continuous-process tracking starts from a feasible operating point and
immediately exposes a different target. Crystallization follows the same
time-zero convention with a reachable finite-batch endpoint. Training and
tracking therefore have no in-episode reference event; they remain distinct
only in deterministic default-case selection, randomized training sampling,
and fixed Benchmark case-seed ownership. Three-Tank intentionally resolves its
deterministic default to the tracking Benchmark's seed-0 Episode while keeping
the training environment itself non-Benchmark.

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
- comparison: compact `comparison.json`, `comparison.svg`, and compressed
  `trajectories.npz`; benchmark comparisons default to
  `runs/<scenario>/benchmarks/<benchmark>/` and replace those three managed
  files.

Physics models declare the time unit used by their numerical integration.
Crystallization, CSTR, Heater, HVAC, Quadruple-Tank, and Three-Tank use seconds;
Extraction uses hours. Evaluation carries this unit in its trajectory schema
and comparison plots label every time axis from that schema.

Local run artifacts are grouped first by Scenario and then by artifact type:
training outputs use `runs/<scenario>/training/<run-name>/seed-<seed>/`, while
formal comparisons use `runs/<scenario>/benchmarks/<benchmark>/`.

There is one evaluation path. Without `benchmark=`, metrics come from Reward.
With `benchmark=`, metrics and lexicographic ranking come from Benchmark.
Comparison reuses those evaluation results, ranks metric medians, and overlays
the complete trajectory from the first ordered seed. Its return panel retains
one point for every evaluated seed plus the median. The compact JSON retains all
per-seed Episodes and metrics but only the plotted seed's trajectory; the NPZ
archive holds every numeric trajectory as `float64` arrays. Its keys combine the
entry id recorded in `trajectory_archive.entries` with a field name using
`<entry-id>__<field>`, while `trajectory_archive.columns` defines vector and
mapping columns. Evaluation seeds are Benchmark case seeds; policy randomness
is held at seed zero so a case seed does not simultaneously change both the
plant case and controller random stream. The built-in formal artifact protocol
uses seeds `0--19`; every built-in Benchmark factory resolves those seeds to 20
distinct physical Episodes.

All trainable-algorithm contracts, SB3 integration, learned-policy adaptation,
behavior cloning, and shipped RL configurations live under `aiogym.rl`.
Training has one algorithm-backend boundary. A backend constructs and advances
its model, reports one `TrainingStep` per environment transition, saves and
loads its trainable checkpoint payload, and exposes the live or loaded model as
the existing public `Policy`. The payload retains model parameters, optimizer
state, and algorithm-owned replay state. AIO-Gym wraps that payload in one
`model.zip` format
containing `manifest.json` and `payload.zip`; the manifest identifies the
algorithm, training environment, completed environment steps, seed, resolved
algorithm configuration, and required Dataset identity. Loading requires the
target environment and validates its Scenario, Reward, parameters, control
interval, and interface shapes before the backend payload is extracted.

`train(..., resume_from=...)` uses that same boundary to add the requested
number of environment steps. It requires the checkpoint's algorithm,
environment, seed, and resolved configuration, and writes a new empty output
directory. The new training curve and periodic-evaluation history cover that
invocation and use cumulative step coordinates beginning at the checkpoint's
completed step. AIO-Gym remains responsible for the training curve, periodic
evaluation, best-checkpoint selection, artifacts, evaluation, and comparison.
The built-in DDPG/PPO/RLPD/SAC/TD3 implementations use this same public
contract as external backends. RLPD is the native PyTorch backend; the other
four use the shared SB3 adapter. Any additional SB3 `BaseAlgorithm` subclass
can use `register_sb3_algorithm()`.

Behavior cloning is an optional backend capability inside that same training
path. It validates a Dataset against the target environment before invoking the
backend hook. The built-in DDPG/SAC/TD3 hook supervises
`observation -> commanded_action` and then hands the same model to online
learning. It does not define a second checkpoint or policy loader.

RLPD declares that a Dataset is required and receives the validated
`DatasetReader` in `AlgorithmBackend.learn()`. It uses observation, commanded
action, reward, next observation, and termination semantics on every update,
mixing fixed prior and growing online replay according to `offline_ratio`
(`0.5` by default). Dataset compatibility is validated once by the training
workflow.

## Experimental hardware

Hardware transport, calibration, and real-log code lives under
`aiogym.experimental.three_tank_hardware`. The hardware environment uses the
Three-Tank tracking benchmark protocol, including its three level references,
and is not imported by the default package.
