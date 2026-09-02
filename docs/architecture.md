# Features and workflow

AIO-Gym brings process simulation, classical control, reinforcement learning,
and evaluation into one Python workflow. A policy trained in the package can be
loaded and compared with PID or MPC without a separate evaluation program.

```text
make_env -> reset/step -> controller -> collect -> train
         -> save/load -> evaluate/compare
```

The [Quickstart](quickstart.md) provides runnable examples. This page explains
what each stage means and where its boundaries are.

## Choose a process-control scenario

Each scenario provides a physical model, observations, actions, control targets,
rewards, safety limits, and fixed test protocols. Use:

- `aiogym.list_scenarios()` to see available processes;
- `aiogym.list_parameters(id)` to inspect adjustable model parameters;
- `aiogym.list_rewards(id)` to see available rewards;
- `aiogym.list_benchmarks(id)` to see available fixed tests;
- `aiogym.make_env(id, ...)` to create a Gymnasium environment.

The scenario guides define the state order, physical units, supported parameter
overrides, disturbance names, and controller settings for each process.

An ordinary environment is for simulation, data collection, or training. It may
start from a user-supplied physical state through `initial_state=`. This changes
only the initial state; it does not infer a matching steady action or target.

A randomized environment generates a new feasible training episode at each
reset. A Benchmark environment is evaluation-only: it owns the complete test
case, fixes the plant settings and protocol, and rejects training variation or
manual initial-state overrides. This keeps policies directly comparable.

## Run classical or learned controllers

The built-in controllers are PID, successive-linearization MPC, hold, and
random. DDPG, PPO, SAC, TD3, and RLPD are available with the `rl` installation
extra. Loaded models and built-in controllers use the same policy interface, so
the same evaluation and comparison functions work for both.

Every policy receives the environment observation plus only the current step,
physical time, and public reference. Configured measurement noise and delay are
therefore applied consistently to PID, MPC, and learned policies.

The `info` mapping is diagnostic output, not an additional policy input. It may
record true state, applied action, safety margins, physical disturbances, and
the complete episode schedule for users and saved results without exposing
hidden simulator values to a controller.

MPC retains its configured nominal process model, but it does not receive hidden
runtime disturbance values. Any external condition intended for control must be
an explicit observation channel so every policy receives the same measurement.

## Collect data and train policies

`collect()` records complete trajectories from a controller or learned policy.
The resulting Dataset includes observations, rewards, safety information, and
the distinction between:

- the action requested by the policy;
- the action after optional delay or actuator faults;
- the physical action ultimately applied to the model.

A Dataset can be inspected directly, used for behavior cloning with supported
online algorithms, or supplied to RLPD. Dataset compatibility includes the
scenario, reward, physical parameters, training variation, control interval,
and observation/action shapes.

`train()` is the single training and continuation entry point. A completed run
writes:

- `model.zip`: the final trainable checkpoint;
- `training_curve.json` and `training_curve.svg`;
- `metadata.json` with the run settings;
- `best/model.zip`, `best/tracking.svg`, and `evaluation_history.json` when
  periodic validation is enabled.

Use `load_policy()` to run a checkpoint. Use `resume_from=` with `train()` to
continue optimization in a new output directory; `steps` is the additional
environment-step budget. The source checkpoint remains unchanged.

A trainable checkpoint retains the optimizer and algorithm-owned replay state
needed for continuation. Before restoring them, AIO-Gym verifies the scenario,
reward, parameters, training variation, control interval, and interface shapes.
Continuation reuses the recorded algorithm settings and seed instead of
starting a second training workflow.

## Train across varied conditions

An ordinary training environment can randomize feasible operating points and
add physical disturbances, measurement noise, observation or action delay, and
temporary actuator loss of effectiveness. These options do not change the
number or physical meaning of observation and action channels.

- `randomize=True` samples a scenario-owned feasible operating condition;
- `boundary_probability` chooses how often randomized training starts from a
  reachable state near a safety boundary;
- `disturbance` changes the simulated process or load;
- `noise` and `delay` change the policy measurement or command path;
- `fault` changes actuator effectiveness after the policy command.

Scenario guides document the physical meaning and timing of their randomized
conditions. An ordinary environment may also accept a piecewise-constant
schedule for disturbances declared by that scenario. Custom schedules cannot
be combined with the scenario's automatic disturbance sampler or a fixed
Benchmark.

## Evaluate fairly

Built-in Benchmarks answer three common questions:

- `tracking`: can the controller follow a target?
- `disturbance-rejection`: can it recover after the process changes?
- `boundary-safety`: can it remain safe near an operating limit?

For a Benchmark, a seed selects one complete reproducible physical case. Every
policy in `compare_policies()` receives the same ordered seeds. The standard
formal comparison uses seeds `0--19`.

Each scenario owns the physical construction of its Benchmark cases and its
settling or recovery bands. Boundary cases are generated from reachable process
conditions rather than arbitrary state vectors. These acceptance bands are
separate from observation normalization and reward weighting.

Comparison ranks the mean safety metrics across all cases before performance
metrics. Remaining metrics use the aggregate and direction declared by the
Benchmark, commonly a median. The exact order is stored in
`comparison.json["ranking_metrics"]`; a favorable median performance score
cannot hide an unsafe case.

Use `evaluate()` for one policy and `compare_policies()` for two or more. Both
save per-case metrics and trajectories; comparison additionally produces one
common ranking and figure.

## Select the best checkpoint

Periodic validation is separate from formal Benchmark evaluation. It uses a
second `randomize=True` environment and the shared fixed seeds `1000--1019`.
Disturbance, noise, delay, and actuator faults remain disabled so checkpoint
selection compares the shared randomized operating cases directly.

Every checkpoint candidate is ordered by:

1. higher mean safe-completion rate;
2. longer mean episode length, so later unsafe termination is preferred;
3. better worst 10th-percentile value of the primary validation metric;
4. better median value of that metric.

The selected candidate is saved as `best/model.zip`. The top-level `model.zip`
always remains the final learner for continuation and late-training diagnosis.
Formal learned-policy comparisons use `best/model.zip` uniformly across
algorithms and use held-out Benchmark cases rather than validation cases.

## Keep experiments repeatable

For results that other users can inspect or rerun:

- compare policies on the same scenario, Benchmark, and ordered seeds;
- keep the reward and model parameters unchanged within one comparison;
- retain `comparison.json` and `trajectories.npz`, not only the SVG;
- keep the training settings, Dataset information, library versions, and source
  revision with the result;
- report safety separately from tracking quality.

Training metadata records the algorithm, seed, completed steps, environment
settings, training variation, and relevant library versions. When available it
also records the Git commit, but that value does not describe uncommitted source
changes.

## Understand the output files

| Workflow | Main output |
|---|---|
| Data collection | `metadata.json` and compressed episode files |
| Training | `model.zip`, metadata, learning-curve JSON/SVG, and optional `best/` output |
| Evaluation | JSON with per-case results and trajectories |
| Comparison | `comparison.json`, `comparison.svg`, `trajectories.npz` |

Without an explicit comparison output, Benchmark results are written under
`runs/<scenario>/benchmarks/<benchmark>/` and replace the three result
files. Provide a separate empty output directory when retaining another
comparison.

The SVG shows trajectories for the first seed and returns for all evaluated
seeds. The NPZ archive keeps every numeric trajectory for further analysis.
Time axes use the selected scenario's documented physical unit.

## Know the scope

AIO-Gym's safety metrics describe the included simulation models and fixed test
protocols. They are not a substitute for plant calibration, hardware
interlocks, commissioning, or operational risk assessment. Any experimental
hardware support is opt-in and documented by its owning scenario.
