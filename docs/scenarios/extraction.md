# Multistage extraction model

The built-in `extraction` Scenario models a five-stage counter-current
liquid-gas extraction column. It adapts the nonlinear stage balances from the
PC-Gym `multistage_extraction` model for simulation, PID/MPC control, Dataset
collection, RL training, and fixed Benchmark comparisons.

The physical interface is:

- ten states, alternating liquid and gas solute concentration for stages 1--5;
- controlled output `stage_5_liquid_concentration` (`CX5`);
- action `[liquid_feed_flow, gas_feed_flow]`, normalized to `[0, 1]`;
- physical action ranges `5--500 volume/h` and `10--1000 volume/h`;
- disturbances `liquid_feed_concentration`, `gas_feed_concentration`, and
  `mass_transfer_coefficient`.

Both RL policies and MPC command both flow inputs. The PI baseline holds the gas
flow at its nominal normalized value `0.5` and regulates `CX5` with the liquid
flow. Supplying only one action value is an interface error.

The policy observation contains eleven normalized values: all ten column states
and the `CX5` reference. Disturbances are recorded in step information but are
not appended to the policy observation. The controller interval is `0.1 h`
(six minutes) and the plant is integrated internally with a maximum `0.01 h`
step. These are simulated hours derived from the equation dimensions; they do
not claim that a real uncalibrated column has the same dynamic response.

At the default `CX5=0.30` reference, the exact steady-state inverse gives
normalized action `(0.320519, 0.5)`, corresponding to physical flows
`(163.657, 505)`. The inverse fixes the nominal gas action, solves the
counter-current column equilibrium, and then solves the liquid action required
for the target. Infeasible targets return no operating point instead of
clipping the state or action.

The deterministic training episode starts from this `CX5=0.30` equilibrium and
immediately tracks `CX5=0.40` for 100 steps, or 10 h.

## Model source and intended use

The Scenario uses the five-stage equations and parameters declared by the
upstream PC-Gym source. Its extraction training case uses liquid/gas action
ranges `5--500` and `10--1000` with `CX5` targets `0.3` and `0.4`:

- [PC-Gym extraction equations](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py#L1166-L1236)
- [PC-Gym extraction training case](https://github.com/MaximilianB2/pc-gym/blob/main/pc-gym_paper/train_policies/multistage_extraction/me_train.py#L44-L89)

This is still a normalized benchmark model derived from literature, not a
column calibrated against plant measurements. Units and controller performance
must not be interpreted as deployment validation.

## Parameters

| Parameter | Default | Unit |
|---|---:|---|
| `liquid_stage_volume` | 5 | volume |
| `gas_stage_volume` | 5 | volume |
| `equilibrium_constant` | 1 | dimensionless |
| `mass_transfer_coefficient` | 5 | 1/h |
| `equilibrium_exponent` | 2 | dimensionless |
| `liquid_feed_concentration` | 0.6 | fraction |
| `gas_feed_concentration` | 0.05 | fraction |
| `minimum_liquid_flow` | 5 | volume/h |
| `maximum_liquid_flow` | 500 | volume/h |
| `minimum_gas_flow` | 10 | volume/h |
| `maximum_gas_flow` | 1000 | volume/h |
| `nominal_gas_action` | 0.5 | fraction |

All parameters accept direct validated overrides. Unknown, non-finite, or
physically inconsistent mappings fail at environment construction.

## Benchmarks

```python
import aiogym

print(aiogym.list_benchmarks("extraction"))
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | exact `CX5=0.30` equilibrium start that immediately tracks one case-seeded target | 100 steps / 10 h | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded feed concentrations, mass transfer, event time, and recovery time | 580 steps / 58 h | unsafe rate, cumulative return |
| `boundary-safety` | normal state is forward pre-run with minimum liquid and near-maximum gas flow until `CX5` reaches a case-seeded `0.01--0.04` lower boundary, followed by a `0.30` target | 100 steps / 10 h | unsafe rate, cumulative return |

Tracking targets are sampled in `0.12--0.44`, with at least a `0.05` move from
the initial output; tracking begins on the first control step. The model derives
the state and steady action for the target,
requires actions to remain in `0.05--0.95`, and verifies a zero ten-state
derivative before accepting the case. Identical seeds give every compared
policy identical targets.

Disturbance cases begin at step `140--240` and last `200--min(320, 464-start)` steps. They
sample liquid feed concentration in `0.45--0.75`, gas feed concentration in
`0.02--0.09`, and mass-transfer coefficient in `3.5--6.5`. Boundary cases
use a forward pre-run from the normal state, with minimum liquid and
near-maximum gas flow, until `CX5` reaches a sampled threshold in `0.01--0.04`.
The resulting state is reachable but is not required to be an equilibrium. Seeds `0--19`
select 20 distinct episodes for all three Benchmarks.

The latest disturbance restoration is at step 464 (46.4 h). The 580-step
protocol then reserves 58 recovery steps before the final 58-step stability window. Boundary safety uses the same 10 h window as
tracking.

## Training variation

`randomize=True` starts every episode from the exact default interior equilibrium
by default and immediately tracks one feasible target. Setting
`boundary_probability=p` makes fraction `p` start from the same minimum-liquid,
high-gas forward pre-run used by the `0.01--0.04`
low-concentration boundary Benchmark. All training episodes use the tracking
Benchmark's 100-step horizon. `disturbance=True` samples all three
physical disturbances, begins at step `23--40`, and lasts `33--min(53, 80-start)` steps (restored by step 80). Noise, delay, and
actuator loss-of-effectiveness remain the shared optional channel variations.

## Quick start and controllers

The extraction PI configuration uses `(Kp, Ki, Kd) = (1.0, 10.0, 0)` on the liquid-flow
channel and zero gains on the gas-flow channel, with a fixed default-action bias.
The PI gains were selected on tracking seeds `1000–1007`, then evaluated without
further tuning on seeds `0–19` of all three Benchmarks.
The MPC uses a `0.01 h`
prediction step, a 10-step horizon, model steady-state feedforward, and both
flow variables. Its parameters were selected using five randomized training
episodes with physical disturbances, not the fixed formal Benchmark cases.

```python
import aiogym

env = aiogym.make_env("extraction", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
result = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc},
    seeds=range(20),
)
env.close()
```

Use the [task guide](../workflows.md) for Dataset collection and RL
training with `scenario = "extraction"`. Generated comparison figures and
trajectories use hours, matching this model's native time unit; current scores
remain in `runs/extraction/benchmarks/<benchmark>/comparison.json`.
