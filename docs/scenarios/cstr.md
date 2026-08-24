# Two-input, two-output CSTR model

The built-in `cstr` Scenario models an exothermic continuous stirred-tank
reactor through the common Scenario, Reward, Benchmark, controller, Dataset,
and training workflow contracts. Both physical commands and both controlled
variables are part of the public interface.

The physical interface is:

- state `[reactant_concentration, reactor_temperature]` in `mol/L` and `degC`;
- controlled output `[reactant_concentration, reactor_temperature]` in `mol/L`
  and `degC`;
- action `[feed_pump, cooling]`, with both commands normalized to `[0, 1]`;
- disturbances `feed_temperature`, `feed_concentration`, and
  `coolant_temperature`.

This is a 2-by-2 multivariable control environment. RL policies and MPC command
both actuators. The PI baseline uses the feed pump for concentration and the
cooling actuator for temperature. Supplying one action value or one setpoint is
an interface error.

The policy observation contains four normalized values: reactant concentration,
reactor temperature, concentration reference, and temperature reference. The
observation does not expose disturbances directly. The controller interval is
1 second and the plant is integrated internally with a maximum 0.01-second
step.

At the default `(0.117710 mol/L, 60 degC)` reference, the exact steady-state
inverse gives state `(0.117710, 60)` and action `(0.5, 0.263499)`. For any
requested concentration and temperature pair, the inverse solves the mass
balance for feed flow and the energy balance for cooling. It returns no
operating point when the requested pair is infeasible; it does not fix one
actuator or clip the physical solution.

The deterministic training Episode starts from that default equilibrium and
immediately tracks `(0.075 mol/L, 72 degC)` for 225 seconds.

## Parameters and evidence boundary

The main executable defaults are:

| Parameter | Default | Unit |
|---|---:|---|
| `maximum_dilution_rate` | 0.02 | 1/s |
| `feed_concentration` | 1.0 | mol/L |
| `pre_exponential_factor` | 1e8 | 1/s |
| `activation_temperature` | 7000 | K |
| `reaction_temperature_gain` | 120 | degC/(mol/L) |
| `cooling_coefficient` | 0.05 | 1/s |
| `coolant_temperature` | 10 | degC |
| `feed_temperature` | 20 | degC |
| `nominal_feed_action` | 0.5 | fraction |
| `temperature_trip` | 92 | degC |

The kinetics and thermal coefficients define a normalized reactor simulation.
They are not calibrated to a named reactor or validated against plant
measurements. This Scenario supports reproducible simulation and controller
comparison, not a process-deployment or physical-fidelity claim.

## Benchmarks

```bash
aiogym list benchmarks --scenario cstr
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded two-output equilibrium start that immediately tracks one independently sampled target | 225 steps / 225 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded feed temperature, concentration, coolant temperature, event time, and recovery time | 400 steps / 400 s | unsafe rate, cumulative return |
| `boundary-safety` | case-seeded `0.02--0.04 mol/L`, `86--90 degC` equilibrium start inside the `92 degC` trip limit, followed by the default target | 100 steps / 100 s | unsafe rate, cumulative return |

Each tracking seed samples the start and one target in
`0.02--0.20 mol/L` and `50--82 degC`. Tracking begins on the first control step;
the target changes
concentration by at least `0.02 mol/L` and temperature by at least `5 degC`.
The model derives both operating-point equilibrium actions, requires them to remain in
`0.05--0.95`, and verifies a zero model derivative before accepting the case.
Identical seeds therefore give every compared policy identical operating
conditions.

Disturbance cases begin at step `100--180` and last `100--180` steps. They
sample feed temperature in `12--32 degC`, feed concentration in
`0.75--1.30 mol/L`, and coolant temperature in `5--20 degC`. Boundary cases
sample an exact high-temperature equilibrium in `0.02--0.04 mol/L` and
`86--90 degC`. Seeds `0--19` resolve to 20 distinct Episodes for all three
Benchmarks.

The disturbance horizon leaves at least 40 seconds after the latest possible
event recovery; the observed maximum PI recovery was 23 seconds. Boundary
cases settled within 13 seconds for both controllers, so 100 seconds retains a
wide transient and safety margin.

## Training variation

`randomize=True` samples a feasible equilibrium tracking Episode on every reset.
Most episodes start at another equilibrium and immediately track one sampled
two-output target; 20% start at a high-temperature equilibrium in
`0.02--0.04 mol/L` and `86--90 degC`. Every training Episode uses the tracking
Benchmark's 225-step horizon.
`disturbance=True` independently samples all three physical disturbances plus
an event at step `50--90` and a `50--90` step duration. Noise, delay, and actuator loss-of-effectiveness
remain the shared optional channel variations.

## Controllers and workflows

The CSTR PI configuration uses `(Kp, Ki, Kd) = (16, 0.4, 0)` from concentration to
feed flow and `(-0.2, -0.005, 0)` from temperature to cooling. The MPC uses a
1-second solve interval, a 5-step prediction horizon, both controlled outputs,
the exact two-action steady-state input, and both action variables. Configuration
selection used randomized, disturbance-enabled training seeds `100--139`; the
fixed formal Benchmark cases `0--19` were kept separate.

```python
import aiogym

env = aiogym.make_env("cstr", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
result = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc},
    seeds=range(20),
)
env.close()
```

The ordinary environment also follows the common
`collect -> train -> save/load -> evaluate/compare` workflow. Short automated
SAC runs verify that path but are not learned-controller performance evidence.

## Current controller comparison

The active local comparison artifacts under `runs/cstr/benchmarks/` contain PI and MPC
only. Each Benchmark uses 20 distinct cases (`0--19`), for 120 evaluated
controller Episodes. Every Episode completed its full horizon with
`unsafe_rate = 0`. Return and IAE aggregate both normalized controlled outputs.

| Benchmark | PI median return | MPC median return | PI median IAE | MPC median IAE | Ranking |
|---|---:|---:|---:|---:|---|
| `tracking` | -0.752299 | -0.723318 | 6.772606 | 5.416064 | MPC, PI |
| `disturbance-rejection` | -0.011040 | -0.000000 | 2.311340 | 0.000007 | MPC, PI |
| `boundary-safety` | -0.782261 | -0.791479 | 5.141274 | 5.111865 | PI, MPC |

The smallest normalized safety margin over all formal runs was `0.014728`.
These are deterministic simulator baselines for the declared parameter set.
They do not establish reactor calibration, robustness to parameter
uncertainty, or learned-policy performance.
