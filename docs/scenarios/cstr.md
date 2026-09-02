# Two-input, two-output CSTR model

The built-in `cstr` Scenario models an exothermic continuous stirred-tank
reactor. You can use it with the common PID, MPC, Dataset, training, and
Benchmark workflows. Both physical commands and both controlled variables are
available to controllers and learned policies.

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

The deterministic training episode starts from that default equilibrium and
immediately tracks `(0.075 mol/L, 72 degC)` for 225 seconds.

## Model parameters and intended use

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

```python
import aiogym

print(aiogym.list_benchmarks("cstr"))
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded two-output equilibrium start that immediately tracks one independently sampled target | 225 steps / 225 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded feed temperature, concentration, coolant temperature, event time, and recovery time | 400 steps / 400 s | unsafe rate, cumulative return |
| `boundary-safety` | normal state is forward pre-run under high feed and low cooling until a case-seeded `86--90 degC` boundary, followed by the default target | 100 steps / 100 s | unsafe rate, cumulative return |

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
`86--90 degC`. Seeds `0--19` select 20 distinct episodes for all three
Benchmarks.

The disturbance horizon leaves at least 40 seconds after the latest possible
event restoration. The boundary protocol retains a 100-second response window.

## Training variation

`randomize=True` samples a feasible equilibrium tracking episode on every reset.
By default every episode starts at another interior equilibrium and immediately
tracks one sampled two-output target. Setting `boundary_probability=p` makes
fraction `p` start from the same high-feed, low-cooling forward pre-run used by
the `86--90 degC` boundary Benchmark. Every training episode
uses the tracking Benchmark's 225-step horizon.
`disturbance=True` independently samples all three physical disturbances plus
an event at step `50--90` and a `50--90` step duration. Noise, delay, and actuator loss-of-effectiveness
remain the shared optional channel variations.

## Quick start and controllers

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

Use the common [Quickstart](../quickstart.md) for Dataset collection and RL
training with `scenario = "cstr"`. Current comparison scores remain in the
generated `runs/cstr/benchmarks/<benchmark>/comparison.json` files.
