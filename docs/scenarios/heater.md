# Fired-heater model

The built-in `heater` Scenario models a refinery fired-heater surrogate with
combustion, firebox heat storage, process-side heat transfer, and flue-oxygen
dynamics through the common Scenario, Reward, Benchmark, controller, Dataset,
and training workflow contracts.

The physical interface is:

- state `[firebox_temperature, outlet_temperature, flue_oxygen]` in
  `degC`, `degC`, and `%`;
- controlled output `[flue_oxygen, outlet_temperature]`;
- action `[air_damper, fuel_valve]`, with both commands normalized to `[0, 1]`;
- disturbances `feed_temperature`, `ambient_temperature`, `feed_flow`, and
  `fuel_heating_value_factor`.

The action order is part of the public contract: air is index 0 and fuel is
index 1. The model converts them to mass flows using `maximum_air_flow` and
`maximum_fuel_flow`. Lean combustion determines the equilibrium flue-oxygen
concentration from excess air. Fuel energy, process heat demand, and flue-stack
loss determine the firebox and outlet-temperature balances.

The policy observation contains five normalized values: all three states,
followed by the oxygen and outlet-temperature references. Disturbances are not
included in the policy observation. The controller interval is 1 second and
the plant is integrated internally with a maximum 0.05-second step.

At the default reference `(3%, 370 degC)`, the exact steady-state inverse gives
state `(758.714286, 370, 3)` and action `(0.383882, 0.758275)`. The default
Episode starts from this model-consistent equilibrium and immediately tracks
`(4%, 366 degC)` for 300 seconds.

For a requested oxygen and outlet-temperature pair, the inverse first solves
the process heat balance for firebox temperature. It then derives the lean
air/fuel ratio from oxygen and solves the firebox balance for fuel flow. It
returns no operating point if the required state or either physical action is
infeasible; it never clips a steady-state solution.

## Parameters and evidence boundary

The main executable defaults are:

| Parameter | Default | Unit |
|---|---:|---|
| `maximum_fuel_flow` | 1.0 | kg/s |
| `fuel_lower_heating_value` | 46e6 | J/kg |
| `stoichiometric_air_fuel_ratio` | 17.2 | kg_air/kg_fuel |
| `maximum_air_flow` | 40 | kg/s |
| `flue_gas_heat_capacity` | 1400 | J/(kg*K) |
| `heat_transfer_coefficient` | 42000 | W/K |
| `firebox_heat_capacity` | 3.5e6 | J/K |
| `process_heat_capacity` | 7e6 | J/K |
| `nominal_feed_flow` | 88 | kg/s |
| `feed_specific_heat_capacity` | 2300 | J/(kg*K) |
| `oxygen_time_constant` | 20 | s |
| `feed_temperature` | 280 | degC |
| `ambient_temperature` | 20 | degC |
| `outlet_temperature_trip` | 415 | degC |
| `minimum_safe_oxygen` | 1.2 | % |

These values and lumped equations define a normalized fired-heater surrogate.
They are not calibrated to a named furnace or validated against plant
measurements. In particular, modeled outlet temperature is not a tube-skin
measurement, and flue oxygen is not a complete combustion or emissions model.
This Scenario supports reproducible simulation and controller comparison, not
process deployment or physical-fidelity claims.

Safety termination uses the modeled physical state limits plus the explicit
`415 degC` outlet-temperature and `1.2%` oxygen trips. The narrower reference
ranges are operating targets, not additional termination rules.

## Benchmarks

```bash
aiogym list benchmarks --scenario heater
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded feasible equilibrium start that immediately tracks one independently sampled target | 300 steps / 300 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded feed temperature, ambient temperature, feed flow, fuel quality, event time, and recovery time | 700 steps / 700 s | unsafe rate, cumulative return |
| `boundary-safety` | case-seeded exact equilibrium start in `(1.30--1.70%, 382--390 degC)`, followed by the default target | 450 steps / 450 s | unsafe rate, cumulative return |

Each tracking seed samples oxygen in `2.5--5.0%` and outlet temperature in
`364--372 degC`. The target differs from the initial output by at least
`0.4 percentage points` and `2 degC`, and tracking begins on the first control
step. The model derives the corresponding states and actions, requires
both commands to remain in `0.05--0.95`, and verifies a zero model derivative
before accepting the case. Identical seeds therefore give every compared
policy identical operating conditions.

Disturbance cases begin at step `140--240` and last `200--320` steps. They
sample feed temperature in `255--305 degC`, ambient temperature in
`5--35 degC`, feed flow in `75--105 kg/s`, and fuel-quality factor in
`0.85--1.15`. A case is accepted only when the default `(3%, 370 degC)` target
has an exact disturbed equilibrium with both actions in `0.05--0.95`.
Boundary cases sample exact equilibria with oxygen in `1.30--1.70%` and outlet
temperature in `382--390 degC`. Seeds `0--19` resolve to 20 distinct Episodes
for all three Benchmarks.

Disturbance rejection remains 700 seconds because PI did not satisfy the strict
settling band before the existing horizon and MPC required up to 644 seconds.
Boundary cases settled within 404 seconds, so 450 seconds preserves a measured
buffer without retaining the unused final 50 seconds.

## Training variation

`randomize=True` samples a feasible equilibrium tracking Episode on every reset.
Most episodes start at one sampled equilibrium and immediately track one random
target; 20% start near the low-oxygen/high-temperature boundary at
`1.5--1.7%` and `382--388 degC`. Every training Episode uses the tracking
Benchmark's 300-step horizon.

`disturbance=True` independently samples all four physical disturbances plus
an event at step `70--120` and a `100--160` step duration. Noise, delay, and
actuator loss-of-effectiveness remain the shared optional channel variations.

## Controllers and workflows

The heater PI configuration uses `(Kp, Ki, Kd) = (0.09, 0.001, 0)` from flue oxygen
to the air damper and `(0.025, 0.0003, 0)` from outlet temperature to the fuel
valve. The other two gain-matrix entries are zero. The MPC solves every 2
seconds with a 20-step prediction horizon, model-derived steady input, and both
physical actions. A steady-input weight of 50 per action keeps the
successive-linearization controller from making unsafe low-oxygen recovery
moves.

```python
import aiogym

env = aiogym.make_env("heater", benchmark="tracking")
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

The active local comparison artifacts under `runs/heater/benchmarks/` contain PI and MPC
only. Each Benchmark uses 20 distinct cases (`0--19`), for 120 evaluated
controller Episodes. Every Episode completed its full horizon with
`unsafe_rate = 0`.

| Benchmark | PI median return | MPC median return | PI median IAE | MPC median IAE | Ranking |
|---|---:|---:|---:|---:|---|
| `tracking` | -4.312733 | -4.053086 | 40.074490 | 28.573402 | MPC, PI |
| `disturbance-rejection` | -20.806917 | -6.943544 | 117.150685 | 54.588201 | MPC, PI |
| `boundary-safety` | -44.400929 | -49.394676 | 132.159871 | 87.801897 | PI, MPC |

The smallest normalized safety margin over all formal runs was `0.017217`.
MPC has lower absolute-error integral in the boundary recovery, while PI ranks
first on the declared cumulative-return metric. These deterministic simulator
baselines do not establish furnace calibration, robustness outside the tested
conditions, or learned-policy performance.
