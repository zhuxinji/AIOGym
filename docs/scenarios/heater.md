# Fired-heater model

The built-in `heater` Scenario models a refinery fired-heater surrogate with
combustion, firebox heat storage, process-side heat transfer, and flue-oxygen
dynamics. You can use it for simulation, PID/MPC control, Dataset collection,
RL training, and fixed Benchmark comparisons.

The physical interface is:

- state `[firebox_temperature, outlet_temperature, flue_oxygen]` in
  `degC`, `degC`, and `%`;
- controlled output `[flue_oxygen, outlet_temperature]`;
- action `[air_damper, fuel_valve]`, with both commands normalized to `[0, 1]`;
- disturbances `feed_temperature`, `ambient_temperature`, `feed_flow`, and
  `fuel_heating_value_factor`.

The action order is always air at index 0 and fuel at index 1. The model
converts them to mass flows using `maximum_air_flow` and
`maximum_fuel_flow`. Lean combustion determines the equilibrium flue-oxygen
concentration from excess air. Fuel energy, process heat demand, and flue-stack
loss determine the firebox and outlet-temperature balances.

The policy observation contains five normalized values: all three states,
followed by the oxygen and outlet-temperature references. Disturbances are not
included in the policy observation. The controller interval is 1 second and
the plant is integrated internally with a maximum 0.05-second step.

At the default reference `(3%, 370 degC)`, the exact steady-state inverse gives
state `(758.714286, 370, 3)` and action `(0.383882, 0.758275)`. The default
episode starts from this model-consistent equilibrium and immediately tracks
`(4%, 366 degC)` for 300 seconds.

For a requested oxygen and outlet-temperature pair, the inverse first solves
the process heat balance for firebox temperature. It then derives the lean
air/fuel ratio from oxygen and solves the firebox balance for fuel flow. It
returns no operating point if the required state or either physical action is
infeasible; it never clips a steady-state solution.

## Model parameters and intended use

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

```python
import aiogym

print(aiogym.list_benchmarks("heater"))
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded feasible equilibrium start that immediately tracks one independently sampled target | 300 steps / 300 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded feed temperature, ambient temperature, feed flow, fuel quality, event time, and recovery time | 700 steps / 700 s | unsafe rate, cumulative return |
| `boundary-safety` | two-stage forward pre-run first heats the normal process to `382--385 degC`, then reduces excess air until oxygen reaches `1.30--1.70%`, followed by the default target | 450 steps / 450 s | unsafe rate, cumulative return |

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
Boundary cases use a two-stage forward pre-run to reach oxygen in
`1.30--1.70%` after heating the outlet to `382--385 degC`. Seeds `0--19`
select 20 distinct episodes for all three Benchmarks.

Disturbance rejection uses a 700-second response window. Boundary safety uses a
450-second window after the forward pre-run reaches the selected limit.

## Training variation

`randomize=True` samples a feasible equilibrium tracking episode on every reset.
By default every episode starts at one sampled interior equilibrium and
immediately tracks one random target. Setting `boundary_probability=p` makes
fraction `p` start from the same two-stage forward pre-run used by the
low-oxygen/high-temperature boundary Benchmark. Every training episode uses
the tracking Benchmark's 300-step horizon.

`disturbance=True` independently samples all four physical disturbances plus
an event at step `70--120` and a `100--min(160, 240-start)` step duration (restored by step 240). Noise, delay, and
actuator loss-of-effectiveness remain the shared optional channel variations.

## Quick start and controllers

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

Use the [task guide](../workflows.md) for Dataset collection and RL
training with `scenario = "heater"`. Current scores and safety metrics remain
in `runs/heater/benchmarks/<benchmark>/comparison.json`.
