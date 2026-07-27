# Heated-tank cascade

The `cascade` scenario models three constant-area, perfectly mixed heated-water
tanks connected by one-way free drains. It is intended as a reproducible control
benchmark, not as a validated digital twin of a specific installation.

## Public vectors

The state remains interleaved:

```text
x = [h0, T0, h1, T1, h2, T2]
```

The normalized actuator vector is:

```text
u = [feed_pump, outlet_valve_0, outlet_valve_1, outlet_valve_2,
     heater_0, heater_1, heater_2]
```

Controlled outputs are reordered without clipping:

```text
y = [h0, h1, h2, T0, T1, T2]
```

Only UI-oriented `display_outputs()` clips negative numeric liquid levels for
presentation. Controllers, KPI calculations, constraints, and CasADi receive the
raw state.

## Physical and safety semantics

- Numeric and CasADi actions are defensively limited to `[0, 1]` inside the model.
- Empty tanks have zero gravity-valve flow; an empty third tank also disables
  `extra_outflow`.
- A heater transfers no power to the liquid below `heater_min_level=0.05 m` or
  at/above `temperature_trip=92 degC`.
- A negative level, level above `height_max=0.8 m`, or temperature at/above
  `temperature_hard_limit=120 degC` terminates the environment transition with a
  structured reason. States are not silently projected back into range.
- `heater_efficiency` is the fraction of electrical heater power transferred to
  the liquid and is restricted to `[0.4, 1.0]` in the benchmark disturbance model.

The model assumes constant liquid properties, constant tank area, perfect
mixing, instantaneous actuators, lumped heat loss, and no pipe transport delay,
wall heat capacity, downstream backpressure, boiling, evaporation, or phase
change.

## Throughput targets

Direct environments use `product_flow_sp=min_product_flow=0`, so they have no
required or economically valued throughput. Use the task-owned continuous
benchmark explicitly:

```python
import aiogym

env = aiogym.make_env(
    "cascade",
    objective="economic",
    task="continuous-benchmark",
)
```

`continuous-benchmark` declares `product_flow_sp=min_product_flow=4.0e-4 m3/s`.
The same operation configuration is applied to the environment model and the
controller model. Economic scoring includes actual product flow, electrical
energy, temperature/level violations, and normalized flow shortfall. The
throughput value is an assumed benchmark, not a measured plant capacity.

There is no separate batch operation mode. Zero throughput targets disable
production value and shortfall accounting without changing the physical
equations: pump and valve actions can still create physical flow. This keeps
task economics independent from the actuator-driven hydraulic model.

At the default setpoint and nominal disturbances, the analytic requirement is:

```text
u* = [0.25, 0.2293403, 0.2293403, 0.2293403,
      0.3787556, 0.2924000, 0.2990667]
ideal electrical power = 87.695 kW
```

The helper `steady_state_requirements()` reports these commands, component heat
loads and powers, feasibility, and per-actuator infeasibility reasons.

## Formal control tasks

Four additional tasks use the same assumed `4.0e-4 m3/s` continuous operating
point while keeping their non-economic objectives separate:

- `commissioning` holds the nominal model-consistent equilibrium;
- `temperature-step` moves the first-stage target from 35 to 40 degC at step
  120 and returns it at step 1200;
- `disturbance-rejection` schedules feed-pump capacity, common heater
  efficiency, and ambient-temperature changes;
- `safety-recovery` starts below the Tank 0 heater minimum level and above the
  Tank 2 heater temperature trip, but inside the hard termination limits.

Run all four with their task-owned default objectives:

```bash
aiogym benchmark suite cascade-control --episodes 3
```

Each task binds PID to the `cascade-control-benchmark` controller profile,
which initializes the six feedback loops and fixed third outlet valve at the
analytic steady input. Single-task and suite runs therefore use the same PID
parameters. MPC obtains the same feasible input through
`tracking_steady_state_action()`.

These task names mirror the recirculating task family, but their numerical
states, actuator contract, disturbance meaning, and protection conditions are
specific to the open cascade. In particular, they do not add passive return
overflows, a Tank 3 pump interlock, or a single-H1 topology to this model.

## Migration notes

- Code that previously used `heater_efficiency > 1` must use a capacity change
  through `heater_max`/plant regime instead; efficiency is now capped at 1.
- `controlled_output()` no longer hides negative liquid levels. UI code should
  use `display_outputs()` if nonnegative presentation is required.
- Level and temperature hard boundaries now terminate regardless of
  `terminate_on_runaway`; callers must handle `terminated=True`,
  `termination_reason`, and `safety_events`.
- Economic comparisons requiring production must select
  `task="continuous-benchmark"`. Zero-target results are not
  continuous-production rankings.

## Evidence limits

All geometry, flow, heat-loss, pump, heater, and safety values are marked
`legacy-unverified` or `assumed-benchmark` in the parameter profile. No plant
identification dataset, equipment datasheet, or independent validation series is
bundled. The model can therefore be internally verified as software and as an
implementation of its equations, but real-device RMSE, MAE, bias, and uncertainty
remain unknown.
