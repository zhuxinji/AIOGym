# PDF-derived recirculating heated-tank cascade

`cascade-recirculating` is the closed-loop retrofit scenario derived from the
design proposal `三级水箱加热系统.pdf`. It is separate from the historical
open `cascade` benchmark because it has a different mass-flow topology, four
instead of seven actuators, one instead of three heaters, passive overflow
returns, and no product-flow economics.

## Use the scenario

```python
import aiogym

env = aiogym.make_env(
    "cascade-recirculating",
    task="commissioning",
    objective="tracking",
    seed=7,
)
```

Available named tasks are:

- `commissioning`: deterministic hold at the model-consistent equilibrium;
- `temperature-step`: a 30 to 35 degC Tank 1 target move and return;
- `disturbance-rejection`: scheduled P101 capacity, H1 efficiency, and ambient
  changes;
- `safety-recovery`: recoverable passive-overflow and interlock state;

The scenario supports tracking, KPI, robustness, and safety objectives.
`economic` is rejected because the selected design has no normal product stream
or source-backed production-value model.

The four formal tasks use the same `RecirculatingCascadeModel` equations and
the same default numerical parameter set. They differ in initial state,
setpoint/disturbance schedule, control period, and episode length. None of these
task profiles declares `model_params`. Their `auto_events: false` setting disables
generic automatically generated events; it does not disable the model's
differential equations or a task's deterministic event schedule.

## Benchmark workflows

Specify a task when developing or comparing controllers on one fixed
experiment. Omitting `--objective` uses the selected task's default:

```bash
aiogym benchmark suite cascade-recirculating temperature-step \
  --controllers pid,mpc \
  --episodes 3 \
  --save-rollouts
```

This command compares both controllers only on
`(cascade-recirculating, temperature-step, tracking)`.

Use the built-in suite for a formal multi-task comparison:

```bash
aiogym benchmark suite cascade-recirculating \
  --episodes 3
```

The suite expands `commissioning`, `temperature-step`,
`disturbance-rejection`, and `safety-recovery` for PID and MPC. Each task keeps
its own default objective and ranking identity; scores from different
`(scenario, task, objective)` tuples are not combined. Use `--controllers pid`
or `--controllers mpc` to filter the built-in matrix during development.

## Public contract

```text
x = [h1, T1, h2, T2, h3, T3]
u = [pump_P101, valve_V12, valve_V23, heater_H1]
y = [h1, h2, h3, T1, T2, T3]
```

All actions are normalized to `[0, 1]`. Their conversion to VFD frequency,
valve travel, current, voltage, or SSR duty cycle is not identified.

The normal loop is:

```text
Tank 3 -> P101 -> Tank 1 -> V12 -> Tank 2 -> V23 -> Tank 3
```

Tank 1 and Tank 2 also have passive overflow branches returning directly to
Tank 3. Only Tank 1 receives heat from the declared 2 kW H1.

## Model equations

Interstage valve flows use one-way Torricelli relations proportional to valve
command and the square root of the upstream level. P101 uses a normalized
linear capacity relation. The three mass balances close exactly because every
normal or overflow stream has both a source and a destination.

Thermal dynamics assume perfectly mixed, constant-property water. Each tank
loses heat through a provisional lumped UA. P101, V12, V23, and both overflow
branches transport source-stream enthalpy. Tank 1 heater power is multiplied by
the declared heat-transfer efficiency and is removed when an H1 interlock is
active.

The nominal benchmark equilibrium is:

```text
circulation flow = 8.0e-5 m3/s
x* = [0.40, 30.0, 0.40, 28.9329919, 0.40, 27.5756694]
u* = [0.05, 0.0486504, 0.0486504, 0.6059299]
ideal electrical power = 1.236859838 kW
```

The provisional equivalent areas are `[0.04, 0.04, 0.15] m2`, so the 0.40 m
nominal level represents 16 L, 16 L, and 60 L, or 92 L total. These values are
generated from a range-derived interpretation of the PDF and are not measured
commissioning values.

## Protection layers

- Tank 1/Tank 2 overflow returns are physical flow branches and do not by
  themselves terminate an episode.
- Tank 3 low level disables P101.
- Tank 1 low level, Tank 1/Tank 2 high level, or Tank 1 over-temperature
  disables H1 and removes actual H1 electrical power from the energy metric.
- Negative level, a level above the provisional hard height, or a temperature
  at the hard single-phase boundary terminates with a structured reason.

These are executable software semantics, not certification of the real RCD,
contactors, switches, wiring, or protection setpoints.

## Evidence boundary

The PDF is a design proposal rather than an as-built record. It contains an old
three-heater/separate-reservoir concept as well as the later one-heater/Tank 3
reservoir retrofit. The model follows the repeated retrofit narrative on pages
2-7. H1's 2 kW rating is design-specified. The 0.4-0.5 m tank-height and
15-20 L small-tank ranges on page 1, together with the 60-80 L reservoir range,
support the provisional 16/16/60 L geometry; they do not establish as-built
dimensions.

P101's 500 W benchmark is inside the PDF's 0.37-0.75 kW range but is not a
nameplate value. Flow coefficients, P101 flow, UA values, overflow capacity,
interlock thresholds, actuator dynamics, sensor dynamics, and the nominal
operating point remain benchmark assumptions or unverified legacy values. Do
not infer field accuracy, RMSE, equipment capacity, or safety ratings from the
scenario.

See the complete
[accuracy and authenticity report](../reports/cascade-recirculating-accuracy-authenticity.md)
and its
[machine-readable evidence](../reports/cascade-recirculating-validation-results.json).
