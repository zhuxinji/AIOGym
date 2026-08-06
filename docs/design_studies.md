# Parameterised equipment design studies

Design studies screen user-supplied three-tank equipment configurations without
changing the identity of an official benchmark Track. They are intended for
design comparison and virtual commissioning, not safety certification or final
procurement approval.

## Phase-1 scope

The `aiogym.design_spec.v1` topology is fixed to the closed loop
`Tank 3 -> P101 -> Tank 1 -> V12 -> Tank 2 -> V23 -> Tank 3`.

Within that topology, one design can set:

- independent area, height, transmitter span, heat loss, and protection levels
  for all three tanks;
- zero to three heaters, with one fixed H1/H2/H3 slot per tank, installed power,
  and efficiency;
- pump motor power, maximum flow, static head, and shutoff head;
- both gravity-valve coefficients, elevation drops, and passive-overflow
  coefficients;
- target circulation, levels, temperatures, initial condition, and ambient;
- heat-up, overshoot, energy, actuator-margin, temperature-protection, and
  robustness acceptance requirements;
- uncertainty ranges and deterministic Monte Carlo seed.

The compiled design model always exposes six normalized actions:
`[P101, V12, V23, H1, H2, H3]`.

An absent heater has zero capacity and is hard-masked even if a controller sends
full command. The existing four-action `cascade-recirculating` benchmark model
is unchanged, so its controllers, checkpoints, Tracks, and anchors remain
compatible.

## Commands

The repository includes a complete example at
`configs/design/cascade-recirculating-example-v1.json`.

For normal use, create a design with the interactive wizard. It accepts tank
dimensions in mm, pump flow in L/min, power in kW, heat-up time in minutes, and
rates in percent, then writes the SI-unit DesignSpec automatically:

```bash
aiogym design
```

This simplest form automatically creates a timestamped design name, starts the
wizard, saves the DesignSpec, runs the study, and writes the reports. Press
Enter to accept any displayed default. Use the advanced form to configure
transmitter spans, UA values, protection levels, valve/overflow coefficients,
control timing, and uncertainty ranges:

```bash
aiogym design --advanced
```

Use `aiogym design new my-three-tank --run` when a specific design name is
important.

To run without creating a file first, while optionally saving the resolved
inputs for reproducibility:

```bash
aiogym design run --interactive \
  --name my-three-tank \
  --save-spec designs/my-three-tank.json \
  --output runs/design/my-three-tank
```

The lower-level config-first commands remain available for automation and exact
reproduction:

```bash
aiogym design validate \
  configs/design/cascade-recirculating-example-v1.json --json

aiogym design run \
  configs/design/cascade-recirculating-example-v1.json \
  --output runs/design/example

aiogym design sweep \
  configs/design/cascade-recirculating-example-v1.json \
  --parameter pump.max_flow_m3s \
  --values 0.00030,0.00040,0.00050 \
  --output runs/design/pump-sweep

aiogym design report runs/design/example/report.json \
  --output runs/design/example-rerendered
```

`run` and `sweep` never overwrite report files unless `--overwrite` is passed.
Each output directory contains:

```text
report.json     complete machine-readable evidence and resolved DesignSpec
report.html     self-contained human-readable decision report
summary.svg     standalone visual summary
```

## Decision gates

A design receives `PASS` only if every gate passes:

1. **Static engineering:** pump and valve capacity, approximate net heating
   capacity, theoretical heat-up lower bound, and provenance presence.
2. **Model readiness:** model contract, finite dynamics and integration, solver
   configuration, and independent mass/energy balances.
3. **Steady-state feasibility:** target hydraulic and thermal equilibrium,
   finite normalized commands, no cooling requirement, and the requested
   actuator margin.
4. **Hardware interlocks:** P101 dry-run, heater dry-fire and over-temperature
   protection, and absent-heater masks.
5. **Dynamic commissioning:** a declared initial condition is controlled toward
   the target and checked against heat-up time, final tolerance, overshoot,
   energy, protection events, and hard process limits.
6. **Robustness:** the same steady and dynamic checks are repeated over seeded
   pump-capacity, heater-efficiency, and heat-loss samples. The observed pass
   rate must meet the declared threshold.

The steady-state gate represents plant capability. The dynamic gate uses the
explicitly identified `commissioning-pi-v1` baseline and represents performance
with that controller. A steady-state pass plus dynamic failure is therefore a
control/commissioning finding, not automatically proof that the hardware is
incapable.

## Python API

```python
from aiogym.design import run_design_study, write_design_report_bundle

result = run_design_study("my-design.json")
paths = write_design_report_bundle(result, "runs/design/my-design")
print(result["verdict"], paths)
```

The interactive builder is also available as
`aiogym.design.prompt_design_spec(...)` for custom frontends.

## Engineering limitations

Pump motor power alone does not define hydraulic capability. A usable design
must also provide maximum flow, static head, and shutoff head; field work should
replace this reduced curve with measured or manufacturer Q-H data. Valve Cv,
heat loss, heater efficiency, sensor/actuator dynamics, and switch elevations
have the same provenance requirement.

The Phase-1 model assumes constant-area, perfectly mixed water tanks, a fixed
three-tank topology, and no boiling or structural/electrical failure model. Its
reports support option screening and virtual commissioning. They do not replace
equipment sizing calculations, HAZOP, certified electrical protection,
mechanical review, or on-site commissioning.
