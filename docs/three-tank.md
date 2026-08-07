# Unified three-tank scenario

`three_tank` is one physical-system family with two topology strategies and
three built-in plants.

| Plant | Topology | Actions | Default condition | Tasks |
| --- | --- | ---: | --- | --- |
| `open-cascade-v1` | `open_cascade` | 7 | `continuous-benchmark` | regulation, economic |
| `recirculating-h1-v1` | `recirculating_loop` | 4 | `commissioning` | regulation |
| `lab-three-tank-v1` | `recirculating_loop` | 6 | `commissioning` | regulation |

All use state order `h1, T1, h2, T2, h3, T3` and controlled-output order
`level_1, level_2, level_3, temperature_1, temperature_2, temperature_3`.

## Installed actuator layout

Parameterized laboratory plants keep six private physical slots, but their
public action contains only installed equipment:

```text
H1 only       -> pump_P101, valve_V12, valve_V23, heater_H1             (4)
H1 and H3     -> pump_P101, valve_V12, valve_V23, heater_H1, heater_H3  (5)
H1, H2 and H3 -> pump_P101, valve_V12, valve_V23, heater_H1, H2, H3     (6)
```

The order is stable and heater declarations must have unique tank numbers and
positive power. PID terms are name-bound and filtered to installed equipment;
MPC derives its input dimension from the public schema. A layout change alters
`plant_hash` and `interface_hash`; old checkpoints cannot cross that boundary.

## Physical implementation

```text
ThreeTankPhysicsKernel
├── OpenCascadeTopology       -> open-cascade-v1
└── RecirculatingTopology     -> recirculating-h1-v1
    └── ThreeTankDesignModel  -> parameterized laboratory plants
```

The common kernel owns state/output structure and balance assembly. Topologies
own flow connections and mixing. Equipment compilation owns installed heaters,
interlocks, parameters, and public-to-internal action expansion.

## Economic task

Only `open-cascade-v1` exposes product flow. The v2 objective is:

```text
product_value_rate = product_value_per_m3 * product_flow_m3s
energy_cost_rate   = electricity_price_per_kwh * power_kw / 3600
reward             = (product_value_rate - energy_cost_rate) * control_dt
```

The coefficients are part of `TaskSpec.objective_config` and therefore
`task_hash`:

```text
value_unit                = normalized_value
product_value_per_m3      = 100000.0
electricity_price_per_kwh = 0.7
```

`normalized_value` is a benchmark value unit, not SGD, USD, or another claimed
real-world price. Evaluation reports net value, product value, positive energy
cost, production volume in m³, and energy in kWh.
