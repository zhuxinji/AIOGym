# Exothermic CSTR (`cstr`)

Two-state exothermic reactor with feed dilution and cooling-jacket action.

## Recommended Use

economic control, nonlinear tracking, and NMPC/oracle comparisons

## Dynamics And Assumptions

A two-state exothermic reactor tracks concentration and temperature with feed dilution, Arrhenius reaction rate, heat release, and cooling action.

- Physical constants are simplified and partly lumped for a stable benchmark operating range.
- Cooling is represented as a normalized actuator rather than detailed jacket hydraulics.
- Economic value is production-oriented and should be compared only within a
  declared economic Goal and Track.

The executable source of truth is the model implementation under `aiogym.models.scenarios`.

## Physical Metadata Status

- Parameter status: `legacy-unverified`
- Fidelity: `legacy-benchmark`
- Time unit: `s`
- Solver: `rk4` with maximum step `0.01`

## State Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| Ca | mol/L | [0, 1.5] |
| T | degC | [0, 200] |

## Action Vector

| Name | Kind | Index | Bounds |
| --- | --- | --- | --- |
| feed_pump | pump | 0 | [0, 1] |
| cooling | heater | 1 | [0, 1] |

## Controlled Output Vector

| Name | Unit | Bounds |
| --- | --- | --- |
| reactor_temperature | degC | [45, 90] |

## Setpoint Vector

| Name | Output | Unit | Bounds |
| --- | --- | --- | --- |
| reactor_temperature | reactor_temperature | degC | [45, 90] |

## Disturbances

| Name | Event | Unit | Bounds | Default |
| --- | --- | --- | --- | --- |
| t_cold | cold_inlet_step | degC | [0, 60] | 20 |
| Caf | feed_concentration_step | mol/L | [0, 5] | 1 |
| Tcool | coolant_temperature_step | degC | [-20, 50] | 10 |
| t_amb | ambient_step | degC | [0, 45] | 20 |

## Parameters

| Name | Value | Unit | Bounds |
| --- | --- | --- | --- |
| Caf | 1 | mol/L | [0, 5] |
| Dmax | 0.02 | 1/s | [0, 1] |
| EaR | 7000 | K | [0, 50000] |
| Hr | 120 | degC/(mol/L) | [0, 1000] |
| Tcool | 10 | degC | [-20, 50] |
| Uc | 0.05 | 1/s | [0, 10] |
| cool_max | 80000 | W | [0, 500000] |
| feed_power_max | 1200 | W | [0, 10000] |
| h_floor | 0.001 | m | [1e-06, 0.1] |
| k0 | 1e+08 | 1/s | [0, 1e+12] |
| t_amb | 20 | degC | [0, 45] |
| t_cold | 20 | degC | [0, 60] |

## Safety And Operating Bounds

| Constraint | States | Bounds |
| --- | --- | --- |
| concentration_nonnegative | Ca | [0, none] |
| temperature_trip | T | [none, 92] |
| temperature_hard_cap | T | [none, 200] |

## Benchmark Semantics

- State vector length: 2
- Action vector length: 2
- Controlled output vector length: 1
- Setpoint vector length: 1
- Dynamics disturbances: [t_cold, Caf, Tcool, t_amb]
- Micro integration step: 0.01 s
- Energy is scored: False

### Plant-Regime Randomization

| Parameter | Multiplier Bounds |
| --- | --- |
| Hr | [0.85, 1.2] |
| Uc | [0.5, 1.6] |
| k0 | [0.55, 1.7] |

### Economic Configuration

| Key | Value |
| --- | --- |
| level_band | [] |
| temp_band | [[none, 88]] |
| value | production |
| w_energy | 0.7 |
| w_value | 1575 |
| w_viol | 14 |

### Supervisory Layout

| Entry | Meaning |
| --- | --- |
| 0 | [y_sp, 0, 45, 90] |
| 1 | [mv, 0, 0.3, 1] |
