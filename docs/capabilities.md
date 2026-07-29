# Capabilities

## Scenarios

AIO-Gym includes cascade, cascade-recirculating, crystallization, CSTR,
extraction, fired heater, HVAC, and quadruple-tank models. Use
`aiogym.list_scenarios()` for the authoritative runtime registry.

## Controllers

| Controller | Role | Typical action mode |
|---|---|---|
| PID | fixed classical baseline | actuator |
| MPC | successive-linearization baseline | actuator |
| Oracle | nonlinear MPC baseline | actuator |
| SB3 | learned policy adapter | declared by Track |
| ONNX | portable learned policy adapter | declared by Track |

Construction support is not a performance claim. A benchmark claim requires a
named Track or explicit Case, seeds, Goal, RewardSpec, safety result, and
recorded provenance.

## Benchmark resources

| Resource | Meaning | Discovery |
|---|---|---|
| Scenario | physical process | `list_scenarios()` |
| Case | reproducible experiment | `list_cases()` |
| RewardSpec | learning scalar | `list_reward_specs()` |
| Track | official benchmark contract | `list_tracks()` |
| Controller | executable policy adapter | `list_controllers()` |

All evaluations emit the full Scorecard. Goal chooses ranking semantics; it
does not suppress safety, service, economic, or controller diagnostics.
