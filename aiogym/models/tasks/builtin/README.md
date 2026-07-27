# Task profiles

Task profiles declare scenario-specific timing and reserve stable sections for
initial conditions, setpoints, disturbances, constraints, operation, objective
options, controller defaults, observation semantics, and acceptance rules.
They are opt-in, versioned inputs rather than hidden environment defaults.
Schema and discovery APIs are provided by `aiogym.models.tasks`; evaluation
only consumes resolved tasks and evaluates declared acceptance thresholds.

The environment field `auto_events` controls generic automatically generated
within-episode events. It does not make the physical model static: state always
advances through the scenario's differential equations. A reproducible task can
therefore set `auto_events: false` while declaring deterministic changes under
`setpoints.schedule` or `disturbances`.

`auto_events` is the only accepted field for automatic event generation.

Tasks may declare `default_objective` and `supported_objectives`. Objective
resolution uses runtime/API override, case config, suite config, then the task
default. A selected objective outside `supported_objectives` is rejected rather
than silently replaced.

Direct environments keep their own documented interactive defaults. Named task
profiles represent reproducible experiments and should record their benchmark
source explicitly.

The quadruple tasks own their normalized tracking weights and learned-policy
observation contract. Runtime and CLI values are optional overrides; omitted
values resolve from the task before falling back to the global tracking
defaults `tracking_q_y=0.7` and `tracking_r_move=0.3`.

The optional `controllers` mapping binds task-specific default profiles to
controller IDs. Every benchmark entry point resolves these defaults through
`BenchmarkCase`; explicit CLI, API, or suite controller configuration overrides
the task defaults. A suite therefore selects cases but does not silently change
the controller used by an equivalent single-task run.

`quadruple/minimum-phase` is the first reference-derived task. It uses
the minimum-phase Johansson configuration, an exact nonlinear equilibrium, and
two deterministic lower-tank setpoint moves.

The optional `operation` section declares task-level production semantics
through `product_flow_sp` and optional `min_product_flow`. Both values are
non-negative, and the minimum may not exceed the target. Zero values disable
production value and shortfall accounting; no separate batch mode is needed.
These values configure the environment model and controller model through the
same path and are not copied into physical model parameters.

The optional `acceptance.metrics` mapping defines result thresholds separately
from execution success. Each metric may provide `min`, `max`, or both. Results
record `objective_status=met|not-met`; tasks without thresholds report
`objective_status=not-defined`.

For `cascade`, direct operation uses zero throughput targets. The
`continuous-benchmark` task uses an assumed `4.0e-4 m3/s` throughput target and
minimum flow for economic benchmarking. This value is a reproducible benchmark
assumption, not a device rating or plant-validated operating limit.

The `commissioning`, `temperature-step`, `disturbance-rejection`, and
`safety-recovery` open-cascade tasks reuse that throughput as a common
model-consistent operating point. They are separate tracking, robustness, and
safety experiments and deliberately exclude the economic objective. Their
values are adapted to the seven-action, three-heater open cascade; they are not
copies of the recirculating rig's four-action configuration or protection
topology.

`cascade-recirculating` is a separate closed-loop hardware-design scenario. Its
`commissioning`, `temperature-step`, `disturbance-rejection`, and
`safety-recovery` tasks support tracking/KPI/robustness/safety objectives only.
They deliberately do not inherit `cascade` production economics because the PDF
rig recirculates Tank 3 through P101 and declares no product stream.
