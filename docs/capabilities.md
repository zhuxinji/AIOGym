# Built-in capability matrix

This page distinguishes model-level interface support from formal task and
benchmark evidence. The executable registries and JSON declarations are the
source of truth.

## How to read the matrix

- **Model objectives** come from each model's `supported_objectives`. They mean
  the environment can construct those scoring semantics; they do not prove that
  a formal task or tuned baseline exists.
- **Benchmark objectives** come from each model's `benchmark_objectives`. Suite
  groups use this narrower evidence declaration, so technical scoring support is
  not mistaken for a validated built-in benchmark.
- **Bundled tasks** are the 13 JSON task specs returned by
  `aiogym.list_tasks()` at the stated revision.
- **Task objectives** are narrower declarations enforced by those task specs.
- **Action modes** come from the actuator contract and optional
  `supervisory_layout`. `setpoint` means supervisory setpoint control through an
  inner PID.

Abbreviations: T = tracking, E = economic, K = KPI, R = robustness, S = safety.

## Scenario support

| Scenario | Bundled task specs | Task objectives | Model objectives | Action modes |
| --- | --- | --- | --- | --- |
| `cascade` | `continuous-benchmark`; `commissioning`; `temperature-step`; `disturbance-rejection`; `safety-recovery` | E for the economic task; T, K, R, S for control tasks; defaults vary | T, E, K, R, S | actuator; setpoint (3) |
| `cascade-recirculating` | `commissioning`; `temperature-step`; `disturbance-rejection`; `safety-recovery` | T, K, R, S; defaults vary by task | T, K, R, S; E intentionally unsupported | actuator; setpoint (1) |
| `quadruple` | `minimum-phase`; `nonminimum-phase`; `zero-boundary-stress`; `disturbance-rejection` | T/K/S for phase tasks; T/K/R/S for stress and disturbance tasks | T, E, K, R, S | actuator; setpoint (2) |
| `cstr` | none; direct environment and suites only | — | T, E, K, R, S | actuator; setpoint (2) |
| `hvac` | none; direct environment and suites only | — | T, E, K, R, S | actuator; setpoint (2) |
| `extraction` | none; direct environment and suites only | — | T, E, K, R, S | actuator only |
| `heater` | none; direct environment and suites only | — | T, E, K, R, S | actuator; setpoint (2) |
| `crystallization` | none; direct environment and suites only | — | T, E, K, R, S | actuator only |

The number in parentheses is the supervisory action dimension. It is not the
underlying actuator dimension.

### Task defaults and status

| Task ID | Status | Default objective | Allowed objectives |
| --- | --- | --- | --- |
| `cascade/continuous-benchmark` | assumed benchmark | economic | economic |
| `cascade/commissioning` | assumed benchmark | tracking | tracking, KPI, robustness, safety |
| `cascade/temperature-step` | benchmark designed | tracking | tracking, KPI, robustness, safety |
| `cascade/disturbance-rejection` | benchmark designed | robustness | tracking, KPI, robustness, safety |
| `cascade/safety-recovery` | benchmark designed | safety | tracking, KPI, robustness, safety |
| `cascade-recirculating/commissioning` | design-derived benchmark | tracking | tracking, KPI, robustness, safety |
| `cascade-recirculating/temperature-step` | benchmark designed | tracking | tracking, KPI, robustness, safety |
| `cascade-recirculating/disturbance-rejection` | benchmark designed | robustness | tracking, KPI, robustness, safety |
| `cascade-recirculating/safety-recovery` | benchmark designed | safety | tracking, KPI, robustness, safety |
| `quadruple/minimum-phase` | reference derived | tracking | tracking, KPI, safety |
| `quadruple/nonminimum-phase` | reference derived | tracking | tracking, KPI, safety |
| `quadruple/zero-boundary-stress` | reference-inspired stress | tracking | tracking, KPI, robustness, safety |
| `quadruple/disturbance-rejection` | benchmark designed | robustness | tracking, KPI, robustness, safety |

`legacy-default` is not a current task. References to it in dated audit files
describe historical snapshot behavior.

## Controller entry points

The controller registry exposes six IDs:

| Controller ID | Kind | Action mode | Caller-supplied artifact | Current availability |
| --- | --- | --- | --- | --- |
| `pid` | built-in decentralized PID | actuator | no | constructs for all 8 scenarios; explicit scenario presets for all 8 |
| `mpc` | built-in linear MPC | actuator | no | constructs for all 8 scenarios; explicit scenario overrides for 7, with generic defaults for `extraction` |
| `oracle` | built-in nonlinear MPC | actuator | no | constructs for all 8 scenarios; scenario/task-specific profiles exist only where declared in its config |
| `policy` | Python callable adapter | actuator or setpoint | policy object | all scenarios in actuator mode; setpoint only where listed above |
| `sb3` | Stable-Baselines3 adapter | actuator or setpoint | model object or path | all scenarios in actuator mode; setpoint only where listed above |
| `onnx` | ONNX Runtime adapter | actuator or setpoint | ONNX path | all scenarios in actuator mode; setpoint only where listed above |

“Constructs” means the factory accepts the scenario and creates the controller.
It does not mean every controller has been tuned or performance-validated on
every model objective. Use a named suite or explicit benchmark configuration to
make performance comparisons.

## Evidence levels

| Level | What it establishes | Where to verify |
| --- | --- | --- |
| Interface support | A model, environment, action mode, or controller can be constructed | model/controller registries and contract tests |
| Bundled specification | A versioned task or suite declaration exists | `aiogym/models/tasks/builtin/` and `aiogym/evaluation/suites/` |
| Executed benchmark | A controller completed resolved cases for stated seeds | benchmark artifact metadata and result rows |
| Objective met | Declared task acceptance thresholds passed | artifact `objective_status` and acceptance metrics |
| Physical validation | Equations or predictions were checked against stated evidence | scenario documentation and dated validation reports |

Do not infer a higher evidence level from a lower one. In particular, model
objective support and controller construction are not claims of tuned control
quality or real-plant accuracy.

## Machine-readable discovery

```python
import aiogym

print(aiogym.list_scenarios())
print(aiogym.list_tasks())
print(aiogym.list_suites())
print(aiogym.list_controllers())

task = aiogym.load_task_profile(
    "quadruple/minimum-phase"
)
print(task["supported_objectives"])
```

The four `list_*` functions are the supported discovery API.
See the [concepts page](concepts.md) for terminology and the
[public API guide](public_api.md) for execution examples.
