# Three-tank unification report (historical 0.3 snapshot)

> This report records the 0.3 unification. See `docs/three-tank.md` and
> `docs/refactor/v0.4-hardening-report.md` for current behavior.

Date: 2026-08-06

The implementation removes the two old scenario packages, adds three
PlantConfig v2 declarations, and registers only `quadruple` and `three_tank`
as public scenarios. Validation below was run from source and from the built
0.3.0 wheel on 2026-08-06.

See [the migration table](../migration-v0.3.md) and
[the unified scenario guide](../three-tank.md) for the public contract.

## Source changes

Removed:

- `aiogym/scenarios/cascade/`
- `aiogym/scenarios/cascade_recirculating/`
- the obsolete `core-scenario-migration-v1.json` golden fixture

Added or renamed:

- `aiogym/core/compat.py`, the only legacy task alias table
- `aiogym/scenarios/three_tank/{physics,topologies,equipment,checks,study,controllers}.py`
- `aiogym/scenarios/three_tank/plants/{open-cascade-v1,recirculating-h1-v1,lab-three-tank-v1}.json`
- `tests/core/test_specs_v2.py`
- `tests/scenarios/test_three_tank_{phase0_golden,plants,unified_model}.py`
- `tests/scenarios/golden/three-tank-unification-phase0-v1.json`
- `docs/migration-v0.3.md` and `docs/three-tank.md`

## Legacy task mapping

| Deprecated task | Unified task | Plant | Condition |
|---|---|---|---|
| `cascade/regulation` | `three_tank/regulation` | `open-cascade-v1` | `continuous-benchmark` |
| `cascade/economic` | `three_tank/economic` | `open-cascade-v1` | `continuous-benchmark` |
| `cascade_recirculating/regulation` | `three_tank/regulation` | `recirculating-h1-v1` | `commissioning` |

These aliases emit `DeprecationWarning`, reject conflicting explicit plant or
condition values, and are scheduled for removal in 0.4.

## Plant acceptance matrix

| Plant ID | Topology | Actions | Default condition | Supported tasks | PID | MPC | Five-step golden equivalence |
|---|---|---:|---|---|---|---|---|
| `open-cascade-v1` | `open_cascade` | 7 | `continuous-benchmark` | regulation, economic | PASS | PASS | PASS |
| `recirculating-h1-v1` | `recirculating_loop` | 4 | `commissioning` | regulation | PASS | PASS | PASS |
| `lab-three-tank-v1` | `recirculating_loop` | 6 | `commissioning` | regulation | PASS | PASS | PASS |

The Phase 0 fixture compares state/action/output schemas, initial observation,
default and controller actions, and five fixed-action steps including state,
observation, reward, energy, constraints, and protection events. The three
interfaces also produced distinct `interface_hash` values.

The follow-up physical-layer cleanup on 2026-08-07 introduced one
`ThreeTankPhysicsKernel`. `OpenCascadeTopology` and `RecirculatingTopology`
inherit it directly, while `ThreeTankDesignModel` extends the recirculating
topology. State/output handling, runtime-input validation, action clipping,
disturbance vectors, MPC initialization, and mass/energy derivative assembly
are therefore shared rather than copied per plant. The topology classes retain
only their connection flows, installed actuators, interlocks, steady-state
inverses, energy models, and topology-specific reporting.

The same cleanup removed the separate design-v1 schema, default declaration,
and conversion module. Equipment compilation and engineering studies now read
the resolved PlantConfig v2 declaration directly.

## Removed empty condition names

Only conditions with executable definitions were migrated. The unimplemented
open-cascade names `commissioning`, `disturbance-rejection`, `safety-recovery`,
and `temperature-step`, and recirculating names `hydraulic-commissioning`,
`disturbance-rejection`, `safety-recovery`, and `temperature-step`, were
removed rather than registered as empty shells.

## Validation summary

- Source suite after legacy-schema removal: **80 passed**; Ruff and
  `git diff --check` passed.
- PID/MPC smoke: both controllers produced correctly shaped actions and one
  finite, non-terminal transition on all three plants.
- Workflow E2E: design study passed for both supported topologies; Dataset v3
  wrote, checksum-verified, and read a two-transition episode; a two-step SAC
  smoke train wrote `model.zip` and `contract.json`, then reloaded and evaluated
  for two steps without transfer flags.
- Packaging: isolated PEP 517 build produced `aiogym-0.3.0.tar.gz` and
  `aiogym-0.3.0-py3-none-any.whl`. Wheel inspection found all three built-in
  plant JSON files and no old scenario packages or tests.
- Clean install: the wheel plus `[rl]` dependencies was installed into a new
  temporary venv. Core import did not load Stable-Baselines3; CLI resource
  listing, all 7/4/6 action environments, design, PID/MPC, Dataset, SAC, and
  checkpoint replay all passed from `site-packages`.

The controller and SAC checks are short commissioning/smoke evidence, not
formal-horizon controller performance or research-scale RL validation.
