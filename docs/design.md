# Plant design workflow

Plant design is part of the owning `ScenarioPlugin`, not a parallel model
stack. A strict `aiogym.plant.v1` `PlantConfig` contains equipment, operating
point, study requirements, uncertainty declarations, and references.

```bash
aiogym design new three_tank plant.json
aiogym design validate plant.json
aiogym design run plant.json --samples 20 --seed 7 --output runs/design/plant
aiogym design sweep plant.json \
  --parameter plant.heaters.H1.rated_power_kw --values 4 5 6 \
  --samples 10 --seed 7 --output runs/design/heater-sweep
```

The three-tank provider runs geometry, actuator, steady-state, dynamic
commissioning, protection, energy, and seeded robustness checks through the
same model, environment, PID Policy, and rollout executor used elsewhere.
Outputs include `result.json`, `manifest.json`, and `report.md`; existing output
is not overwritten by default.

Legacy `aiogym.design_spec.v1` JSON can be loaded and is deterministically
converted to `PlantConfig`. This is an input conversion only: the resulting run
is governed by the 0.2 scenario model and workflow.

A PASS is simulation-screening evidence. It is not a safety certification and
does not replace procurement reconciliation, protection-layer review, hazard
analysis, or field commissioning.
