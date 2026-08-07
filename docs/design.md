# Plant design workflow

Plant design is part of the owning `ScenarioPlugin`, not a parallel model
stack. A strict `aiogym.plant.v2` `PlantConfig` contains equipment, topology,
study requirements, uncertainty declarations, conditions, and references.

```bash
aiogym design new three_tank plant.json
aiogym design validate plant.json
aiogym design run plant.json --condition commissioning --samples 20 --seed 7 --output runs/design/plant
aiogym design sweep plant.json \
  --parameter plant.heaters.H1.rated_power_kw --values 4 5 6 \
  --samples 10 --seed 7 --output runs/design/heater-sweep
```

The three-tank provider runs geometry, actuator, steady-state, dynamic
commissioning, protection, energy, and seeded robustness checks through the
same model, environment, PID Policy, and rollout executor used elsewhere.
Outputs include `result.json`, `manifest.json`, and `report.md`; existing output
is not overwritten by default.

Only `aiogym.plant.v2` declarations are accepted. The retired design-v1 schema,
default file, loader, and conversion API are not part of the runtime package.
Use `aiogym/scenarios/three_tank/plants/lab-three-tank-v1.json` as the complete
laboratory equipment example.

A PASS is simulation-screening evidence. It is not a safety certification and
does not replace procurement reconciliation, protection-layer review, hazard
analysis, or field commissioning.
