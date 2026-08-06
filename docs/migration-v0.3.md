# Migrating to AIO-Gym 0.3

Version 0.3 merges the three related public scenarios into one `three_tank`
plugin. The compatibility aliases below remain for the 0.3 release and are
scheduled for removal in 0.4.

| Deprecated task | Task | Plant | Condition |
|---|---|---|---|
| `cascade/regulation` | `three_tank/regulation` | `open-cascade-v1` | `continuous-benchmark` |
| `cascade/economic` | `three_tank/economic` | `open-cascade-v1` | `continuous-benchmark` |
| `cascade_recirculating/regulation` | `three_tank/regulation` | `recirculating-h1-v1` | `commissioning` |

Calls through these aliases emit `DeprecationWarning`. An explicit plant or
condition that conflicts with the mapping is rejected. The removed scenario
plugins and their source directories are not registered or retained.

`preset=` is a one-release alias for `condition=`. New Python and CLI calls
must use `condition`. The prior empty names `commissioning`,
`disturbance-rejection`, `safety-recovery`, and `temperature-step` from the
open cascade, plus `hydraulic-commissioning`, `disturbance-rejection`,
`safety-recovery`, and `temperature-step` from the recirculating scenario,
were not migrated because the 0.2 resources contained no behavior definition.

PlantConfig v1 and `aiogym.design_spec.v1` remain readable through one-shot
conversion. Newly emitted declarations use `aiogym.plant.v2` and separate
`plant_hash`, `condition_hash`, `study_hash`, and `config_hash`.

SB3 training now writes `model/model.zip` and `model/contract.json`. Learned
policies require compatible Task semantics and `interface_hash`; evaluating the
same interface on a different plant or condition is recorded as transfer.
