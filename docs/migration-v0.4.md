# Migrating to AIO-Gym 0.4

Version 0.4 hardens the 0.3 architecture without introducing a parallel
runtime. Four changes intentionally affect compatibility.

## Dataset v4 and schedule timing

Dataset schemas are now `aiogym.dataset.v4` and
`aiogym.dataset.episode.v4`. Event key `k` means the event is applied before
action `k`. Episode arrays distinguish `transition_reference` and
`transition_disturbance` from the `reference` and `disturbance` associated with
the next observation.

Dataset v3 is not accepted by the v4 runtime reader. Recollect scheduled data
when policy/action timing matters; changing metadata alone cannot reconstruct
what a policy observed under the old timing.

## Custom three-tank actions

Parameterized equipment exposes only installed actuators. H1-only, H1+H3, and
H1/H2/H3 layouts have 4, 5, and 6 public actions. Layout changes alter
`interface_hash`; retrain or deliberately rebuild a compatible policy.

The three built-in Plant dimensions remain 7 (`open-cascade-v1`), 4
(`recirculating-h1-v1`), and 6 (`lab-three-tank-v1`).

## Economic Task revision 2

`three_tank/economic` now uses `net-economic-value-v2`,
`economic-core-v2`, and coefficients included in `task_hash`. Rewards use
product volume and kWh-compatible energy cost integrated over `control_dt`.
Old economic checkpoints fail the strict Task contract and must not be silently
reused.

## Policy transfer

Checkpoint Plant and Condition transfer default to off. Enable only the
specific intended difference with `allow_plant_transfer=True` or
`allow_condition_transfer=True`. Task and interface transfer are always
rejected. Results and Dataset manifests record the training contract, target
contract, status, and flags.

## Extras

The CasADi extra is named `symbolic`. Retired `oracle`, `onnx`, and `hpo`
extras are no longer published because the corresponding product features do
not exist in this runtime.
