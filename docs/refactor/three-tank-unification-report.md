# Three-tank unification report

Date: 2026-08-06

This report is finalized after Phase 8 validation. The implementation removes
the two old scenario packages, adds three PlantConfig v2 declarations, and
registers only `quadruple` and `three_tank` as public scenarios.

See [the migration table](../migration-v0.3.md) and
[the unified scenario guide](../three-tank.md) for the public contract.

## Removed empty condition names

Only conditions with executable definitions were migrated. The unimplemented
open-cascade names `commissioning`, `disturbance-rejection`, `safety-recovery`,
and `temperature-step`, and recirculating names `hydraulic-commissioning`,
`disturbance-rejection`, `safety-recovery`, and `temperature-step`, were
removed rather than registered as empty shells.

## Validation summary

Phase 8 records numerical equivalence, PID/MPC smoke results, workflow E2E,
and isolated wheel-install evidence here.
