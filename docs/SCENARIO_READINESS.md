# Scenario readiness

This document records engineering maturity, not API stability. Every scenario
listed here remains discoverable through `aiogym.list_scenarios()` and
constructible through `aiogym.make_env()`. A lower readiness level does not
mean that a scenario is deprecated.

The inventory was audited against the repository on 2026-08-04. It describes
checked-in resources and automated tests only; it does not infer capability
from a model merely being registered.

## Readiness levels

| Level | Evidence required |
|---|---|
| L1 Environment-ready | Registered model, valid model metadata, and finite deterministic `make_env`/`reset`/`step` smoke |
| L2 Case-ready | L1 plus at least one versioned Case v2 with explicit operating constraints and reward context |
| L3 Benchmark-ready | L2 plus a training distribution, validation/test Track, ranking and safety configuration, and baselines |
| L4 Pipeline-ready | L3 plus automated Dataset v2 collection, online RL, BC/RLPD, checkpoint reload, validation, benchmark, and final-test workflow coverage |
| L5 Calibrated | L4 plus publishable parameter provenance, physical validation, and multi-seed reference performance ranges |

These levels are deliberately not exposed as a runtime registry, enum, or
filter on `list_scenarios()`.

## Audited capability matrix

| Scenario | Parameter profile | Case v2 | Training distribution | Official Track | Dataset E2E | Online RL E2E | BC / RLPD E2E | Level | Next promotion requirement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `quadruple` | `reference-parameterized`; one published reference | 4 | 5 (L0-L4) | 2 | Yes | Yes (SAC) | Yes / Yes | L4 | Complete release-grade physical validation and publish multi-seed reference ranges |
| `cascade` | `legacy-unverified`; no external reference in the profile | 5 | 3 (L0-L2) | 4 | Yes | Yes (SAC) | Yes / Yes | L4 | Validate benchmark parameters against evidence and publish multi-seed reference ranges |
| `cascade-recirculating` | `design-provisional`; V2.0 baseline with documented BOM conflicts | 5 | 6 (v1/v2 L0-L2) | 3 | Yes | Yes (SAC) | Yes / Yes | L4 | Complete release-grade physical validation and publish multi-seed reference ranges |
| `crystallization` | `legacy-unverified`; empty parameter provenance | 0 | 0 | 0 | No | No | No / No | L1 | Add a Case v2 with an explicit regulation goal and constraints |
| `cstr` | `legacy-unverified`; empty parameter provenance | 0 | 0 | 0 | No | No | No / No | L1 | Add a Case v2, then a versioned training distribution |
| `extraction` | `legacy-unverified`; empty parameter provenance | 0 | 0 | 0 | No | No | No / No | L1 | Add a physically reviewed Case v2 with explicit constraints |
| `heater` | `legacy-unverified`; empty parameter provenance | 0 | 0 | 0 | No | No | No / No | L1 | Add a physically reviewed Case v2 with explicit constraints |
| `hvac` | `legacy-unverified`; empty parameter provenance | 0 | 0 | 0 | No | No | No / No | L1 | Add a physically reviewed Case v2 with explicit constraints |

No scenario currently meets L5.

## Evidence behind the matrix

The shared L1 contract is enforced by
`aiogym/tests/test_all_scenarios_contract.py`. It checks the exact eight-ID
registry, structured metadata validation, state/action schema dimensions,
finite initial state, the Gymnasium five-value step contract, finite true state
and observations, and deterministic two-step behavior with stochastic features
disabled.

The checked-in resource inventory is:

- Case v2: four Quadruple cases, five Cascade cases, and five
  Cascade-recirculating cases. The other five scenarios have none.
- Training distributions: Quadruple L0-L4, Cascade L0-L2 plus its v2 L2
  distribution, and Cascade-recirculating v1/v2 L0-L2. No distribution is
  registered for the other five scenarios.
- Official Tracks: two Quadruple regulation Tracks; four Cascade Tracks
  (two regulation, economic, and recovery); and three Cascade-recirculating
  Tracks (two regulation versions and recovery). No Track is bundled for the
  other five scenarios.
- The regulation generalist Tracks for all three benchmark-ready scenarios bind
  a versioned training distribution. Their ranking declarations include a
  safety gate; the Quadruple, Cascade, and Cascade-recirculating regulation
  Tracks also bind checked-in anchor files.

The `*-regulation-generalist-v2` Tracks are the protocol-ready regulation
contracts: validation/test resolved Case identities are disjoint and reviewed
v3 fixed anchors are bundled. Their v1 predecessors remain loadable for
compatibility and diagnostics; they are not evidence of split isolation.

Quadruple pipeline evidence is distributed across
`aiogym/tests/e2e/test_collect_train_validate.py`,
`aiogym/tests/e2e/test_rlpd_offline_to_online.py`, and
`aiogym/tests/e2e/test_final_test_lock.py`. Cascade has a combined real
collect/BC/SAC/RLPD/reload/validation/baseline benchmark test in
`aiogym/tests/e2e/test_cascade_v1_workflow.py` and shares the final-test lock
test. The final-test lock test injects a deterministic evaluator to exercise
one-shot test-split consumption and statistical report generation; real Track
evaluation is exercised separately by the training and benchmark tests.

Cascade-recirculating has a combined real V2 Dataset collection,
BC/SAC/RLPD training, digest-verified checkpoint reload, validation, Python and
CLI benchmark workflow in
`aiogym/tests/e2e/test_cascade_recirculating_v2_workflow.py`. The test verifies
the four-action contract and the V2 L2 sensor/actuator dynamics recorded in
Dataset metadata. Its V2 Track also shares the one-shot final-test lock test.
Packaged collection and BC/SAC/RLPD quick profiles make the same workflow
available through the guided CLI. This is pipeline evidence for L4, not a
claim of research-scale learned-policy performance.

## Parameter provenance limits

All eight scenarios have a checked-in
`aiogym.parameter_profile.v1` document, but their evidence quality differs:

- Quadruple contains eight parameter records and cites Johansson (2000).
- Cascade contains thirteen parameter records sourced from legacy benchmark
  configuration or benchmark assumptions, with an empty references list.
- Cascade-recirculating contains twenty parameter records and one design bundle.
  It uses the V2.0 sheet as the executable baseline while explicitly recording
  conflicting procurement candidates and uncalibrated values.
- Crystallization, CSTR, extraction, heater, and HVAC carry
  `legacy-unverified` placeholder profiles with zero parameter records and zero
  references.

Consequently, environment readiness must not be presented as parameter
calibration or plant validity. Promotion to L5 requires evidence beyond passing
software contracts.
