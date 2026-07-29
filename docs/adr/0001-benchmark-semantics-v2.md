# ADR 0001: Case/Goal/RewardSpec/Track benchmark semantics

Status: accepted and implemented.

## Context

The former API used overlapping abstractions for experiment conditions,
training reward, reporting views, and benchmark expansion. This made it
possible for constructor defaults, evaluation configuration, and training
configuration to disagree.

## Decision

Use five independent contracts:

1. Scenario owns physical process semantics.
2. Case owns reproducible operating conditions.
3. RewardSpec owns the scalar learning signal.
4. Goal owns evaluation intent and primary metric.
5. Track owns an official benchmark population and policy contract.

Every evaluation emits a reward-independent Scorecard. Track ranking uses its
Goal metric and safety gate. Training return is comparable only for the same
RewardSpec.

Official generalist Tracks train one checkpoint across a declared Case
distribution and evaluate that checkpoint across validation/test Cases.
Explicit scenario/Case runs are specialist diagnostics.

## API decisions

- The environment class is `AIOGymEnv`.
- The environment constructor accepts `case` and `reward_spec`.
- Cases are discovered through `list_cases()` and `load_case()`.
- Tracks are discovered through `list_tracks()` and `load_track()`.
- Evaluation accepts an optional `goal_specification`.
- Live result/artifact fields use `case`, `goal`, `reward_spec_id`,
  `scorecard`, `official_score`, and Track provenance.

The former compatibility entry points are removed. Historical artifact parsing
is confined to `aiogym.evaluation.legacy_artifacts` and is not imported by live
execution or artifact writing.

## Invariants

1. A Case cannot select Goal or RewardSpec.
2. A RewardSpec cannot select the Case population.
3. Environment Goal must match its RewardSpec.
4. A Track validates every Case against one policy contract.
5. Validation and test seeds are disjoint from training sampling.
6. Scorecard metrics do not depend on RewardSpec selection.
7. Safety gating is separate from the primary metric.
8. New artifacts never pass through historical normalization.

## Consequences

The public surface is smaller and stricter. Old calls fail visibly instead of
being silently translated. Archived files remain inspectable through an
explicit offline migration import.
