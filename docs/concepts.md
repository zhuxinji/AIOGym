# Concepts

The redesign separates five concerns that used to overlap:

```text
Scenario ──> physical dynamics and metadata
Case ──────> one reproducible operating experiment
RewardSpec > scalar learning signal
Goal ──────> evaluation intent and primary metric
Track ─────> official population, policy contract, seeds, ranking, safety
```

## Scenario

A Scenario owns process equations, state and action meaning, physical bounds,
disturbances, default operating values, and economic metadata. It does not own
an experiment schedule or benchmark ranking.

## Case

A Case is a versioned, scenario-bound declaration. It can define:

- initialization and model parameters;
- control interval and episode length;
- setpoint and disturbance schedules;
- operating requirements;
- safety/recovery conditions;
- optional metric acceptance thresholds.

A Case cannot define Goal, RewardSpec, controller defaults, or official ranking.
Built-ins live under `aiogym/models/cases/builtin/<scenario>/`.

## RewardSpec

A RewardSpec defines the scalar reward used by learning:

- `regulation` resolves to canonical `regulation-v1`;
- `economic` resolves to canonical `economic-v1`.

Both use the canonical reward engine and expose a decomposition in transition
info. Reward selection does not change state evolution. A
`RewardScaleWrapper` can rescale the returned scalar without changing scorecard
measurements.

Run `aiogym describe reward regulation` or `aiogym describe reward economic`
to inspect every reward term, cost channel, weight, and consuming Track. Human
descriptions are display metadata and do not participate in RewardSpec hashes.

## Goal and Scorecard

Goal is evaluation intent:

- `regulation`, ranked by a minimize metric such as `regulation_cost`;
- `economic`, ranked by a maximize metric such as `profit`.

Every evaluation records the full Scorecard regardless of Goal. Training return
is tagged with `reward_spec_id` and is not used as a cross-RewardSpec score.
Safety gates can make an otherwise high-scoring run ineligible.

## Track

A Track is the official benchmark unit. It fixes:

- scenario and Case population;
- specialist or generalist policy scope;
- Goal and RewardSpec;
- observation and action contract;
- training, validation, and test seed namespaces;
- primary metric, direction, aggregation, and safety gate.

One generalist Track evaluates one checkpoint across all declared Cases.
Specialist diagnostics remain possible through an explicit scenario/Case run.

## Precedence

Live APIs have no implicit compatibility precedence. Resolution is direct:

1. an official Track supplies its complete contract; or
2. an explicit run supplies Scenario, optional Case, Goal, and RewardSpec.

Conflicting Track overrides fail instead of silently replacing Track fields.

## Reproducibility

A defensible result records the Track ID/hash or explicit Case hash, Goal,
RewardSpec ID/hash, controller metadata, policy scope, seed namespace, package
version, code revision, and safety eligibility. Robustness comparisons use
paired nominal/shifted seeds.
