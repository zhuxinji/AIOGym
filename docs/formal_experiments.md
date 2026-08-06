# Formal experiments

This document covers the reproducibility and benchmark controls intentionally
kept out of the introductory README.

## One ExperimentSpec per experiment

The recommended formal entry point is:

```bash
aiogym run configs/experiments/quadruple/sac-pilot-v2.json
```

An `aiogym.experiment.v1` declaration contains:

```json
{
  "schema_version": "aiogym.experiment.v1",
  "name": "quadruple-sac-pilot-v2",
  "description": "Three-seed SAC pilot.",
  "output_dir": "runs/experiments/quadruple-sac-pilot-v2",
  "seeds": [0, 1, 2],
  "training": {
    "schema_version": "aiogym.rl_training_config.v3",
    "track_id": "quadruple-regulation-generalist-v2",
    "algorithm_id": "sac",
    "budget": {
      "unit": "environment_transitions",
      "value": 100000
    },
    "n_envs": 1,
    "validation_seeds": [5000]
  }
}
```

The experiment owns `seeds`, `output_dir`, and the internal run name. These
fields must not be duplicated inside `training`. `--dry-run` resolves the full
training declaration without creating the output directory.

## Output contract

The experiment result is committed last as `run.json`. Its presence means the
workflow completed. `metrics.json` contains compact validation and training-seed
statistics; `report.md` is the readable view. Everything needed for recovery or
audit remains under `internal/`, including:

- the resolved ExperimentSpec;
- resolved per-seed training configurations;
- checkpoint and replay state;
- run and multi-seed claims;
- benchmark artifacts and detailed provenance.

The public files summarize the internal record; they do not replace it.

## Lower-level commands

The following commands remain available for debugging and custom pipelines:

```bash
aiogym collect TARGET --profile quick
aiogym train TARGET ALGORITHM --profile quick
aiogym evaluate RUN_RESULT
aiogym benchmark --run RUN_RESULT
aiogym tune --config TUNING_CONFIG
```

`aiogym train --config` accepts a low-level
`aiogym.rl_training_config.v3` declaration. Formal repository experiments use
`aiogym run` and an `aiogym.experiment.v1` declaration instead.

## Protocol identity

Cases define reproducible operating conditions. RewardSpecs define the scalar
training signal. Tracks bind the evaluation Goal, Case population, policy
contract, safety gate, fixed anchor, and ranking rule. Friendly selectors are
input conveniences; resolved artifacts always record canonical IDs and hashes.

Validation is the public model-selection split. Training and tuning cannot
consume the locked test split. The one-shot command is deliberately separate:

```bash
aiogym final-test --config FILE
```

Run it only after the model and experiment selection are frozen. Its lock is not
reusable.

## Custom checkpoint evaluation

Self-describing run manifests are preferred:

```bash
aiogym evaluate runs/example/example.run-result.json
aiogym benchmark --run runs/example/example.run-result.json
```

Raw checkpoints remain available as an advanced escape hatch:

```bash
aiogym evaluate \
  --checkpoint runs/policy.zip \
  --algorithm sac \
  --track quadruple-regulation-generalist-v2
```

Evaluation and benchmark outputs use no-overwrite writes by default. Learned
checkpoints are digest-verified before backend deserialization, and action
normalization is applied exactly once by the canonical loader.
