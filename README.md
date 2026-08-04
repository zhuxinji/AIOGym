# AIO-Gym

AIO-Gym is a native Gymnasium toolkit for reproducible process-control
research. Its stable v1 product surface covers versioned Cases, RewardSpecs,
training distributions, Dataset v2, classical controllers, learned-policy
checkpoints, and official benchmark Tracks.

Supported in v1:

- eight process scenarios: Cascade, Cascade Recirculating, Crystallization,
  CSTR, Extraction, Heater, HVAC, and Quadruple Tank, with official benchmark
  coverage concentrated on Quadruple Tank and Cascade;
- PID, MPC, SAC, TD3, PPO, BC, and RLPD;
- validation selection, immutable ranking anchors, and one-shot final testing.

Scenario maturity is intentionally uneven:

- Stable benchmark/pipeline-ready: `quadruple`, `cascade`.
- Preview benchmark-ready: `cascade-recirculating`.
- Environment-only / experimental: `crystallization`, `cstr`, `extraction`,
  `heater`, `hvac`.

Real-device deployment, arbitrary Gym registration IDs, legacy Dataset v1
migration, and backend-specific command-line interfaces are not supported.

## Install

```bash
pip install .
pip install 'aiogym[rl]'      # Torch and Stable-Baselines3
pip install 'aiogym[onnx]'    # ONNX export and runtime
pip install 'aiogym[oracle]'  # CasADi oracle controller
pip install 'aiogym[hpo]'     # Optuna tuning
```

The core package requires only Gymnasium and NumPy. Importing `aiogym` or
creating an environment does not import optional dependencies.

## Create an environment

```python
import aiogym

env = aiogym.make_env("quadruple", case="minimum-phase",
                      reward_spec="regulation")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`aiogym.make_env(...)` is the only public construction path. Set randomness
only with `env.reset(seed=...)`.

## Discover and run

Start with short, explicit selectors. AIO-Gym always resolves them to canonical
versioned IDs before producing configurations or artifacts:

```bash
aiogym list tracks
aiogym describe track quadruple
aiogym list rewards
aiogym list profiles
```

Collect a small Dataset v2 tutorial bundle:

```bash
aiogym collect quadruple --profile quick --seed 0
```

Run the guided SAC tutorial profile:

```bash
aiogym train quadruple sac --profile quick --seed 0
```

Inspect the fully resolved configuration without training by adding
`--dry-run`. The `quick-v1` profiles are smoke tests, not credible
research-scale baselines. No built-in `baseline` or `full` profile is
registered because the repository does not yet contain a verified source
configuration for either budget. Use a reviewed immutable config for formal
experiments:

```bash
aiogym train --config experiments/my-sac.json
```

After training, evaluate its self-describing run manifest on validation:

```bash
aiogym evaluate runs/quadruple-sac-quick-seed0.run-result.json
aiogym benchmark --run runs/quadruple-sac-quick-seed0.run-result.json
```

Raw checkpoint and config-first workflows remain available:

```bash
aiogym evaluate \
  --checkpoint runs/policy.zip \
  --algorithm sac \
  --track quadruple-regulation-generalist-v1
aiogym benchmark --config configs/quickstart/quadruple/benchmark.json
```

Consume a locked test split exactly once only after selection is frozen:

```bash
aiogym final-test --config FILE
```

The same guided collect, SAC, BC, and RLPD tutorial paths are available for
`cascade`. BC and RLPD require `--dataset PATH`. Guided profiles are versioned
package resources, so they also work from an installed wheel. The retained
`configs/quickstart/*/benchmark.json` files are source-repository examples,
not profile sources. Formal experiments should use reviewed, immutable custom
configs rather than the tutorial-scale `quick` budgets.

Artifact output paths are protected by default. `aiogym evaluate`, `aiogym
benchmark`, and multi-seed training sweeps fail if their summary output already
exists; pass `--overwrite` only when replacement is intentional. Evaluation
requires environments created with `info_level="full"`; minimal info remains
available for high-throughput training and raw stepping.

## Official Tracks and canonical identity

Use `aiogym list tracks` for friendly selectors and descriptions, or
`aiogym list tracks --ids` for canonical IDs. Short selectors are accepted as
input only. Artifacts, hashes, resolved configurations, papers, and exact
reproduction use canonical IDs. Profiles are versioned templates; the emitted
resolved config is the actual experiment identity.

Cases define operating conditions; RewardSpecs define the scalar training
signal; Tracks bind the evaluation goal, split, cases, weights, safety gate,
fixed anchors, and ranking formula. Training and tuning can access only
training and validation. `final-test` is the sole ordinary command allowed to
consume a test split, and its lock cannot be reused.

For new Quadruple, Cascade, and Cascade-recirculating regulation experiments,
use the `*-regulation-generalist-v2` Tracks. They freeze disjoint
validation/test Case identities and reviewed v3 anchors; v1 regulation Tracks
remain available for compatibility and diagnostics.

Scenario implementation and benchmark maturity are tracked separately in
[Scenario readiness](docs/SCENARIO_READINESS.md). Validation is the public
selection and comparison split; the locked test split is not available through
`evaluate` or `benchmark`.

Every Dataset, resolved training configuration, checkpoint, Track, Case, and
anchor carries a reproducible identity or checksum. A learned checkpoint is
loaded through one canonical loader that verifies its SHA-256 digest before
backend deserialization and applies action normalization exactly once.
`aiogym describe reward regulation` shows the canonical reward terms, cost
channels, weights, and the Tracks that use them.

Advanced material is in [Documentation](docs/index.md),
[Public API](docs/public_api.md), [Architecture](docs/architecture.md), and
[API compatibility](docs/api_compatibility.md).

## Test and release checks

Core tests exclude optional dependency groups:

```bash
pytest -q -m "not rl and not oracle and not onnx and not e2e"
pytest -q -m rl      # install aiogym[rl]
pytest -q -m oracle  # install aiogym[oracle]
pytest -q -m onnx    # install aiogym[rl,onnx]
pytest -q -m e2e     # install aiogym[all]
```

Official source ZIPs are produced with `git archive`, which avoids macOS
resource-fork metadata. Release wheels and source archives are checked for
`__MACOSX`, `.DS_Store`, AppleDouble `._*`, Python bytecode, and cache paths.
Online SB3 configs use the cross-platform `spawn` subprocess start method by
default. Linux users may explicitly select `forkserver`; `fork` is retained
only as an advanced explicit option.
