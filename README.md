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
                      reward_spec="regulation-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`aiogym.make_env(...)` is the only public construction path. Set randomness
only with `env.reset(seed=...)`.

## Five workflows

The paths below refer to the checked-in examples in a source checkout. They
also ship in the source distribution, but not in the wheel; wheel users should
copy `configs/quickstart/` from the source distribution or repository before
running these commands.

Collect a Dataset v2 bundle:

```bash
aiogym collect --config configs/quickstart/quadruple/collect.json
```

Train the algorithm declared by one immutable configuration:

```bash
aiogym train --config configs/quickstart/quadruple/sac.json
```

Evaluate a frozen checkpoint on validation:

```bash
aiogym evaluate \
  --checkpoint runs/quickstart/quadruple/sac.zip \
  --algorithm sac \
  --track quadruple-regulation-generalist-v1
```

Compare PID, MPC, and optionally one learned checkpoint on an official Track:

```bash
aiogym benchmark --config configs/quickstart/quadruple/benchmark.json
```

Consume a locked test split exactly once:

```bash
aiogym final-test --config FILE
```

The same collect, SAC, BC, RLPD, and benchmark quickstarts are provided under
`configs/quickstart/cascade/`.

Artifact output paths are protected by default. `aiogym evaluate`, `aiogym
benchmark`, and multi-seed training sweeps fail if their summary output already
exists; pass `--overwrite` only when replacement is intentional. Evaluation
requires environments created with `info_level="full"`; minimal info remains
available for high-throughput training and raw stepping.

## Official Tracks

- `quadruple-regulation-generalist-v1`
- `cascade-regulation-generalist-v1`
- `cascade-economic-specialist-v1`
- `cascade-recovery-diagnostic-v1`
- `cascade-recirculating-regulation-generalist-v1`
- `cascade-recirculating-recovery-diagnostic-v1`

Cases define operating conditions; RewardSpecs define the scalar training
signal; Tracks bind the evaluation goal, split, cases, weights, safety gate,
fixed anchors, and ranking formula. Training and tuning can access only
training and validation. `final-test` is the sole ordinary command allowed to
consume a test split, and its lock cannot be reused.

Scenario implementation and benchmark maturity are tracked separately in
[Scenario readiness](docs/SCENARIO_READINESS.md). Validation is the public
selection and comparison split; the locked test split is not available through
`evaluate` or `benchmark`.

Every Dataset, resolved training configuration, checkpoint, Track, Case, and
anchor carries a reproducible identity or checksum. A learned checkpoint is
loaded through one canonical loader that verifies its SHA-256 digest before
backend deserialization and applies action normalization exactly once.

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
