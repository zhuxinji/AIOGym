# Repository maintenance tools

These commands are for AIO-Gym maintainers. They are deliberately separate
from the stable `aiogym` user CLI and do not add console entry points.

Ordinary experiments should use `aiogym run EXPERIMENT.json`; the tools below
maintain benchmark definitions or release archives.

## Audit fixed anchors

Audit every built-in fixed-anchor manifest against its stored quality data:

```bash
python -m aiogym.benchmarks.anchors.audit
```

The reusable function is available from the explicit domain module:

```python
from aiogym.benchmarks.anchors.audit import audit_builtin_anchors
```

## Profile environment throughput

Run the repository script from an installed or editable development checkout:

```bash
python scripts/benchmark_env.py \
  --track quadruple \
  --n-envs 1,2,4,8 \
  --steps 100000
```

Track selectors use the package registry. Output records the resolved
canonical Track ID.

## Generate anchor candidates

Generate review candidates without replacing built-in official anchors:

```bash
python scripts/calibrate_official_anchors.py \
  --output-dir artifacts/anchor-candidates
```

The command fails if a candidate already exists and always uses the fixed
official calibration seeds. Some reference controllers require optional
dependencies such as CasADi.

## Review a regulation experiment

Protocol preflight, runtime capture, training review, learning-curve SVG, and
small review-bundle helpers live under `scripts/experiments/`. They read
canonical artifacts and refuse test-split content; they do not introduce a
second training API.

Run the prepared Quadruple v2 pilot through the same generic entry point used
by every formal experiment:

```bash
aiogym run configs/experiments/quadruple/sac-pilot-v2.json
```

The ExperimentSpec declares all three independent training seeds and stops at
validation. Existing outputs are not replaced unless `--overwrite` is passed.
`aiogym run` never calls `final-test`.

## Check release archives

Build and inspect both distribution formats before publication:

```bash
python -m build
python scripts/check_release_artifacts.py dist/*
```

The checker rejects missing runtime resources as well as tests, bytecode,
caches, macOS metadata, and other generated contamination.
