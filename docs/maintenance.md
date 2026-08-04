# Repository maintenance tools

These commands are for AIO-Gym maintainers. They are deliberately separate
from the stable `aiogym` user CLI and do not add console entry points.

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

## Review a regulation pilot

Protocol preflight, runtime capture, training review, learning-curve SVG, and
small review bundles live under `scripts/experiments/`. They read the canonical
config-first artifacts and refuse test-split content; they do not introduce a
second training API.

The prepared Quadruple v2 pilot is intentionally user-run:

```bash
bash scripts/experiments/run_quadruple_sac_pilot.sh
```

It runs three independent SAC training seeds and stops at validation review.
Existing outputs are not replaced unless `AIOGYM_PILOT_OVERWRITE=1` is set.
The script never calls `final-test`.
