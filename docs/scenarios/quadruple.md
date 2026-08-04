# Johansson quadruple tank

The quadruple-tank model follows the standard four-level, two-pump nonlinear
process. Pump split fractions determine minimum- or nonminimum-phase behavior.

Bundled Cases:

- `quadruple/minimum-phase`
- `quadruple/nonminimum-phase`
- `quadruple/zero-boundary-stress`
- `quadruple/disturbance-rejection`

```python
import aiogym

env = aiogym.make_env(
    "quadruple",
    case="minimum-phase",
    reward_spec="regulation",
)
```

The minimum- and nonminimum-phase Cases declare their own plant parameters,
initial equilibria, timing, and setpoint schedules. Randomized references are
seeded and constrained to reachable operating regions.

`quadruple-regulation-generalist-v1` trains one policy contract across the
declared Case distribution and evaluates that same checkpoint on validation or
test Cases. Specialist controller profiles remain available for diagnostics.

`quadruple-regulation-generalist-v2` is the protocol-ready successor for new
experiments. It preserves the physical policy-facing contract, keeps the
held-out disturbance unobserved, separates validation/test resolved Case
identities, and binds reviewed v3 fixed anchors. The v1 Track remains loadable
for compatibility and diagnostics.
