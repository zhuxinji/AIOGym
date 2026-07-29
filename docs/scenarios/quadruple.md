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

env = aiogym.AIOGymEnv(
    "quadruple",
    case="minimum-phase",
    reward_spec="regulation-v1",
)
```

The minimum- and nonminimum-phase Cases declare their own plant parameters,
initial equilibria, timing, and setpoint schedules. Randomized references are
seeded and constrained to reachable operating regions.

`quadruple-regulation-generalist-v1` trains one policy contract across the
declared Case distribution and evaluates that same checkpoint on validation or
test Cases. Specialist controller profiles remain available for diagnostics.
