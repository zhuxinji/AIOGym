# Open cascade

The open cascade models three heated tanks with seven actuators, level and
temperature outputs, pump/heater interlocks, overflow and hard safety limits.

Bundled Cases:

- `cascade/commissioning`
- `cascade/temperature-step`
- `cascade/disturbance-rejection`
- `cascade/safety-recovery`
- `cascade/continuous-benchmark`

Example:

```python
import aiogym

env = aiogym.make_env(
    "cascade",
    case="continuous-benchmark",
    reward_spec="economic",
)
```

`continuous-benchmark` declares the production throughput used by model
economics. Other Cases remain regulation/recovery experiments and do not embed
Goal or RewardSpec choices.

Official Tracks include a regulation generalist, an economic specialist, and a
recovery diagnostic. Use `aiogym.list_tracks()` for their canonical IDs.

For new regulation experiments, use
`cascade-regulation-generalist-v2`. It freezes finite fixed-bounds observation
normalization, treats disturbances as unmeasured, keeps validation/test Case
identities disjoint, and binds reviewed v3 anchors. The v1 regulation Track is
retained for compatibility and diagnostics. These software contracts do not
upgrade the Cascade parameter profile beyond `legacy-unverified`, and the
economic Track is outside the v2 regulation experiment protocol.
