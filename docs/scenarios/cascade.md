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

env = aiogym.AIOGymEnv(
    "cascade",
    case="continuous-benchmark",
    reward_spec="economic-v1",
)
```

`continuous-benchmark` declares the production throughput used by model
economics. Other Cases remain regulation/recovery experiments and do not embed
Goal or RewardSpec choices.

Official Tracks include a regulation generalist, an economic specialist, and a
recovery diagnostic. Use `aiogym.list_tracks()` for their canonical IDs.
