# Recirculating cascade

The recirculating cascade models a three-vessel closed hydraulic loop with four
actuators, passive overflow return, heater and pump interlocks, and distinct
hard termination conditions.

Bundled Cases:

- `cascade-recirculating/commissioning`
- `cascade-recirculating/temperature-step`
- `cascade-recirculating/disturbance-rejection`
- `cascade-recirculating/safety-recovery`

```python
import aiogym

env = aiogym.AIOGymEnv(
    "cascade-recirculating",
    case="commissioning",
    reward_spec="regulation-v1",
)
```

The scenario has no production-economic benchmark. Its official resources are
a regulation generalist Track and a recovery diagnostic Track. Passive
protection events are reported separately from hard process termination.
