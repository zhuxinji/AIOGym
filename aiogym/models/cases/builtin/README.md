# Built-in Cases

Each JSON file is a `aiogym.case_profile.v2` declaration identified by
`scenario/name`.

A Case owns reproducible experiment conditions: environment timing,
initialization, setpoints, disturbances, constraints, operation, optional model
parameters, and acceptance thresholds. It does not own Goal, RewardSpec,
official ranking, or a generalist controller policy.

Load Cases through:

```python
case = aiogym.load_case("quadruple/minimum-phase")
env = aiogym.AIOGymEnv(
    "quadruple",
    case=case,
    reward_spec="regulation-v1",
)
```

Case overrides are validated and limited to Case-owned sections. The resolved
profile has a stable content hash used by Tracks and evaluation provenance.
