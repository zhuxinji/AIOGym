# Plant design workflow

Design studies are owned by the Scenario plugin and use the same resolved Plant,
Condition, model, controller, and rollout path as normal execution.

```bash
aiogym design new three_tank plant.json
aiogym design validate plant.json
aiogym design run plant.json --condition commissioning --samples 20 --seed 7
aiogym design sweep plant.json --parameter plant.heaters.0.power_w \
  --values 1900 2000 2100 --condition commissioning --controller pid
```

Python accepts a complete custom condition:

```python
import aiogym

result = aiogym.study(
    "lab-three-tank-v1",
    condition={
        "id": "custom-test",
        "initial_state": [0.20, 20.0, 0.21, 20.0, 0.22, 20.0],
        "reference": [0.20, 0.21, 0.22, 30.0, 29.0, 28.0],
        "control_dt": 1.0,
        "horizon": 1800,
        "disturbances": {
            "t_amb": 18.0,
            "pump_flow_factor": 1.0,
            "heater_efficiency": 1.0,
            "heat_loss_factor": 1.0,
        },
        "observation": "controlled-output",
    },
)
```

Static, steady-state, safety, dynamic, and robustness checks all use this exact
resolved condition. If an explicit `maximum_heatup_time_s` exceeds
`control_dt * horizon`, the result includes
`insufficient_assessment_horizon`; the requirement is never truncated or
replaced by the default condition's horizon.

Only `aiogym.plant.v2` declarations are accepted. A PASS is simulation-screening
evidence, not safety certification or field commissioning evidence.
