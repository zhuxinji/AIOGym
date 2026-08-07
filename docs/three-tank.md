# Unified three-tank scenario

`three_tank` is one physical-system family with two supported topology
strategies and three built-in equipment declarations.

| Plant | Topology | Actions | Default condition | Tasks |
|---|---|---:|---|---|
| `open-cascade-v1` | `open_cascade` | 7 | `continuous-benchmark` | regulation, economic |
| `recirculating-h1-v1` | `recirculating_loop` | 4 | `commissioning` | regulation |
| `lab-three-tank-v1` | `recirculating_loop` | 6 | `commissioning` | regulation |

All plants use the physical state order `h1, T1, h2, T2, h3, T3` and the
controlled-output order `level_1, level_2, level_3, temperature_1,
temperature_2, temperature_3`. The action schema is compiled from installed
actuators; there are no dummy action slots.

## Physical implementation boundary

The built-in plants are declarations, not separate model copies. Their runtime
inheritance is:

```text
ThreeTankPhysicsKernel
├── OpenCascadeTopology          -> open-cascade-v1
└── RecirculatingTopology        -> recirculating-h1-v1
    └── ThreeTankDesignModel     -> lab-three-tank-v1
```

`ThreeTankPhysicsKernel` owns the common six-state split, state/output bounds,
action validation and clipping, disturbance-vector handling, controlled and
display outputs, integral limits, MPC initialization, and three-tank mass and
energy derivative assembly. A topology supplies its flow connections, mixing
terms, installed heaters, interlocks, and topology-specific reporting. The lab
model changes the equipment and heater layout while reusing the recirculating
flow topology and the common balance assembly.

The remaining modules are not per-plant dynamics: `controllers.py` stores
plant-aware controller profiles; `equipment.py` compiles the laboratory
equipment declaration; `checks.py` contains engineering assessments; and
`study.py` adapts those assessments to the generic study workflow. All four
consume the resolved PlantConfig v2 contract; there is no parallel design
schema or migration layer.

```python
import aiogym

env = aiogym.make_env(
    "three_tank/regulation",
    plant="recirculating-h1-v1",
    condition="commissioning",
)
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
```

`PlantConfig` owns topology, equipment, actuators, physical parameters, and
safety settings. `OperatingCondition` independently owns the initial state,
reference, control cadence, horizon, disturbances, schedules, and observation
mode. `TaskSpec` owns regulation or economic reward and metric semantics.

Only the open cascade exposes `product_flow`, so requesting
`three_tank/economic` on either recirculating plant fails during environment
construction. Custom conditions may be passed as an `OperatingCondition`, a
mapping, a JSON path, or a built-in condition ID.
