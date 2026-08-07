# AIO-Gym

AIO-Gym 0.4 is a compact process-control toolkit built around one execution
path:

```text
Scenario -> PlantConfig -> OperatingCondition -> TaskSpec -> Environment -> Run
```

The public surface remains small:

```python
import aiogym

env = aiogym.make_env("quadruple/regulation", condition="minimum-phase")
policy = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(policy, seeds=[0, 1, 2])
env.close()
```

Core execution requires NumPy and Gymnasium. Optional dependencies are grouped
by implemented capability:

```bash
pip install .
pip install 'aiogym[rl]'        # Stable-Baselines3 and Torch
pip install 'aiogym[symbolic]'  # CasADi symbolic dynamics
```

## CLI

```bash
aiogym list scenarios
aiogym list tasks
aiogym list plants --scenario three_tank
aiogym list conditions --scenario three_tank --plant lab-three-tank-v1

aiogym design run plant.json --condition commissioning --output runs/design/example
aiogym collect quadruple/regulation --condition minimum-phase --controller pid \
  --episodes 3 --output runs/data/quadruple-pid-v4
aiogym train quadruple/regulation sac --condition minimum-phase --steps 10000 \
  --eval-seeds 100 101 102 --output runs/train/quadruple-sac
aiogym evaluate quadruple/regulation --condition minimum-phase --controller pid \
  --seeds 0 1 2 --output runs/evaluate/quadruple-pid
```

Dataset v4 records both the reference/disturbance used by each action and the
context of the returned next observation. A schedule event at key `k` is
visible before action `k` is produced.

Three-tank design plants expose only installed actuators. Changing a heater
layout changes both `plant_hash` and `interface_hash`, so a checkpoint from a
different layout cannot be loaded implicitly.

The economic task uses `normalized_value`, not a claimed real currency. Its
reward integrates product value and electricity cost over `control_dt`; see
[the three-tank guide](docs/three-tank.md) for the formula.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Plant design](docs/design.md)
- [Three-tank scenario](docs/three-tank.md)
- [Migration to 0.4](docs/migration-v0.4.md)

Design studies are simulation screening, not safety certification. Installed
equipment and protection settings still require independent engineering review
and field commissioning. Short RL runs validate the pipeline, not controller
performance.
