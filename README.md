# AIO-Gym

AIO-Gym 0.2 is a compact, scenario-oriented process-control toolkit. A
`Scenario` owns its process model and supported `Task` definitions; a
`PlantConfig` owns equipment parameters; a `Run` records one design,
collection, training, or evaluation workflow.

The stable Python surface is deliberately small:

```python
import aiogym

env = aiogym.make_env("quadruple/regulation", preset="minimum-phase")
policy = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(policy, seeds=[0, 1, 2])
env.close()
```

Core execution requires only Gymnasium and NumPy. Install the optional RL
dependencies for Stable-Baselines3 training:

```bash
pip install .
pip install 'aiogym[rl]'
```

## Five commands

```bash
aiogym list scenarios
aiogym list tasks

aiogym design run configs/design/cascade-recirculating-example-v1.json \
  --output runs/design/example

aiogym collect quadruple/regulation \
  --preset minimum-phase --controller pid --episodes 3 \
  --output runs/data/quadruple-pid-v3

aiogym train quadruple/regulation sac \
  --preset minimum-phase --steps 10000 --eval-seeds 100 101 102 \
  --output runs/train/quadruple-sac

aiogym evaluate quadruple/regulation \
  --preset minimum-phase --controller pid --seeds 0 1 2 \
  --output runs/evaluate/quadruple-pid
```

Dataset collection writes episode-oriented Dataset v3 bundles. Training is a
thin SAC/PPO/TD3/DDPG wrapper that saves a checkpoint and immediately evaluates
it on explicit seeds. Short examples are pipeline smoke tests, not performance
claims.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Plant design](docs/design.md)
- [Migration from 0.1](docs/migration-v0.2.md)

Design studies are simulation screening, not safety certification. Installed
equipment and protection settings still require independent engineering review
and field commissioning.
