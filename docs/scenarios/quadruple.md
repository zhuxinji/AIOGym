# Quadruple-Tank model

The built-in `quadruple` Scenario implements Johansson's nonlinear four-tank,
two-pump laboratory process. Pump flow is split between lower and upper tanks;
the split fractions determine the phase configuration.

The model retains its source laboratory units:

- state `[h1, h2, h3, h4]`: `cm`;
- controlled output `[h1, h2]`: `cm`;
- action `[pump_1_voltage, pump_2_voltage]`: normalized `[0, 1]`, mapped to
  physical voltage through `max_voltage`;
- area, flow, voltage, and time: `cm^2`, `cm^3/s`, `V`, and `s`.

The policy observation has eight values: four normalized tank levels, two
normalized lower-tank reference errors, and two previous pump commands. Default,
randomized, and benchmark environments all keep this interface.

## Benchmarks

```bash
aiogym list benchmarks --scenario quadruple
```

| Benchmark | Fixed protocol | Horizon | Noise | Ranking |
|---|---|---:|---|---|
| `tracking` | equilibrium start; SP becomes `(16, 10)` cm at 120 s and `(10, 16)` cm at 360 s | 600 s | std 0.001 | tracking IAE |
| `disturbance-rejection` | pump-flow factor 0.85 at step 150, restored at 350 | 600 s | std 0.001 | unsafe rate, disturbance IAE, recovery time |
| `boundary-safety` | all tanks start at 90% of the resolved maximum level | 400 s | std 0.001 | unsafe rate, time to violation, tracking IAE |

The deterministic default training episode starts at the default equilibrium,
holds its initial reference for 300 s, and then steps to `(15, 10.5)` cm for
the remaining 300 s. The target has a feasible unsaturated steady pump command.
This single-step protocol is distinct from the fixed tracking Benchmark below.
With `randomize=True`, every reset instead samples one of the three training
families with values and event times distinct from the fixed benchmarks. Random
tracking targets span at least 15% of each output's physical range from the
nominal SP.

The tracking SP spans 6 cm, or 30% of each output's 0--20 cm physical range.
The targets stay away from the hard bounds and both have feasible steady pump
commands.

## SAC training

The repository includes a Quadruple-specific SAC profile at
`configs/quadruple-sac.json`. It keeps the generic training workflow unchanged
and is passed explicitly so the profile is not applied to unrelated scenarios:

```bash
aiogym train quadruple sac --steps 50000 --seed 0 --record-every 500 \
  --algorithm-kwargs configs/quadruple-sac.json \
  --output checkpoints/quadruple/sac-optimized-50k/seed-0
```

The profile uses a `0.001` learning rate, `0.995` discount, fixed `0.0005`
entropy coefficient, and batch size `256`. It was selected on the deterministic
default training episode and then checked separately on the fixed tracking
Benchmark. Training still starts from a new policy and uses only online
environment interaction; there is no warm start or offline dataset.

For a multi-seed result, repeat training with different `--seed` values and
report all checkpoints. Do not select a checkpoint using the formal Benchmark.

## PPO training

The Quadruple PPO profile aligns each on-policy rollout with the 600-step
training episode and performs ten optimization epochs per rollout:

```bash
aiogym train quadruple ppo --steps 50000 --seed 0 --record-every 500 \
  --algorithm-kwargs configs/quadruple-ppo.json \
  --output checkpoints/quadruple/ppo-optimized-50k/seed-0
```

The profile uses `n_steps=600`, batch size `100`, ten epochs, a `0.995`
discount, and the standard `0.0003` learning rate. PPO rounds the requested
training budget up to a whole rollout, so a 50,000-step request executes
50,400 environment steps. As with SAC, the formal tracking Benchmark is used
only after training.

## DDPG training

DDPG needs explicit exploration noise because its actor is deterministic. The
Quadruple profile supplies normal action noise through the JSON training
contract:

```bash
aiogym train quadruple ddpg --reward smooth-regulation \
  --steps 50000 --seed 0 --record-every 500 \
  --algorithm-kwargs configs/quadruple-ddpg.json \
  --output checkpoints/quadruple/ddpg-smooth-50k/seed-0
```

The profile uses action-noise standard deviation `0.1`, 1,000 random warm-up
steps, batch size `256`, a `0.995` discount, and learning rate `0.001`.
`smooth-regulation` adds a `0.1 * sum(delta_action**2)` training penalty to
discourage deterministic two-step actuator oscillation, then scales the dense
training reward by `100` so it is not overwhelmed by the safety penalty.
Formal Benchmarks keep the original `regulation` reward and unchanged tracking
metrics.
It is an explicit reproducible profile rather than a universal optimum; report
multiple training seeds before making statistical performance claims.

## Phase configuration and parameter overrides

Phase is derived only from the resolved `gamma` parameter:

- `gamma[0] + gamma[1] > 1`: minimum phase;
- `gamma[0] + gamma[1] < 1`: nonminimum phase;
- `gamma[0] + gamma[1] == 1`: transmission zero at the origin.

The default `gamma` is `[0.7, 0.6]`, so all formal Benchmarks use the
minimum-phase model. Benchmark construction rejects `parameters=` and
`reward=` overrides. To inspect tracking behavior with a nonminimum-phase
model, create an ordinary custom environment:

```python
import aiogym

env = aiogym.make_env(
    "quadruple",
    parameters={"gamma": [0.4, 0.4]},
)
```

That run is recorded with its resolved parameter set but is not labeled as a
formal Benchmark. This prevents results from different plants sharing the same
Benchmark identity.

## Run a controller

```python
env = aiogym.make_env("quadruple", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=[0, 1, 2])
env.close()
```

PID and MPC do not select different settings by Benchmark. The same controller
configuration is used across tracking, disturbance rejection, and boundary
safety so comparisons represent one policy under three tests.
