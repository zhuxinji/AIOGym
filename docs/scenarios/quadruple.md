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

The policy observation has six values: four normalized tank levels and the two
normalized lower-tank setpoints. Both states and setpoints use their physical
`0--20 cm` ranges and therefore appear in `[0, 1]`. The policy learns the
tracking error from the current levels and setpoints instead of receiving an
explicit error feature. Default, randomized, and benchmark environments all
keep this interface. The controller selects a new action every second; the
plant is integrated internally at 0.1-second intervals while that action is
held constant.

## Benchmarks

```bash
aiogym list benchmarks --scenario quadruple
```

| Benchmark | Protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded feasible equilibrium start; independently resolved equilibrium targets at steps 120 and 360 | 600 steps / 600 s | unsafe rate, cumulative return |
| `disturbance-rejection` | pump-flow factor 0.85 at step 150 (150 s), restored at step 350 (350 s) | 600 steps / 600 s | unsafe rate, cumulative return |
| `boundary-safety` | all tanks start at 90% of the resolved maximum level | 400 steps / 400 s | unsafe rate, cumulative return |

All Benchmarks use exact observations without measurement noise. For tracking,
each non-negative evaluation seed selects one reproducible case. Reset samples
the initial lower-tank reference and two staged targets in `7–16 cm`. Both
controlled levels must change by at least `3 cm` at each event. The model then
derives both pump actions and the complete four-tank equilibrium for every
reference. Each equilibrium must reproduce the requested lower-tank levels,
keep all four tanks inside `0–20 cm`, keep both normalized pump actions inside
`0.02–0.95`, and have zero model derivative. All three equilibria are resolved
once during reset; no precomputed case table or event-time steady-state solve is
used.

The deterministic default training episode has 600 control steps. It starts at
the default equilibrium while requesting `(18, 8)` cm, so both controlled
outputs have an immediate error of at least 4 cm. After 300 steps (300 s), the
reference changes to `(8, 18)` cm for the remaining 300 steps. Both targets have
feasible unsaturated steady pump commands and are distinct from the formal
tracking Benchmark targets.
This single-step protocol is distinct from the case-seeded tracking Benchmark.
With `randomize=True`, every reset instead samples a new operating condition
for one tracking task using an independent random stream. 80% of episodes start
from a feasible interior equilibrium and apply one sampled target at a random
time. 20% start with all four levels at `80–92%` of the hard upper bound and
immediately track a feasible interior target. `disturbance=True` is independent
of this operating-condition distribution: it temporarily multiplies pump flow
by `0.78–0.94`, beginning at step `120–240` for `120–240` steps, before
restoring the factor to `1.0`. The fixed disturbance-rejection and
boundary-safety Benchmarks remain evaluation-only protocols. Interior tracking
moves span at least 15% of each controlled output's physical range.

The tracking Benchmark's `7–16 cm` envelope stays away from the hard bounds.
The `3 cm` minimum move equals 15% of each output's physical range.

## SAC training

Without `--algorithm-kwargs`, Stable-Baselines3 supplies every SAC
hyperparameter from its defaults:

```bash
aiogym train quadruple sac --steps 50000 --seed 0 --record-every 500 \
  --evaluate-every 5000 --evaluation-seed 0 \
  --output rl/checkpoints/quadruple/sac-sb3-default-50k/seed-0
```

SAC retains its native SB3 initialization and exploration.

For a multi-seed result, repeat training with different `--seed` values and
report all checkpoints. Do not select a checkpoint using the formal Benchmark.

## PPO training

PPO likewise uses its SB3 defaults when no algorithm-kwargs file is supplied:

```bash
aiogym train quadruple ppo --steps 50000 --seed 0 --record-every 500 \
  --evaluate-every 5000 --evaluation-seed 0 \
  --output rl/checkpoints/quadruple/ppo-sb3-default-50k/seed-0
```

With the default `n_steps=2048`, PPO rounds a 50,000-step request up to 51,200
environment steps.

## DDPG training

DDPG also uses its SB3 defaults. In particular, AIO-Gym does not add action
noise unless it is explicitly configured:

```bash
aiogym train quadruple ddpg --steps 50000 --seed 0 --record-every 500 \
  --evaluate-every 5000 --evaluation-seed 0 \
  --output rl/checkpoints/quadruple/ddpg-sb3-default-50k/seed-0
```

Training uses the Scenario's default `regulation` Reward.

Periodic training evaluation always uses a separate deterministic default
environment, never a fixed Benchmark. It evaluates at step zero, every 5,000
steps, and the final step. The final learner is saved as `model.zip`; checkpoint
selection first requires safe completion, then prefers the longer episode, and
only then maximizes the Reward's cumulative `return` primary metric. The
selected model is saved as
`best/model.zip`, its fixed-training-case trajectory is written to
`best/tracking.svg`, and compact results are recorded in `evaluation_history.json`.
Formal tracking, disturbance-rejection, and boundary-safety Benchmarks remain
post-training tests.

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

Here, seeds `0`, `1`, and `2` are three different operating cases. The same
seed resolves the same initial state, initial action, and staged references for
every compared policy. Disturbance-rejection and boundary-safety remain fixed
protocols, so their case seeds intentionally resolve the same episode.

PID and MPC do not select different settings by Benchmark. The same controller
configuration is used across tracking, disturbance rejection, and boundary
safety so comparisons represent one policy under three tests. The Quadruple PID
uses diagonal gains `(Kp, Ki, Kd) = (1.0159437333, 0.1, 0)` and
`(2.0, 0.0900169896, 0)` for the two lower-tank loops. These gains use the
1-second control interval.
