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

```python
import aiogym

print(aiogym.list_benchmarks("quadruple"))
```

| Benchmark | Protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded feasible equilibrium start that immediately tracks one independently resolved target | 180 steps / 180 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded pump-flow loss, event time, and recovery time | 500 steps / 500 s | unsafe rate, cumulative return |
| `boundary-safety` | the normal state is forward pre-run with high legal pump voltages until one tank reaches a case-seeded `82--90%` high-level boundary | 180 steps / 180 s | unsafe rate, cumulative return |

All Benchmarks use exact observations without measurement noise. For tracking,
each non-negative evaluation seed selects one reproducible case. Reset samples
an initial lower-tank operating point and one target in `7–16 cm`. Both
controlled levels differ by at least `3 cm` from the initial output, and
tracking begins on the first control step. The model derives both pump actions
and the complete four-tank equilibrium for both operating points. Each
equilibrium must reproduce the requested lower-tank levels,
keep all four tanks inside `0–20 cm`, keep both normalized pump actions inside
`0.02–0.95`, and have zero model derivative. Both equilibria are resolved
once during reset; no precomputed case table or run-time steady-state solve is
used.

Disturbance cases begin at step `120--220`, last `150--250` steps, and sample a
pump-flow factor in `0.78--0.90`. Boundary cases pre-run high legal pump
voltages from the normal state until one tank reaches `82--90%` of the resolved
maximum. Seeds `0--19` select 20 distinct episodes for all three
Benchmarks.

The latest possible disturbance restoration is at step 470, so the 500-step
protocol keeps a 30-second response tail. Boundary safety uses the same
180-second response window as tracking.

The deterministic default training episode starts at the default equilibrium,
immediately requests `(8, 18)` cm, and uses the tracking Benchmark's 180-step
horizon. The target has a feasible unsaturated steady pump command.
With `randomize=True`, every reset instead samples a new operating condition
for one tracking episode using an independent random stream. By default every
episode starts from a feasible interior equilibrium and immediately tracks one
sampled target. Setting `boundary_probability=p` makes fraction `p` start from
the forward-pre-run family with all four levels at `80–92%` of the hard upper
bound and immediately track a feasible interior target. Every training episode lasts 180
steps. `disturbance=True` is independent
of this operating-condition distribution: it temporarily multiplies pump flow
by `0.78–0.94`, beginning at step `36–72` for `36–72` steps, before
restoring the factor to `1.0`. The fixed disturbance-rejection and
boundary-safety Benchmarks remain evaluation-only protocols. Interior tracking
moves span at least 15% of each controlled output's physical range.

The tracking Benchmark's `7–16 cm` envelope stays away from the hard bounds.
The `3 cm` minimum move equals 15% of each output's physical range.

## RL training

The example below trains SAC with the Stable-Baselines3 defaults. Change
`algorithm` to `"ppo"` or `"ddpg"` to train the other included baselines.

```python
import aiogym

training_env = aiogym.make_env("quadruple", randomize=True)
evaluation_env = aiogym.make_env("quadruple", randomize=True)
trained = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=50_000,
    seed=0,
    record_every=500,
    evaluation_env=evaluation_env,
    evaluate_every=5_000,
    output=(
        "runs/quadruple/training/sac/"
        "randomized-sac-sb3-default-50k/seed-0"
    ),
)
training_env.close()
evaluation_env.close()
```

SAC retains its native SB3 initialization and exploration. DDPG does not add
action noise unless it is explicitly configured. With PPO's default
`n_steps=2048`, a 50,000-step request completes 51,200 environment steps.
Training uses the scenario's default `regulation` Reward.

For a multi-seed result, repeat training with different `seed` values and
report all checkpoints. Do not select a checkpoint using the formal Benchmark.

Periodic evaluation follows the shared
[best-checkpoint selection](../architecture.md#select-the-best-checkpoint) on a
separate randomized environment. Formal tracking, disturbance-rejection, and
boundary-safety Benchmarks remain held-out post-training tests.

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

## Quick start and controllers

```python
import aiogym

env = aiogym.make_env("quadruple", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=range(20))
env.close()
```

Here, seeds `0--19` are 20 different operating cases. The same seed selects
the same complete physical case for every compared policy. This applies to
tracking, disturbance rejection, and boundary safety.

PID and MPC do not select different settings by Benchmark. The same controller
configuration is used across tracking, disturbance rejection, and boundary
safety so comparisons represent one policy under three tests. The Quadruple PID
uses diagonal gains `(Kp, Ki, Kd) = (1.0159437333, 0.1, 0)` and
`(2.0, 0.0900169896, 0)` for the two lower-tank loops. These gains use the
1-second control interval.

Current controller scores and safety metrics remain in the generated
`runs/quadruple/benchmarks/<benchmark>/comparison.json` files.
