# Three-Tank model

The built-in `three_tank` Scenario represents one fixed laboratory process:

- three 60 L process tanks;
- one separate 180 L source/return reservoir boundary;
- pump P101;
- valves V12, V23, and V34;
- one 2 kW heater H1 in Tank 1.

The state is `[h1, T1, h2, T2, h3, T3]`. The physical action is
`[P101, V12, V23, V34, H1]`, with every value normalized to `[0, 1]`. All modes
keep the same 17-dimensional observation and 5-dimensional action interface.

Three-Tank parameters primarily use SI units (`m`, `m^2`, `m^3/s`, `W`, and
`W/K`) with temperatures in `degC`. Use `list_parameters()` for the exact unit
of every override.

## Default episode, Rewards, and Benchmarks

The deterministic default is a feasible equilibrium-to-equilibrium heating
step. It starts at the nominal `0.18 m` equilibrium with `T3 = 23.219 degC`,
changes at 600 s to `0.30 m` and `T3 = 23.7 degC`, and has a 3000 s horizon.
Both ends use 5 L/min and have unsaturated steady physical actions. This
training protocol is distinct from the fixed tracking Benchmark and is not a
named public selector.

Rewards are:

- `regulation`: full-output tracking, with `0.4 m` level and `10 degC`
  temperature error scales, plus a penalty for deviation from the
  reference's feasible steady action;
- `tank3-regulation`: Tank 3 level and temperature tracking;
- `economic`: outlet production value minus energy cost.

Every Reward includes an explicit terminal safety-violation penalty.
Formal Benchmarks always use the default `regulation` Reward and default model
parameters; `reward=` and `parameters=` are rejected when `benchmark=` is set.

```bash
aiogym list benchmarks --scenario three_tank
aiogym list rewards --scenario three_tank
aiogym list parameters --scenario three_tank
```

| Benchmark | Fixed protocol | Horizon | Noise | Ranking |
|---|---|---:|---|---|
| `tracking` | at 5 L/min, all levels `0.12 -> 0.30 m` and temperature profile `[20.62, 20.56, 20.5] -> [23.73, 23.34, 23.0] degC at 900 s | 4200 s | std 0.001 | unsafe rate, tracking IAE |
| `disturbance-rejection` | nominal equilibrium; pump factor 0.85 at 800, restored at 1400 | 2400 s | std 0.001 | unsafe rate, disturbance IAE, recovery time |
| `boundary-safety` | all levels start at 90% of their hard maximum | 1200 s | std 0.001 | unsafe rate, time to violation, tracking IAE |

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    benchmark="tracking",
)
result = aiogym.evaluate(env=env, policy="pid", seeds=[0, 1, 2])
env.close()
```

The disturbance benchmark explicitly restores the changed process factor; it
therefore measures both rejection and recovery. The boundary benchmark starts
inside the legal state space and measures whether the policy remains safe.
The tracking liquid-level step covers 45% of the 0--0.4 m physical range. Its
temperature step is a heating process between two feasible 5 L/min equilibria
rather than a step across the 0--100 degC hard state bounds: with the installed
2 kW heater, most of that hard range is not a reachable steady operating range.

## SAC training

Direct five-dimensional physical-action SAC did not pass the short-run safety
gate. The supported experimental route therefore keeps the physical plant
interface unchanged internally and explicitly trains a two-dimensional residual
over V34 and H1. Upstream hydraulic control and steady feedforward remain owned
by the Three-Tank scenario:

```python
import json

import aiogym
from aiogym.scenarios.three_tank import Tank3ResidualWrapper

env = Tank3ResidualWrapper(
    aiogym.make_env("three_tank", reward="tank3-regulation")
)
with open("configs/three-tank-residual-sac.json", encoding="utf-8") as stream:
    algorithm_kwargs = json.load(stream)
try:
    aiogym.train(
        env=env,
        algorithm="sac",
        steps=50000,
        seed=0,
        record_every=500,
        algorithm_kwargs=algorithm_kwargs,
        output="checkpoints/three-tank/sac-residual-optimized-50k/seed-0",
    )
finally:
    env.close()
```

The profile uses a long discount because downstream thermal effects take
hundreds of seconds, and a low fixed entropy coefficient selected by a safety
gate. This is a hybrid residual controller and must be reported separately from
a direct five-action SAC policy. Do not rank an early-terminated rollout by its
truncated IAE.

## Training variation

```python
env = aiogym.make_env(
    "three_tank",
    reward="tank3-regulation",
    randomize=True,
    noise=True,
    delay=True,
    fault=True,
)
```

Each reset samples a feasible tracking, disturbance-rejection, or boundary
episode. Tracking samples two complete equilibria: all three levels differ by
at least 25% of the physical scale, and Tank 3 temperature differs by at least
2 degC. Process disturbances are restored within the episode. Noise, delay,
and actuator loss-of-effectiveness are sampled separately and recorded in reset
info and workflow artifacts.

`Tank3ResidualWrapper` remains available when a two-action residual policy over
V34 and H1 is required:

```python
from aiogym.scenarios.three_tank import Tank3ResidualWrapper

env = Tank3ResidualWrapper(env)
```

## Hardware

The default Scenario is simulation-only. Experimental transport, safety,
calibration, and real-log tools require an explicit import:

```python
from aiogym.experimental import three_tank_hardware
```

The experimental hardware environment uses the fixed tracking benchmark
protocol and resolves the complete six-output reference to its steady physical
action. It records `benchmark_id="tracking"`. Closed-loop operation still
requires measured calibration, explicit arming, and residual authority.
