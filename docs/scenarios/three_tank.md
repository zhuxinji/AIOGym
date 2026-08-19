# Three-Tank model

The built-in `three_tank` Scenario represents one fixed laboratory process:

- three identical 45 L process tanks (`0.3 x 0.3 x 0.5 m`);
- one separate, well-mixed 180 L source/return reservoir;
- pump P101;
- valves V12, V23, and V34;
- one 2 kW heater H1 in Tank 1.

The 45 L redesign uses a `0.09 m^2` tank cross-section and a `0.5 m` hard
level limit. Its level design points preserve the previous relative operating
range: `0.10 m` heater trip, `0.225 m` nominal level, `0.425 m` pump high-level
trip, and `0.45 m` overflow level. The existing pump, valves, 2 kW heater, and
provisional `40 W/K` per-tank heat-loss coefficients are unchanged so geometry
can be validated independently from later equipment selection and calibration.

The first six state entries and the controlled output use the same order:
`[h1, T1, h2, T2, h3, T3]`. The internal state appends the measured reservoir
temperature `T_reservoir`, so the complete state is seven-dimensional while
the controlled output remains six-dimensional. The physical action is
`[P101, V12, V23, V34, H1]`, with every value normalized to `[0, 1]`. All modes
keep the same 13-dimensional observation and 5-dimensional physical-action
interface. The observation is
`[h1, T1, h2, T2, h3, T3, T_reservoir, h1_sp, T1_sp, h2_sp, T2_sp, h3_sp, T3_sp]`,
normalized by physical bounds. It contains the complete current plant
measurement and six controlled-output setpoints; the previous action is not
part of the observation.

V34 and passive overflow return water to the reservoir, and P101 draws from
its current temperature. The reservoir energy balance includes return mixing,
automatic makeup water when the process-tank inventory is increasing, and a
provisional ambient heat loss. Its nominal volume is `0.18 m^3` and its
provisional `UA` is `120 W/K`; both are direct model parameters. `t_makeup` is
the makeup-water temperature, not the reservoir temperature. Reservoir level
is still held constant and is not a dynamic state.

Three-Tank parameters primarily use SI units (`m`, `m^2`, `m^3/s`, `W`, and
`W/K`) with temperatures in `degC`. Use `list_parameters()` for the exact unit
of every override.

## Default episode, Rewards, and Benchmarks

The deterministic default starts from a feasible seven-state equilibrium and
applies a feasible six-output equilibrium target. It starts at the nominal
`0.225 m` equilibrium with `T3 = 22.819 degC`,
and changes at 600 s to `h3 = 0.375 m`. The temperature references remain at
their initial equilibrium values and are not part of the tracking objective.
The episode has a 2400 s horizon.
Both ends use 3 L/min and have unsaturated steady physical actions. This
training protocol is distinct from the case-seeded tracking Benchmark and is
not a named public selector.

The default `regulation` Reward tracks only `[h1, h2, h3]`, using a `0.1 m`
error scale for each level. Temperatures remain physical states, observations,
plot diagnostics, and safety variables, but temperature error does not
contribute to return, IAE, ISE, ITAE, overshoot, final error, or settling time.
The separate `thermal_regulation` Reward retains six-output tracking with
`0.1 m` level and `3 degC` temperature error scales for Benchmarks whose thermal
disturbances must remain observable in the score.

The Reward includes an explicit terminal safety-violation penalty.
Every formal Benchmark declares its fixed Reward and uses default model
parameters; `reward=` and `parameters=` are rejected when `benchmark=` is set.
State, output/reference, observation, physical action, and safety constraints
retain common dimensions and meanings across all fixed protocols.

```bash
aiogym list benchmarks --scenario three_tank
aiogym list rewards --scenario three_tank
aiogym list parameters --scenario three_tank
```

| Benchmark | Reward | Fixed protocol | Horizon | Ranking |
|---|---|---|---:|---|
| `tracking` | `regulation` | case-seeded feasible start and level target at 600 s; temperature references remain fixed and unscored | 2400 s | unsafe rate, cumulative return |
| `disturbance-rejection` | `thermal_regulation` | nominal equilibrium; pump and V23 flow factors fall to 0.50 at 700 s and recover at 1200 s, then the common heat-loss factor rises to 3.0 at 1700 s and recovers at 2300 s | 3600 s | unsafe rate, cumulative return |
| `boundary-safety` | `thermal_regulation` | all levels start at 90% of their hard maximum | 1200 s | unsafe rate, cumulative return |

All Benchmarks use exact observations without measurement noise. For Three-Tank
tracking, each non-negative evaluation seed selects one reproducible operating
case. Reset samples one episode-wide flow in `1–6 L/min`, three start levels and
three target levels in `0.125–0.4 m`, plus one constant Tank 3 operating
temperature in `20.5–26 degC`. The absolute change of each level is at least
`0.05 m`. The model then derives `T1`, `T2`, reservoir temperature, and all
five steady actions. Both equilibria must keep every action in `0.02–0.85`.

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    benchmark="tracking",
)
result = aiogym.evaluate(env=env, policy="pid", seeds=[0, 1, 2])
env.close()
```

The tracking benchmark resolves the start and level target once during reset;
it does not use a precomputed case table and does not solve a new equilibrium
when the event fires. The episode keeps one flow and one thermal operating point
across both equilibria. The resolved `EpisodeSpec`, initial steady action, and
case seed are stored in evaluation output. The disturbance benchmark applies
and explicitly restores one hydraulic-capacity loss and one heat-loss increase;
it therefore measures both rejection and recovery under `thermal_regulation`.
The boundary benchmark starts inside the legal state space and measures whether
the policy remains safe.

The built-in controller id remains `pid`, but the Three-Tank profile has
`kd=0` and is therefore a PI controller. It uses one fixed default-action bias;
it does not recompute a steady-state action from the reference or current
disturbance. The `mpc` controller is a successive-linearization MPC. The
controller uses the normal model-based steady-state target.

## RL training

The fixed training protocol uses a deterministic 2,400-step episode with exact
observations: `randomize`, `disturbance`, `noise`, `delay`, and `fault` are all
disabled. Tank 3 level moves from `0.225 m` to `0.375 m` at step 600 while all temperature
references remain fixed and unscored. A 240k budget equals 100 full-horizon
episodes when every episode reaches its time limit; safety terminations reset
the environment earlier. SAC, PPO, and DDPG use the same direct five-dimensional
physical action as PI and MPC. There is no action wrapper, internal upstream
controller, or simulator action-rate limit.

```bash
aiogym train three_tank sac --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --algorithm-kwargs rl/configs/three-tank-sac-nstep10.json \
  --output rl/checkpoints/three-tank/regulation-sac-nstep10-level-240k/seed-0

aiogym train three_tank ppo --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --output rl/checkpoints/three-tank/regulation-ppo-level-240k/seed-0

aiogym train three_tank ddpg --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --output rl/checkpoints/three-tank/regulation-ddpg-level-240k/seed-0
```

The SAC profile uses `n_steps=10`; PPO and DDPG use their Stable-Baselines3 defaults.
Observation, reference, and action shapes remain 13, six, and five. These
single-seed runs are baselines, not a statistical RL comparison.

On a safety termination, the regulation reward charges `2.0` for every
remaining physical second in the episode before the environment applies its
fixed `100.0` safety penalty. This preserves every safe full-horizon return but
prevents a short unsafe episode from avoiding the remaining tracking cost.

## Training variation

```python
env = aiogym.make_env(
    "three_tank",
    randomize=True,
    disturbance=True,
    noise=True,
    delay=True,
    fault=True,
)
```

Each reset samples a new operating condition for one tracking task. 80% of
episodes start from a feasible interior steady state and later apply a random
setpoint step. 20% start with all three levels at `80–92%` of their hard upper
bounds and immediately track a feasible interior target. Interior steady
states use levels in `0.125–0.4 m`, flow in `1–6 L/min`, and Tank 3 temperature
in `20.5–26 degC`; the other temperatures and all steady actions are derived
from the same equilibrium calculation. `disturbance=True` independently
multiplies pump flow by `0.72–0.94`, beginning at step `450–800` for `300–600`
steps, and then restores the factor to `1.0`. The disturbance changes the true
dynamics but is not a direct policy-observation channel. Noise, delay, and
actuator loss-of-effectiveness remain separate optional channels. The fixed
disturbance-rejection and boundary-safety Benchmarks are evaluation-only
protocols.

## Current controller comparison

The active `runs/three-tank/` comparisons use the current direct-action model,
exact observations, and only the built-in PI and MPC controllers. Tracking uses
ten fixed Benchmark seeds (`0–9`); disturbance rejection and boundary safety
retain seeds `0–4`. All 40 evaluated episodes completed their full horizons
with `unsafe_rate = 0`.

| Benchmark | PI median return | MPC median return | Ranking |
|---|---:|---:|---|
| `tracking` | -17.438 | -40.276 | PI, MPC |
| `disturbance-rejection` | -16.708 | approximately 0 | MPC, PI |
| `boundary-safety` | -260.437 | -329.948 | PI, MPC |

No learned policy is included in these comparison artifacts. Existing SAC
checkpoints were trained under the previous reward and episode contract and
must be retrained before a separate RL comparison is meaningful.

## Hardware

The default Scenario is simulation-only. Experimental transport, safety,
calibration, and real-log tools require an explicit import:

```python
from aiogym.experimental import three_tank_hardware
```

The experimental hardware environment resolves tracking case seed zero at
construction and accepts the same direct five-dimensional physical action as the
simulator. A reservoir temperature measurement (`TT401` in the provisional I/O
schema) is required for the 13-dimensional policy observation. It records
`benchmark_id="tracking"`. Closed-loop operation requires measured calibration
and explicit arming.
