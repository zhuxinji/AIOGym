# Three-Tank model

The built-in `three_tank` Scenario represents one fixed laboratory process:

- three identical 45 L process tanks (`0.3 x 0.3 x 0.5 m`);
- one separate, well-mixed 180 L source/return reservoir;
- pump P101;
- valves V12, V23, and V34.

The process tanks use a `0.09 m^2` cross-section and a `0.5 m` hard level
limit. Their operating points are a `0.225 m` nominal level, `0.425 m` pump
high-level trip, and `0.45 m` overflow level. The model uses the declared pump,
three valves, and passive overflows.

The complete physical state and controlled output are both
`[h1, h2, h3]`. The physical action is `[P101, V12, V23, V34]`, with every value
normalized to `[0, 1]`. All modes keep the same 6-dimensional observation and
4-dimensional physical-action interface. The observation is
`[h1, h2, h3, h1_sp, h2_sp, h3_sp]`, normalized by physical bounds. Temperature
is not part of the state, observation, output, reference, action, reward, or
safety contract. The previous action is not part of the observation.

V34 and passive overflow return water to the constant-level source reservoir,
and P101 draws from that reservoir. Reservoir inventory and temperature are not
dynamic states in this hydraulic task.

Three-Tank parameters use hydraulic SI units (`m`, `m^2`, and `m^3/s`), plus
`W` for the pump energy estimate. Use `list_parameters()` for every override.

## Default episode, Rewards, and Benchmarks

The deterministic default is the tracking Benchmark's resolved seed-0 case.
It starts from one feasible three-state equilibrium, exposes the seed-0 target
at reset, and has a 600 s horizon. Both ends use the same sampled equilibrium
flow and have unsaturated steady physical actions. The training environment
remains non-Benchmark even though its fixed physical Episode equals tracking
case seed 0; it is not a named public selector. With the default model, seed 0
resolves the initial levels to `(0.365792, 0.253281, 0.371840) m`, the target to
`(0.143198, 0.318803, 0.186675) m`, and the shared equilibrium flow to
`1.840675 L/min`.

Three-Tank exposes one `regulation` Reward. It tracks only `[h1, h2, h3]`,
using a `0.1 m` error scale for each level. Return, IAE, ISE, ITAE, overshoot,
final error, and settling time all use this same three-level objective.

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
| `tracking` | `regulation` | case-seeded feasible equilibrium start that immediately tracks one three-level target | 600 s | unsafe rate, cumulative return |
| `disturbance-rejection` | `regulation` | nominal equilibrium with case-seeded hydraulic-capacity disturbances | 1800 s | unsafe rate, cumulative return |
| `boundary-safety` | `regulation` | all three levels start at independently sampled `86--90%` levels | 600 s | unsafe rate, cumulative return |

These horizons are fixed transient-comparison windows, not claims that every
case reaches steady state. With `0.09 m^2` tank area, the tracking level range
contains about `11.25--36 L` per tank. The `1--6 L/min` equilibrium-flow range
therefore spans approximate single-tank hydraulic residence times of
`112.5--2160 s`, before cascade effects. The 600-second tracking window is a
fixed controller-transient comparison rather than a full residence-time window
for the slowest low-flow case. A reported settling time equal to the horizon means the strict
three-level settling band was not maintained inside this comparison window; it
does not by itself invalidate the horizon or prove eventual non-convergence.

All Benchmarks use exact observations without measurement noise. For Three-Tank
tracking, each non-negative evaluation seed selects one reproducible operating
case. Reset samples one episode-wide flow in `1–6 L/min`, three initial levels,
and one target in `0.125–0.4 m`. Tracking begins on the first control step. The
absolute change of every level is at least `0.05 m`; there is no additional
move cap inside the sampled level range. Both sampled equilibria must keep every
physical action in `0.02–0.85`.

Disturbance cases apply a hydraulic-capacity loss beginning at step `600--850`
for `400--650` steps: pump factor is sampled in `0.40--0.70` and V23 factor in
`0.50--0.70`. Boundary cases independently sample all three initial levels in
`86--90%` of their hard maximum. Seeds `0--19` resolve to 20 distinct Episodes
for all three Benchmarks.

The latest possible hydraulic recovery event is at 1500 seconds. The
1800-second disturbance horizon retains a 300-second tail; both controllers
stayed inside the strict tracking band after the event. Boundary cases settled
within 252 seconds, so 600 seconds retains more than twice the observed worst
settling time.

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    benchmark="tracking",
)
result = aiogym.evaluate(env=env, policy="pid", seeds=range(20))
env.close()
```

The tracking benchmark resolves the start and three-level target once during
reset; it does not use a precomputed case table or an in-episode reference
event. The resolved `EpisodeSpec`, initial steady
action, and case seed are stored in evaluation output. The disturbance
benchmark applies and explicitly restores hydraulic-capacity losses; the
common `regulation` Reward measures hydraulic rejection and recovery.
The boundary benchmark starts inside the legal state space and measures whether
the policy remains safe.

The built-in controller id remains `pid`, but the Three-Tank configuration has
`kd=0` and is therefore a PI controller. It uses one fixed default-action bias;
it does not recompute a steady-state action from the reference or current
disturbance. The `mpc` controller is a successive-linearization MPC. The
controller uses the normal model-based steady-state target.

The tuned PI configuration keeps hydraulic `kp_scale=8` and uses
`hydraulic_ki_scale=0.02`. MPC keeps a 60 s prediction horizon; its
move-suppression and steady-input weights are both `[50, 50, 50, 50]` and
`[5, 5, 5, 5]`, respectively.

## RL training

The fixed training protocol uses the tracking Benchmark's deterministic seed-0
Episode for 600 steps with exact observations: `randomize`, `disturbance`,
`noise`, `delay`, and `fault` are all disabled. A 240k budget equals 400 full-horizon
episodes when every episode reaches its time limit; safety terminations reset
the environment earlier. SAC, PPO, and DDPG use the same direct four-dimensional
physical action as PI and MPC. There is no action wrapper, internal upstream
controller, or simulator action-rate limit.

```bash
aiogym train three_tank sac --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --algorithm-kwargs aiogym/rl/configs/three-tank-sac-nstep10.json \
  --output runs/three-tank/training/regulation-sac-nstep10-240k/seed-0

aiogym train three_tank ppo --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --output runs/three-tank/training/regulation-ppo-240k/seed-0

aiogym train three_tank ddpg --steps 240000 --seed 0 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --output runs/three-tank/training/regulation-ddpg-240k/seed-0

aiogym train three_tank sac --steps 60000 --record-every 1000 \
  --evaluate-every 10000 --evaluation-seed 0 \
  --resume-from runs/three-tank/training/regulation-sac-nstep10-240k/seed-0/model.zip \
  --output runs/three-tank/training/regulation-sac-nstep10-300k/seed-0
```

The SAC configuration uses `n_steps=10`; PPO and DDPG use their Stable-Baselines3 defaults.
The continuation command adds 60,000 steps to the checkpoint's 240,000-step
state; it restores SAC optimizer and replay state and writes a separate run.
Observation, reference, and action shapes are six, three, and four. These
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

Each reset samples a new 600-second operating condition for one tracking
Episode. 80% of episodes start from a feasible interior steady state and
immediately track a sampled target. 20% start with all three levels at
`80–92%` of their hard upper bounds and immediately track a feasible interior target. Interior steady
states use levels in `0.125–0.4 m` and flow in `1–6 L/min`; each interior level
changes by at least `0.05 m`, without an additional move cap. Boundary-start
episodes independently sample their actual high-level state and therefore have
no initial-to-target move cap. `disturbance=True`
independently
multiplies pump flow by `0.72–0.94`, beginning at step `112–200` for `75–150`
steps, and then restores the factor to `1.0`. The disturbance changes the true
dynamics but is not a direct policy-observation channel. Noise, delay, and
actuator loss-of-effectiveness remain separate optional channels. The fixed
disturbance-rejection and boundary-safety Benchmarks are evaluation-only
protocols.

## Current controller comparison

The AIO-Gym 0.15 hydraulic-only PI/MPC comparison uses case seeds `0--19` for
every fixed Benchmark. Both controllers completed all 120 controller-case
rollouts without a safety violation. Values below are medians across 20 cases;
safe completion is the mean of the per-case indicator.

| Benchmark | Controller | Safe completion | Return | Tracking IAE | Final error | Settling time [s] |
|---|---|---:|---:|---:|---:|---:|
| `tracking` | PI | 1.00 | -33.625 | 132.935 | 4.38e-3 | 96.0 |
| `tracking` | MPC | 1.00 | -39.954 | 167.738 | 7.00e-7 | 189.5 |
| `disturbance-rejection` | PI | 1.00 | -0.0204 | 11.737 | 2.18e-3 | 0.0 |
| `disturbance-rejection` | MPC | 1.00 | -4.01e-13 | 8.02e-5 | 1.62e-8 | 0.0 |
| `boundary-safety` | PI | 1.00 | -217.388 | 480.873 | 7.50e-3 | 183.0 |
| `boundary-safety` | MPC | 1.00 | -215.323 | 488.753 | 3.46e-6 | 249.0 |

The fixed ranking is safety first and cumulative return second. It ranks PI
first for `tracking`, and MPC first for `disturbance-rejection` and
`boundary-safety`. The machine-readable results, comparison plots, and full
trajectory archives are stored under `runs/three-tank/benchmarks/<benchmark>/`.
The tracking artifact was regenerated after removing the sampled `0.12 m`
move cap; disturbance rejection and boundary safety do not use that sampler.

## Hardware

The default Scenario is simulation-only. Experimental transport, safety,
calibration, and real-log tools require an explicit import:

```python
from aiogym.experimental import three_tank_hardware
```

The experimental hardware environment resolves tracking case seed zero at
construction and accepts the same direct four-dimensional physical action as
the simulator. LT101, LT201, and LT301 provide the three measurements required
for the 6-dimensional policy observation. It records
`benchmark_id="tracking"`. Closed-loop operation requires measured calibration
and explicit arming.
