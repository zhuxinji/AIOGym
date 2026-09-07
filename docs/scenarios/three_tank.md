# Three-Tank model

The built-in `three_tank` Scenario represents one fixed laboratory process:

- three identical 45 L process tanks (`0.3 x 0.3 x 0.5 m`);
- one separate, well-mixed 180 L source/return reservoir;
- pump P101;
- valves V12, V23, and V34;
- on/off bypasses BV12, BV23, and BV34;
- flowmeters FT101, FT12, FT23, and FT34.

The process tanks use a `0.09 m^2` cross-section and a `0.5 m` hard level
limit. Their operating points are a `0.225 m` nominal level, `0.425 m` pump
high-level trip, and `0.45 m` overflow level. The model uses the declared pump,
three valves, and passive overflows.

The complete physical state and controlled output are both
`[h1, h2, h3]`. The physical action is `[P101, V12, V23, V34]`, with every value
normalized to `[0, 1]`. All modes keep the same 10-dimensional observation and
4-dimensional physical-action interface. The observation is
`[h1, h2, h3, FT101, FT12, FT23, FT34, h1_sp, h2_sp, h3_sp]`, normalized by physical
bounds and the installed `0--10 L/min` flowmeter span. Formal tracking cases
keep their common circulation inside `3--8 L/min`, leaving measurement headroom
at both ends of the instrument range. This
scenario focuses only on liquid levels; use `cascade` when temperature control
is required. The previous action is not part of the observation.

BV12, BV23, and BV34 are binary physical disturbances, not policy actions.
Opening a bypass adds a parallel path around its regulating-valve branch. The
default equivalent coefficient is `4e-5 m^(5/2)/s`, which adds about
`1.74 L/min` at the `0.225 m` nominal level and `1.56--2.01 L/min` across the
formal `0.125--0.4 m` level range. At `3--8 L/min` circulation this is about
`20--67%` of total branch flow, while retaining a positive regulating-valve
flow at every sampled target. The switch position remains hidden
in `info["disturbance"]`. FT12, FT23, and FT34 sit on the regulating-valve
branches before the bypasses rejoin, so they measure only V12, V23, and V34
flow and do not directly reveal bypass flow. FT101 separately measures the
P101 feed flow. Controllers can infer an open bypass only indirectly from the
level response and the resulting flow imbalance.

V34 and passive overflow return water to the constant-level source reservoir,
and P101 draws from that reservoir. Reservoir inventory and temperature are not
dynamic states in this hydraulic task.

Three-Tank parameters use hydraulic SI units (`m`, `m^2`, and `m^3/s`), plus
`W` for the pump energy estimate. Use `list_parameters()` for every override.
The simulator's effective P101 ceiling is `25 L/min`. Across the formal level
range, a fully open regulating valve passes about `19.6--25.1 L/min`, or
`16.6--21.3 L/min` at the formal `0.85` action limit. These are model capacity
limits rather than the operating protocol: the installed flowmeters span
`0--10 L/min`, so formal tracking uses `3--8 L/min` and retains 20% upper
measurement headroom.

## Default episode, Rewards, and Benchmarks

The deterministic default is the tracking Benchmark's resolved seed-0 case.
It starts from one feasible three-state equilibrium, exposes the seed-0 target
at reset, and has a 600 s horizon. Both ends use the same sampled equilibrium
flow and have unsaturated steady physical actions. The training environment
remains non-Benchmark even though its fixed physical episode equals tracking
case seed 0; it is not a named public selector. With the default model, seed 0
resolves the initial levels to `(0.365792, 0.253281, 0.371840) m`, the target to
`(0.143198, 0.318803, 0.186675) m`, and the shared equilibrium flow to
`3.840675 L/min`.

Three-Tank exposes one `regulation` Reward. It tracks only `[h1, h2, h3]`,
using a `0.1 m` error scale for each level. Return, IAE, ISE, ITAE, overshoot,
final error, and settling time all use this same three-level objective.

The Reward also discourages actuator chatter. At each control step it subtracts
the mean squared change of the four normalized pump and valve commands from the
previous applied action. Holding an action adds no movement cost; a change of
`0.1` on one actuator adds `0.0025` at the one-second control interval. It does
not penalize the steady action needed to keep water circulating.

The Reward includes an explicit terminal safety-violation penalty.
Every formal Benchmark declares its fixed Reward and uses default model
parameters; `reward=` and `parameters=` are rejected when `benchmark=` is set.
State, output/reference, observation, physical action, and safety constraints
retain common dimensions and meanings across all fixed protocols.

```python
import aiogym

print(aiogym.list_benchmarks("three_tank"))
print(aiogym.list_rewards("three_tank"))
print(aiogym.list_parameters("three_tank"))
```

| Benchmark | Reward | Fixed protocol | Horizon | Ranking |
|---|---|---|---:|---|
| `tracking` | `regulation` | case-seeded feasible equilibrium start that immediately tracks one three-level target | 600 s | unsafe rate, cumulative return |
| `disturbance-rejection` | `regulation` | track the paired target from reset while a case-seeded single/pair/triple bypass combination opens temporarily | 600 s | unsafe rate, cumulative return |
| `boundary-safety` | `regulation` | the normal state is forward pre-run with legal pump/valve commands until one tank reaches a case-seeded `82--84.8%` high-level boundary | 600 s | unsafe rate, cumulative return |

These horizons are fixed transient-comparison windows, not claims that every
case reaches steady state. With `0.09 m^2` tank area, the tracking level range
contains about `11.25--36 L` per tank. The `3--8 L/min` equilibrium-flow range
therefore spans approximate single-tank hydraulic residence times of
`84.4--720 s`, before cascade effects. The 600-second tracking window is a
fixed controller-transient comparison rather than a full residence-time window
for the slowest low-flow case. A reported settling time equal to the horizon means the strict
three-level settling band was not maintained inside this comparison window; it
does not by itself invalidate the horizon or prove eventual non-convergence.

All Benchmarks use exact observations without measurement noise. For Three-Tank
tracking, each non-negative evaluation seed selects one reproducible operating
case. Reset samples one episode-wide flow in `3–8 L/min`, three initial levels,
and one target in `0.125–0.4 m`. Tracking begins on the first control step. The
absolute change of every level is at least `0.05 m`; there is no additional
move cap inside the sampled level range. Both sampled equilibria must keep every
physical action in `0.02–0.85`.

For the same seed, disturbance cases use the same initial equilibrium, target,
and flow as tracking cases. The target is active from reset, so the only
difference from tracking is the temporary bypass schedule. A case-seeded event
between steps `30--90` opens one bypass mode for 360 seconds, after which every
selected bypass closes. This leaves `150--210` seconds of post-close tracking.
Single mode opens one of BV12, BV23, or BV34; simultaneous-pair
mode selects two distinct branches; simultaneous-triple mode opens all three.
Mode
probabilities are `30%` single, `40%`
simultaneous pair, and `30%` simultaneous triple. Seeds `0--19` resolve to `6`,
`9`, and `5` cases, respectively. Those cases cover every individual branch,
all three unordered simultaneous pairs, and five all-three-open cases. Pump and
valve capacity factors are not changed by this Benchmark. Boundary cases pre-run one of three
legal pump/valve patterns
from the normal state until the selected tank reaches `82--84.8%` of its hard
maximum. Seeds `0--19` select 20 distinct episodes for all three
Benchmarks. The Benchmark uses the ordinary full-episode tracking, endpoint,
settling, return, and safety metrics; the recorded schedule and trajectories
remain available for diagnostic phase analysis when needed.

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    benchmark="tracking",
)
result = aiogym.evaluate(env=env, policy="pid", seeds=range(20))
env.close()
```

The tracking benchmark selects the start and three-level target during reset;
the selected case, initial steady action, and seed are stored in the evaluation
output. The disturbance benchmark uses the matching tracking case's initial state and
action, with its target reference active from reset. It opens the selected
bypass path or paths and later closes them while keeping that reference fixed.
The common `regulation` metrics cover the episode; disturbance error starts at
the first disturbance event and recovery time starts at the final restoration.
The boundary benchmark starts from the last safe state of a deterministic
forward pre-run and measures whether the policy remains safe. It does not
construct or test a boundary equilibrium.

The built-in controller id remains `pid`, but the Three-Tank configuration has
`kd=0` and is therefore a PI controller. It uses one fixed default-action bias;
it does not recompute a steady-state action from the reference or current
disturbance. The `mpc` controller is a successive-linearization MPC. The
controller uses the normal model-based steady-state target. Both controllers
receive the same ten-channel policy observation as a learned policy. Scheduled
bypass positions remain hidden simulator conditions recorded in
`info["disturbance"]`; MPC does not receive those exact positions or the future
disturbance schedule. Pump and valve flow-capacity factors remain available in
the physical model but are not scheduled by this Benchmark.

The tuned PI configuration keeps hydraulic `kp_scale=8` and uses
`hydraulic_ki_scale=0.02`. MPC keeps a 60 s prediction horizon; its
move-suppression and steady-input weights are both `[50, 50, 50, 50]` and
`[5, 5, 5, 5]`, respectively.

## RL training

The recommended training protocol samples feasible randomized 600-second
tracking episodes with exact observations. Periodic evaluation follows the
shared [best-checkpoint selection](../architecture.md#select-the-best-checkpoint)
on a separate randomized environment. A 240k budget equals 400 full-horizon
episodes when every episode reaches its time limit; safety terminations reset
the environment earlier. SAC, PPO, and DDPG use the same direct
four-dimensional physical action as PI and MPC. There is no action wrapper,
hidden controller, or simulator action-rate limit.

```python
import json
from pathlib import Path

import aiogym

sac_settings = json.loads(
    Path("aiogym/rl/configs/three-tank-sac-nstep10.json").read_text()
)
training_env = aiogym.make_env("three_tank", randomize=True)
evaluation_env = aiogym.make_env("three_tank", randomize=True)
trained = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=240_000,
    seed=0,
    algorithm_kwargs=sac_settings,
    record_every=1_000,
    evaluation_env=evaluation_env,
    evaluate_every=10_000,
    output=(
        "runs/three-tank/training/sac/"
        "randomized-regulation-sac-nstep10-240k/seed-0"
    ),
)
training_env.close()
evaluation_env.close()

continued_env = aiogym.make_env("three_tank", randomize=True)
continued_evaluation_env = aiogym.make_env("three_tank", randomize=True)
continued = aiogym.train(
    env=continued_env,
    algorithm="sac",
    steps=60_000,
    record_every=1_000,
    evaluation_env=continued_evaluation_env,
    resume_from=(
        "runs/three-tank/training/sac/"
        "randomized-regulation-sac-nstep10-240k/seed-0/model.zip"
    ),
    output=(
        "runs/three-tank/training/sac/"
        "randomized-regulation-sac-nstep10-300k/seed-0"
    ),
)
continued_env.close()
continued_evaluation_env.close()
```

The SAC configuration uses `n_steps=10`. Set `algorithm="ppo"` or
`algorithm="ddpg"` and omit `algorithm_kwargs` to use their Stable-Baselines3
defaults. The continuation call adds 60,000 steps to the checkpoint's 240,000-step
state; it restores SAC optimizer and replay state and writes a separate run.
Observation, reference, and action shapes are ten, three, and four. These
single-seed runs are baselines, not a statistical RL comparison.
Use the selected `best/model.zip` in formal Benchmarks.

On a safety termination, the regulation reward charges `2.0` for every
remaining physical second in the episode before the environment applies its
fixed `100.0` safety penalty. This preserves every safe full-horizon return but
prevents a short unsafe episode from avoiding the remaining tracking cost.

## Training variation

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    randomize=True,
    boundary_probability=0.30,
    disturbance=True,
    noise=True,
    delay=True,
    fault=True,
)
```

For an ordinary simulation, a piecewise-constant disturbance schedule can open
and close specific bypasses directly:

```python
import aiogym

env = aiogym.make_env(
    "three_tank",
    disturbance_schedule={
        60: {"bv23_open": 1.0},
        420: {"bv23_open": 0.0},
    },
)
```

Each scheduled value persists until another event changes it. Step `0` sets a
non-default initial value. A custom schedule may accompany `randomize=True`,
but cannot be combined with `disturbance=True` or a fixed Benchmark.

Each reset samples a new 600-second operating condition for one tracking
episode. By default every episode starts from a feasible interior steady state
and immediately tracks a sampled target. Setting `boundary_probability=p`
makes fraction `p` start with all three levels at `80–92%` of their hard upper
bounds and immediately track a feasible interior target. Interior steady
states use levels in `0.125–0.4 m` and flow in `3–8 L/min`; each interior level
changes by at least `0.05 m`, without an additional move cap. Boundary-start
episodes independently sample their actual high-level state and therefore have
no initial-to-target move cap. `disturbance=True` opens random single, pair, or
triple combinations of BV12, BV23, and BV34 at step `30--90`, then closes the
same combination exactly 360 seconds later. Mode probabilities and timing
match the fixed Benchmark distribution at `30%`, `40%`, and `30%`. Training
draws new episodes while Benchmark seeds `0--19` freeze reproducible held-out
cases. The bypass position changes
the true dynamics but is not a direct policy-observation channel. Noise, delay,
and `fault=True` action-channel loss-of-effectiveness remain separate optional
channels. Pump and valve flow-capacity factors are not sampled by
`disturbance=True`. No fixed fault-recovery Benchmark is currently registered.
The fixed disturbance-rejection and boundary-safety Benchmarks are
evaluation-only protocols.

Current PI, MPC, and learned-policy scores remain in the generated
`runs/three-tank/benchmarks/<benchmark>/comparison.json` files. Each
disturbance case shares its initial equilibrium and target with the tracking
case carrying the same seed, so their full-episode results can be compared
directly.

## Hardware

The default Scenario is simulation-only. Experimental transport, safety,
calibration, and real-log tools require an explicit import:

```python
from aiogym.experimental import three_tank_hardware
```

The experimental hardware environment resolves tracking case seed zero at
construction and accepts the same direct four-dimensional physical action as
the simulator. LT101, LT201, LT301, FT101, FT12, FT23, and FT34 provide the
seven measured channels required for the 10-dimensional policy observation. It records
`benchmark_id="tracking"`. Closed-loop operation requires measured calibration
and explicit arming.

Calibration currently provides an admission check and provenance: its ID and
hash are written to real logs. Parameter estimates are not applied to the
nominal model, reference reachability calculations, or channel conversion.
Those model calculations still use the default simulation parameters. The
injected transport must supply already calibrated engineering-unit samples;
it owns raw-channel conversion and must be configured separately with the
rig's measured channel calibration. Passing a calibration record to this
environment does not configure the transport or calibrate the simulator.
