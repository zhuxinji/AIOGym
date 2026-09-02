# Heated Cascade model

The built-in `cascade` Scenario starts from the same three 45 L process tanks,
180 L source/return reservoir, P101 pump, and V12/V23/V34 hydraulic path as
`three_tank`. The equipment design contains one independently commanded 2 kW
immersion heater in each process tank: H1, H2, and H3. Heater availability is
configured when the environment is created and defaults to H1 only:

```python
import aiogym

env = aiogym.make_env("cascade", heater=[1, 0, 0])
```

Each entry must be binary. Other combinations such as `[1, 0, 1]` work with
ordinary, randomized, and Benchmark environments. Choose `cascade` for coupled
level and temperature control; choose `three_tank` for the hydraulic-only task.

The simulated physical state includes the three process tanks and the common
return reservoir:

```text
[h1, T1, h2, T2, h3, T3, reservoir_volume, reservoir_temperature]
```

Pass all eight values through `initial_state=` to start an ordinary simulation
from measured or chosen physical conditions:

```python
import aiogym

env = aiogym.make_env(
    "cascade",
    heater=[1, 0, 0],
    initial_state=[
        0.20, 25.0,
        0.25, 28.0,
        0.30, 32.0,
        0.09, 20.0,
    ],
)
observation, info = env.reset(seed=0)
print(info["reservoir_volume_m3"], info["reservoir_temperature"])
env.close()
```

This overrides only the initial state. The initial action, target, horizon, and
disturbances remain unchanged, so a chosen state is not assumed to be a steady
state. An explicitly supplied state may represent a cold start. Randomized
training environments sample their own initial state, and fixed Benchmarks own
their complete episodes; neither accepts `initial_state=`.

The controlled output and reference group the three levels followed by the
three temperatures:

```text
[h1, h2, h3, T1, T2, T3]
```

The direct physical action is:

```text
[P101, V12, V23, V34, H1, H2, H3]
```

In the ordinary environment, every action is a normalized physical command in
`[0, 1]`. There is no hidden lower-level controller or shared heater command.
The interface remains seven-dimensional for every heater configuration.
Commands sent to unavailable heaters remain visible as commanded actions, while
their applied actions, delivered heat, and energy use are forced to zero. The
resolved heater list is recorded with environment and checkpoint metadata.
The direct policy observation contains six normalized state measurements,
four normalized flow measurements, and six normalized references:

```text
[h1, T1, h2, T2, h3, T3, FT101, FT12, FT23, FT34,
 h1_sp, h2_sp, h3_sp, T1_sp, T2_sp, T3_sp]
```

All channels are normalized to `[0, 1]`; the four shared flowmeter channels use
the confirmed `0--10 L/min` span. Every default, randomized, and
Benchmark mode therefore keeps the same 16-dimensional observation and
7-dimensional physical action interface. Reservoir volume and temperature are
internal plant states, not policy observation channels or controlled outputs.
They are available in `info["reservoir_volume_m3"]` and
`info["reservoir_temperature"]` for inspection and saved trajectories. Built-in
PID, MPC, and learned policies do not receive the true hidden values through
policy context. Ambient temperature remains an external condition reported in
`info["disturbance"]`.

## Physical balances and safety

The hydraulic balances, vessel geometry, passive overflow, high-level pump
trip, and reservoir availability permissive match `three_tank`. P101 feeds
Tank 1, V12 carries mixed liquid from Tank 1 to Tank 2, V23 carries it to Tank
3, and V34 returns it to the reservoir. Passive overflow returns to the
reservoir rather than bypassing into the next process tank.
BV12, BV23, and BV34 provide binary parallel bypass disturbances around the
three outlet branches. Their positions are hidden conditions, while FT101,
FT12, FT23, and FT34 remain available to every policy. The three outlet meters
sit on the regulating-valve branches before the bypasses rejoin, so their
readings exclude bypass flow; policies observe each bypass only indirectly
through the liquid-level and temperature response.

Each thermal balance includes inlet mixing, its own electrical heater, and
tank-to-ambient heat loss. Tank 3 and all passive tank overflows return liquid
and energy to the reservoir; P101 draws its inlet from the reservoir. The
reservoir has a maximum capacity of `180 L`, starts with `90 L`, starts at the
ambient temperature of `20 degC`, and loses heat to the same ambient. It has no
automatic makeup, drain, or overflow path. Emptying it or exceeding its
capacity is therefore a terminal safety violation instead of silently adding
or discarding liquid. Its default heat-loss coefficient is `40 W/K`, the same
as each tank. Heater electrical power is multiplied by its tank-specific
efficiency disturbance before entering the liquid balance.

Each heater has two local hardwired permissives:

- it is disabled below `0.10 m` liquid level;
- it is disabled at or above the `80 degC` temperature trip.

An episode terminates when any tank level becomes negative or exceeds `0.50 m`,
when the reservoir inventory leaves `0--180 L`, or when any tank or reservoir
temperature becomes negative or reaches the `90 degC` hard limit.
The temperature trip prevents additional heating but is not itself a terminal
condition. `step()` reports actual heater electrical power, heat transferred to
liquid, and the per-heater interlock state.

The default episode starts at `0.225 m` in every tank with a common `3 L/min`
flow. All three tank temperatures, the hidden reservoir temperature, and the
controller bias come from the same closed recirculating steady state. The
deterministic default episode immediately tracks
`(0.30, 0.25, 0.325) m` and a reachable heated temperature profile for 1200
seconds. Its target reservoir inventory is chosen by liquid conservation, so a
level change does not assume external makeup water.

Use the public parameter interface for direct validated overrides:

```python
import aiogym

print(aiogym.list_parameters("cascade"))
```

Thermal parameters include the three heater powers, three heat-loss
coefficients, reservoir capacity, initial reservoir inventory, reservoir heat
loss, ambient temperature, heater low-level permissive, temperature trip and
hard limit, and the small positive level floor used only to keep the
variable-volume energy balance finite near an empty vessel.

## Reward, training variation, and Benchmarks

`cascade` exposes one `regulation` Reward. It scores all six controlled outputs
with fixed error scales `(0.1, 0.1, 0.1) m` and `(5, 5, 5) degC`. Formal metrics
use the same scales for IAE, ISE, and final error. Settling and recovery instead
use direct physical acceptance bands: `0.005 m` for each level and `0.2 degC`
for each temperature. A safety termination adds the explicit Reward safety
penalty and charges the remaining episode time, so stopping early through an
unsafe transition cannot avoid the remaining tracking cost.

```python
import aiogym

print(aiogym.list_rewards("cascade"))
print(aiogym.list_benchmarks("cascade"))
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | complete eight-state steady start and a different feasible six-output target at one common `2--6 L/min` flow | 4200 s | unsafe rate, settling rate, cumulative return |
| `disturbance-rejection` | default operating point with simultaneous pump, available-heater efficiency, and ambient-temperature changes, followed by explicit restoration | 2000 s | unsafe rate, settling rate, cumulative return |
| `boundary-safety` | normal state is forward pre-run with a high pump, staged low outlet flow, and `80--95%` duties on available heaters until one tank reaches a case-seeded high-level boundary | 1200 s | unsafe rate, settling rate, cumulative return |

Tracking cases sample every initial and target level in `0.125--0.40 m` and
keep hydraulic and available-heater steady actions inside `0.02--0.85`;
unavailable heater actions are exactly zero. Every level target moves by at
least `0.05 m`. A tank temperature moves by at least `1 degC` when at least
one available upstream heater can influence it; otherwise that temperature
remains at its reachable passive equilibrium. Temperature profiles are
generated from independently sampled `5--80%` duties on available heaters, so
the reachable envelope automatically widens at low flow and narrows at high
flow. Each target-minus-start temperature move stays within `-4.5--6.5 degC`
for Tank 1, `-3.5--5 degC` for Tank 2, and `-3--3.5 degC` for Tank 3.

Randomized training retains 1200-second episodes, now represented by 600 control
steps, so a fixed physical-time budget sees more operating points; the formal
tracking Benchmark runs for 4200 seconds, represented by 2100 control steps,
to judge full physical settling. Both start from complete hydraulic and thermal
equilibria. By default every randomized start is interior;
setting `boundary_probability=p` makes fraction `p` begin from the same
forward-pre-run high-level family used by the boundary Benchmark. With
`disturbance=True`, one event reduces pump capacity and the efficiency of
every available heater and lowers ambient temperature, then explicitly
restores all defaults. Noise, delay, and actuator faults remain independent
standard policy-channel variations.

Cascade ranks mean unsafe rate first, then the fraction of cases that enter and
remain inside all six physical settling bands, and only then median cumulative
return. A controller that settles more formal cases therefore ranks ahead of an
equally safe controller with a slightly better return.

## Quick start and controllers

The built-in PI controller resolves the target's analytic steady pump, valve,
and heater actions as feedforward. P101 remains at that target-flow command,
while V12, V23, and V34 apply the liquid-level feedback; this prevents redundant
level integrators from moving the process to a different circulation flow.
Because the default H1 heater affects every downstream tank, its feedback
combines the T1/T2/T3 errors with weights `0.75/0.15/0.10`; the weights sum to
one, so this does not increase the total proportional or integral gain. H2 and
H3 retain their local temperature loops when those heaters are installed. The
model masks unavailable loops before their commands reach the plant. When a
public reference changes, the controller resolves the new feedforward and
clears the previous integral; feedback then corrects measured error around that
operating point.
The successive-linearization MPC uses the eight-state plant model and seven
physical inputs. Because the two reservoir states are intentionally hidden, it
uses a nominal inventory and an inlet-temperature estimate derived from the
measured return temperature and flow; it does not read the simulator's true
reservoir state. It initializes from the reachable analytic steady input
whenever the target or relevant disturbance changes and uses a unit
steady-input regularization weight on every actuator.
It updates every 2 seconds and keeps a 60-second physical prediction window
with 30 prediction steps. Its normalized output weights are `3/3/3` for the
three levels and `2/2.5/4` for the three temperatures. Move suppression remains
`50` for the pump and valves and `2` for each heater; larger values left
persistent residuals, while smaller values increased action variation.

```python
import aiogym

env = aiogym.make_env("cascade", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
result = aiogym.evaluate(env=env, policy=pid, seeds=range(20))
env.close()
```

## Optional heater-only learning

Cascade also provides an explicit hybrid experiment in which the Three-Tank PI
controller operates P101 and the three valves while a learned policy commands
only H1/H2/H3. This is an opt-in Cascade interface, not a training variation of
the direct seven-action environment:

```python
import aiogym
from aiogym.scenarios.cascade import three_tank_pid_temperature_control

direct_env = aiogym.make_env("cascade", randomize=True)
heater_env = three_tank_pid_temperature_control(direct_env)
observation, info = heater_env.reset(seed=0)
print(observation.shape, heater_env.action_space.shape)
heater_env.close()
```

`three_tank_pid_heater_control()` exposes the 16 direct observations plus the
four hydraulic PI commands and accepts three heater actions. The temperature
training form above additionally appends the three normalized temperature
errors and uses a temperature-focused reward, giving 23 observations and three
actions.

A checkpoint trained on either wrapped interface must be loaded against that
same interface. For comparison on the ordinary seven-action Cascade Benchmark,
wrap the loaded heater policy with `as_hybrid_physical_policy()` from the same
module. Pass `temperature_error_observation=True` when the checkpoint used the
temperature training form. The adapter combines the PI hydraulic action and
learned heater action before the direct environment receives it.

## RL training

Collection and training use the same Python workflow as the other scenarios.
For RLPD, collect a fresh 100-episode randomized MPC Dataset: this contains
60,000 transitions at the 2-second interval. Checkpoints and Datasets recorded
with another control interval are rejected instead of being mixed with the
current scenario. The recommended SAC run uses the physical-time-equivalent
budget and the shared
[best-checkpoint selection](../architecture.md#select-the-best-checkpoint):

```python
import json
from pathlib import Path

import aiogym

data_env = aiogym.make_env("cascade", randomize=True)
mpc = aiogym.make_controller("mpc", env=data_env)
aiogym.collect(
    env=data_env,
    policy=mpc,
    episodes=100,
    seed=2_000,
    output="runs/cascade/datasets/randomized-mpc-100/seed-2000",
)
data_env.close()

sac_settings = json.loads(
    Path("aiogym/rl/configs/cascade-sac-nstep5.json").read_text()
)
training_env = aiogym.make_env("cascade", randomize=True)
validation_env = aiogym.make_env("cascade", randomize=True)
aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=150_000,
    seed=0,
    algorithm_kwargs=sac_settings,
    record_every=500,
    evaluation_env=validation_env,
    evaluate_every=5_000,
    output=(
        "runs/cascade/training/sac/"
        "randomized-regulation-sac-nstep5-gamma99900025-150k/seed-0"
    ),
)
training_env.close()
validation_env.close()
```

The Cascade SAC configuration uses `gamma=0.99900025` and five-step returns,
preserving the previous physical discount and 10-second return windows at the
2-second control interval. Direct policies use
16 observations, including the four physical flow measurements. Train and
evaluate direct checkpoints on this interface with the same ordered Benchmark
seeds used for other policies.
