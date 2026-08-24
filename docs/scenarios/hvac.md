# Two-zone HVAC model

The built-in `hvac` Scenario models two thermally coupled zones with one
bidirectional heat-pump command per zone. It provides the common Scenario,
Reward, Benchmark, controller, Dataset, and training workflow contracts.

The physical interface is:

- state and controlled output
  `[zone_0_temperature, zone_1_temperature]` in `degC`;
- action `[hvac_zone_0, hvac_zone_1]` in normalized `[0, 1]` coordinates;
- action `0.5` is zero delivered power, values below `0.5` cool, and values
  above `0.5` heat;
- each zone receives at most `maximum_zone_power` in either direction;
- disturbances are outdoor temperature, one scalar internal heat load per
  zone, and a common delivered-power efficiency factor.

The policy observation contains four normalized values: both temperatures and
both temperature references. State temperatures use the hard `-20--60 degC`
range; references use the `18--26 degC` operating range. The observation does
not expose disturbances directly. The controller interval is 5 seconds and the
plant is integrated internally with a maximum 0.25-second step.

The default state is `(22, 22) degC`. At the default outdoor temperature of
`5 degC`, its model-consistent steady action is `(0.7125, 0.7125)`. The
deterministic training Episode immediately tracks `(24, 20) degC` for 60
control steps, or 300 seconds. The steady-state inverse accounts for outdoor heat transfer,
inter-zone coupling, internal heat loads, and HVAC efficiency. Infeasible
references return no steady action instead of clipping the physical solution.

## Parameters and evidence boundary

The executable defaults are:

| Parameter | Default | Unit |
|---|---:|---|
| `zone_thermal_capacity` | 6000 | J/K |
| `maximum_zone_power` | 1800 | W |
| `interzone_conductance` | 35 | W/K |
| `outdoor_conductance` | 45 | W/K |
| `outdoor_temperature` | 5 | degC |

These values define a normalized simulation benchmark. They are not calibrated
to a named building or validated against measured equipment. The Scenario
therefore supports reproducible simulation and controller comparison, not a
claim of physical deployment fidelity.

## Benchmarks

```bash
aiogym list benchmarks --scenario hvac
```

| Benchmark | Fixed protocol | Horizon | Ranking |
|---|---|---:|---|
| `tracking` | case-seeded feasible equilibrium start that immediately tracks one independently sampled target | 60 steps / 300 s | unsafe rate, cumulative return |
| `disturbance-rejection` | case-seeded outdoor temperature, internal loads, efficiency, event time, and recovery time | 210 steps / 1050 s | unsafe rate, cumulative return |
| `boundary-safety` | both zones start at a case-seeded hot or cold boundary condition inside the hard limits | 60 steps / 300 s | unsafe rate, cumulative return |

Each tracking seed samples all start and target temperatures in `19--25 degC`.
Each zone target differs from its initial output by at least `1.5 degC`, and
tracking begins on the first control step. The model
derives the corresponding state and steady action, requires both actions to
remain in `0.05--0.95`, and verifies a zero model derivative before accepting
the case. Identical seeds therefore give every compared policy identical
operating conditions.

Disturbance cases begin at step `50--90` and last `40--80` steps. They sample
outdoor temperature from `-10` to `5 degC`, zone 0 heat load in `300--800 W`,
zone 1 load from `-500` to `-100 W`, and HVAC efficiency in `0.65--0.90`.
Boundary cases sample both zone temperatures either in the hot range
`48--54 degC` or from `-14` to `-8 degC`. Seeds `0--19` resolve to 20 distinct
Episodes for all three Benchmarks.

The latest disturbance recovery event is at 850 seconds and the observed
maximum recovery was 165 seconds, leaving a further 35-second buffer in the
1050-second protocol. Boundary cases settled within 195 seconds, so 300 seconds
matches tracking while retaining more than 100 seconds of post-settling data.

## Training variation

`randomize=True` samples a new tracking Episode on every reset. Most episodes use
a feasible equilibrium start and immediately track one random target; 20% start
near a high or low hard temperature boundary. Every training Episode uses the
tracking Benchmark's 60-step horizon. `disturbance=True` independently samples
the outdoor temperature, both zone heat loads, and HVAC efficiency, beginning
at step `12--22` for `10--20` steps. Noise, delay, and actuator
loss-of-effectiveness remain the shared optional channel variations.

## Controllers and workflows

The HVAC PID configuration is a diagonal PI controller with
`(Kp, Ki, Kd) = (0.12, 0.002, 0)` for both zones. The MPC uses a 5-second solve
interval, a 10-step prediction horizon, the model steady-state input, and the
same fixed configuration for all three Benchmarks. Their parameters were
selected on randomized training episodes with physical disturbances, not on
the fixed formal Benchmark cases.

```python
import aiogym

env = aiogym.make_env("hvac", benchmark="tracking")
pid = aiogym.make_controller("pid", env=env)
mpc = aiogym.make_controller("mpc", env=env)
result = aiogym.compare_policies(
    env=env,
    policies={"pid": pid, "mpc": mpc},
    seeds=range(20),
)
env.close()
```

The ordinary environment also follows the common
`collect -> train -> save/load -> evaluate/compare` workflow. Short automated
SAC runs verify that path but are not learned-controller performance evidence.

## Current controller comparison

The active local comparison artifacts under `runs/hvac/benchmarks/` contain PI and MPC
only. Each Benchmark uses 20 distinct cases (`0--19`), for 120 evaluated
controller Episodes. Every Episode completed its full horizon with
`unsafe_rate = 0`.

| Benchmark | PI median return | MPC median return | Ranking |
|---|---:|---:|---|
| `tracking` | -0.550085 | -1.874356 | PI, MPC |
| `disturbance-rejection` | -2.570576 | -0.264141 | MPC, PI |
| `boundary-safety` | -202.126235 | -217.106827 | PI, MPC |

These are deterministic simulator baselines for the declared parameter set.
They do not establish building calibration, robustness to parameter
uncertainty, or learned-policy performance.
