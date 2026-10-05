# Two-input, two-output CSTR model

`scenario = "cstr"` · [User guide](../user-guide.md)

## Physical interface

Exothermic continuous stirred-tank reactor with two inputs and two controlled outputs.

| Interface | Ordered values / units |
|---|---|
| State and output | `[reactant_concentration, reactor_temperature]`; mol/L, degC |
| Action | `[feed_pump, cooling]`, each in `[0, 1]` |
| Observation | 6 normalized channels: concentration, temperature, their references, their tracking errors |
| Derived errors | `(measurement-reference) / scale`, scales `[0.18 mol/L, 45 degC]`, clipped to `[-1, 1]` |
| Disturbances | `feed_temperature`, `feed_concentration`, `coolant_temperature`; hidden from policy |
| Timing | Control `1 s`; integration substep ≤ `0.01 s` |

Errors use the same noisy/clipped/delayed measurements and aligned references
seen by the policy. References receive no sensor noise; errors receive no
independent noise. Both action values and both target values are required.

## Default task

| Setting | Value |
|---|---|
| Initial state / reference | `(0.117710 mol/L, 60 degC)` |
| Initial steady action | `(0.5, 0.263499)` |
| Target from reset | `(0.075 mol/L, 72 degC)` |
| Horizon | `225 steps = 225 s` |

The inverse solves mass balance for feed flow and energy balance for cooling.
An infeasible target returns no operating point; neither actuator is fixed or clipped.

## Reward and success

Default reward: `regulation`, `-control_dt * mean(((output-reference)/scale)**2)`,
with an additional `100` penalty on a safety violation. See the
[final-window success rule](../user-guide.md#metrics-and-ranking) for how these bands are evaluated.

| Controlled output | Error scale | Success tolerance |
|---|---|---|
| Reactant concentration | 0.18 mol/L | 0.0036 mol/L |
| Reactor temperature | 45 degC | 0.9 degC |

Tolerances are 2% of the error scales; safety limits are listed separately below.

## Controller defaults

| Controller setting | Value |
|---|---|
| PI concentration → feed `(Kp, Ki, Kd)` | `(16, 0.4, 0)` |
| PI temperature → cooling `(Kp, Ki, Kd)` | `(-0.2, -0.005, 0)` |
| MPC prediction / replanning | 1 × 1 s action / every step |
| MPC inputs / outputs | Both actuators / both outputs |
| MPC reference / move weights | Exact steady input / zero for both actuators |

## Training variation

| Option | Sampling |
|---|---|
| `randomize=True` | Interior equilibrium → independent two-output target; 225 steps |
| `boundary_probability=p` | Fraction `p` uses the `86–90 degC` boundary pre-run below |
| `disturbance=True` | All 3 disturbances; start step `50–90`, duration `50–90` steps |

Noise, delay, and faults use the [shared options](../user-guide.md#inspect-and-configure-an-environment).

## Benchmarks

| Benchmark | Protocol | Steps / physical duration |
|---|---|---|
| `tracking` | Equilibrium start → independently sampled two-output target | 225 / 225 s |
| `disturbance-rejection` | Feed temperature/concentration and coolant changes, then restoration | 400 / 400 s |
| `boundary-safety` | High-feed, low-cooling pre-run to `86–90 degC`; recover toward default target | 100 / 100 s |

Ranking: **unsafe rate → mean cumulative return**. Standard cases: seeds `0–19`.
[Shared evaluation rules](../user-guide.md#metrics-and-ranking).

<details>
<summary>Case sampling and recovery windows</summary>

| Setting | Range / rule |
|---|---|
| Tracking envelope | Concentration `0.02–0.20 mol/L`, temperature `50–82 degC`; minimum move `0.02 mol/L` / `5 degC` |
| Equilibrium acceptance | Both actions `0.05–0.95`; zero model derivative at start/target |
| Disturbance timing | Start `100–180`, duration `100–min(180, 320-start)` steps |
| Disturbance values | Feed `12–32 degC`, concentration `0.75–1.30 mol/L`, coolant `5–20 degC` |
| Recovery allowance | Restore by step 320; 40 recovery steps before final 40-step success window |
| Boundary commands | Feed `0.95–1.0`, cooling `0–0.05`, starting at default equilibrium |
| Boundary pre-run | Check each second for at most 50 s; concentration evolves with the pre-run, not independently sampled |
| Boundary acceptance | Temperature may slightly exceed sampled threshold; reject hard-constraint crossings; use reached state and applied action |
| Observation noise | Disabled in formal benchmarks |

</details>

## Parameters and model scope

Query `aiogym.list_parameters("cstr")` or the configured `env.describe()["parameters"]`.

| Constraint / scope | Meaning |
|---|---|
| Safety termination | Physical state bounds or reactor temperature above `92 degC` |
| Trip override | `parameters={"temperature_trip": 95.0}` in an ordinary environment |
| Model scope | Normalized kinetics/thermal coefficients; not calibrated to a named reactor or plant measurements |

## Run this scenario

[Train](../user-guide.md#3-train-and-compare) · [Compare a checkpoint](../user-guide.md#evaluate-a-saved-model)
