# Fired-heater model

`scenario = "heater"` · [User guide](../user-guide.md)

## Physical interface

Fired-heater surrogate: combustion, firebox heat storage, process-side heat transfer, and flue oxygen.

| Interface | Ordered values / units |
|---|---|
| State | `[firebox_temperature, outlet_temperature, flue_oxygen]`; degC, degC, % |
| Controlled output | `[flue_oxygen, outlet_temperature]` |
| Action | `[air_damper, fuel_valve]`, each in `[0, 1]`; scaled by `maximum_air_flow` / `maximum_fuel_flow` |
| Observation | 5 normalized values: 3 states, oxygen reference, outlet-temperature reference |
| Disturbances | `feed_temperature`, `ambient_temperature`, `feed_flow`, `fuel_heating_value_factor`; hidden from policy |
| Timing | Control `1 s`; integration substep ≤ `0.05 s` |

## Default task

| Setting | Value |
|---|---|
| Initial output | `(3%, 370 degC)` |
| Initial state / action | `(758.714286, 370, 3)` / `(0.383882, 0.758275)` |
| Target from reset | `(4%, 366 degC)` |
| Horizon | `300 steps = 300 s` |

The steady-state inverse solves heat balances and the lean air/fuel ratio;
infeasible states/actions return no operating point rather than a clipped solution.

## Reward and success

Default reward: `regulation`, `-control_dt * mean(((output-reference)/scale)**2)`,
with an additional `100` penalty on a safety violation. See the
[final-window success rule](../user-guide.md#metrics-and-ranking) for how these bands are evaluated.

| Controlled output | Error scale | Success tolerance |
|---|---|---|
| Flue oxygen | 3.2 percentage points | 0.064 percentage points |
| Outlet temperature | 8 degC | 0.16 degC |

Tolerances are 2% of the error scales; safety limits are listed separately below.

## Controller defaults

| Controller setting | Value |
|---|---|
| PI oxygen → air `(Kp, Ki, Kd)` | `(0.09, 0.001, 0)` |
| PI temperature → fuel `(Kp, Ki, Kd)` | `(0.025, 0.0003, 0)`; cross gains zero |
| MPC prediction / replanning | 30 × 1 s actions / execute 2, replan every 2 s |
| MPC move / steady-input weights | `5` / `1` per actuator |
| MPC linearization reference | Model-derived steady input |

## Training variation

| Option | Sampling |
|---|---|
| `randomize=True` | Feasible interior equilibrium → independently sampled target; 300 steps |
| `boundary_probability=p` | Fraction `p` uses the boundary pre-run below |
| `disturbance=True` | All 4 disturbances; start step `70–120`, duration `100–min(160, 240-start)`, restored by 240 |

Noise, delay, and faults use the [shared options](../user-guide.md#inspect-and-configure-an-environment).

## Benchmarks

| Benchmark | Protocol | Steps / physical duration |
|---|---|---|
| `tracking` | Equilibrium start → independently sampled target | 300 / 300 s |
| `disturbance-rejection` | Feed, ambient, flow, and fuel-quality changes, then restoration | 700 / 700 s |
| `boundary-safety` | Heat outlet to `382–385 degC`, then lower oxygen to `1.30–1.70%`; recover toward default target | 450 / 450 s |

Ranking: **unsafe rate → mean cumulative return**. Standard cases: seeds `0–19`.
[Shared evaluation rules](../user-guide.md#metrics-and-ranking).

<details>
<summary>Case sampling and recovery windows</summary>

| Setting | Range / rule |
|---|---|
| Tracking envelope | Oxygen `2.5–5.0%`, outlet `364–372 degC`; minimum move `0.4 percentage points` / `2 degC` |
| Equilibrium acceptance | Both actions `0.05–0.95`; zero model derivative at start/target |
| Disturbance timing | Start `140–240`, duration `200–320` steps |
| Disturbance values | Feed `255–305 degC`, ambient `5–35 degC`, flow `75–105 kg/s`, fuel factor `0.85–1.15` |
| Disturbed feasibility | Default `(3%, 370 degC)` target must have an exact equilibrium with both actions `0.05–0.95` |
| Boundary construction | Two-stage forward pre-run; the 450 s recovery begins after reaching the sampled boundary |

</details>

## Parameters and model scope

Query `aiogym.list_parameters("heater")` or the configured `env.describe()["parameters"]`.

| Constraint / scope | Meaning |
|---|---|
| Safety termination | Physical state bounds, outlet temperature above `415 degC`, oxygen below `1.2%` |
| Operating targets | Narrower sampling ranges; not extra trips |
| Model scope | Lumped surrogate, not calibrated to a named furnace or plant data |
| Measurements | Outlet temperature is not tube-skin temperature; oxygen alone is not a combustion/emissions model |

## Run this scenario

[Train](../user-guide.md#3-train-and-compare) · [Compare a checkpoint](../user-guide.md#evaluate-a-saved-model)
