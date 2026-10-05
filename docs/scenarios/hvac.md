# Two-zone HVAC model

`scenario = "hvac"` · [User guide](../user-guide.md)

## Physical interface

Two thermally coupled zones, each with a bidirectional heat-pump command.

| Interface | Ordered values / units |
|---|---|
| State and output | `[zone_0_temperature, zone_1_temperature]`, degC |
| Action | `[hvac_zone_0, hvac_zone_1]` in `[0, 1]`: `0` maximum cooling, `0.5` zero power, `1` maximum heating |
| Power limit | `maximum_zone_power` per zone in either direction |
| Observation | 4 normalized values: temperatures using `-20–60 degC`, references using `18–26 degC` |
| Disturbances | Outdoor temperature, each zone's internal load, common power-efficiency factor; hidden from policy |
| Timing | Control `5 s`; integration substep ≤ `0.25 s` |

## Default task

| Setting | Value |
|---|---|
| Initial state | `(22, 22) degC`; outdoor temperature `5 degC` |
| Initial steady action | `(0.7125, 0.7125)` |
| Target from reset | `(24, 20) degC` |
| Horizon | `60 steps = 300 s` |

The steady inverse includes outdoor transfer, zone coupling, internal loads,
and efficiency. Infeasible targets return no steady action.

## Reward and success

Default reward: `regulation`, `-control_dt * mean(((output-reference)/scale)**2)`,
with an additional `100` penalty on a safety violation. See the
[final-window success rule](../user-guide.md#metrics-and-ranking) for how these bands are evaluated.

| Controlled output | Error scale | Success tolerance |
|---|---|---|
| Each zone temperature | 8 degC | 0.16 degC |

Tolerances are 2% of the error scales; safety limits are listed separately below.

## Controller defaults

| Controller setting | Value |
|---|---|
| PI `(Kp, Ki, Kd)` | `(0.32, 0.01, 0)` for each zone; diagonal gains, fixed default-action bias |
| MPC prediction / replanning | 1 × 5 s action / every step |
| MPC reference | Model steady input |
| MPC move / steady-input weights | Zero / zero |

PI tuning used tracking seeds `1000–1007`; all three benchmarks use the same
configuration on seeds `0–19`.

## Training variation

| Option | Sampling |
|---|---|
| `randomize=True` | Feasible interior equilibrium → independent target; 60 steps |
| `boundary_probability=p` | Fraction `p` uses the hot/cold pre-run family below |
| `disturbance=True` | Outdoor temperature, 2 heat loads, efficiency; start step `12–22`, duration `10–20` steps |

Noise, delay, and faults use the [shared options](../user-guide.md#inspect-and-configure-an-environment).

## Benchmarks

| Benchmark | Protocol | Steps / physical duration |
|---|---|---|
| `tracking` | Equilibrium start → independent two-zone target | 60 / 300 s |
| `disturbance-rejection` | Outdoor temperature, zone loads, efficiency changes, then restoration | 210 / 1050 s |
| `boundary-safety` | Legal hot/cold pre-run toward sampled temperature boundary | 60 / 300 s |

Ranking: **unsafe rate → mean cumulative return**. Standard cases: seeds `0–19`.
[Shared evaluation rules](../user-guide.md#metrics-and-ranking).

<details>
<summary>Case sampling and recovery windows</summary>

| Setting | Range / rule |
|---|---|
| Tracking envelope | Both start/target temperatures `19–25 degC`; each zone moves ≥ `1.5 degC` |
| Equilibrium acceptance | Both actions `0.05–0.95`; zero model derivative at start/target |
| Disturbance timing | Start `50–90`, duration `40–80` steps |
| Disturbance values | Outdoor `-10–5 degC`, zone 0 load `300–800 W`, zone 1 load `-500–-100 W`, efficiency `0.65–0.90` |
| Recovery allowance | Restore by 850 s; at least 200 s remain |
| Boundary thresholds | Hot `48–54 degC` or cold `-14–-8 degC`, inside hard limits |
| Boundary construction | Forward simulation under legal weather/loads/actions; restore driving disturbances before evaluation |

</details>

## Parameters and model scope

Query `aiogym.list_parameters("hvac")` or the configured `env.describe()["parameters"]`.
This is a normalized simulation benchmark, not a building/equipment model
calibrated against measurements.

## Run this scenario

[Train](../user-guide.md#3-train-and-compare) · [Compare a checkpoint](../user-guide.md#evaluate-a-saved-model)
