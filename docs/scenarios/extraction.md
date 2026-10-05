# Multistage extraction model

`scenario = "extraction"` · [User guide](../user-guide.md)

## Physical interface

Five-stage counter-current liquid-gas column, adapted from PC-Gym's nonlinear stage balances.

| Interface | Ordered values / units |
|---|---|
| State | 10 concentrations, alternating liquid/gas for stages 1–5 |
| Output | `stage_5_liquid_concentration` (`CX5`) |
| Action | `[liquid_feed_flow, gas_feed_flow]` in `[0, 1]` |
| Physical flow ranges | Liquid `5–500 volume/h`; gas `10–1000 volume/h` |
| Observation | 11 normalized values: 10 states and `CX5` reference |
| Disturbances | `liquid_feed_concentration`, `gas_feed_concentration`, `mass_transfer_coefficient`; hidden from policy |
| Timing | Control `0.1 h` (6 min); integration substep ≤ `0.01 h` |

Both action channels are required. Time is in simulated hours derived from the
equations, not a calibrated real-column response time.

## Default task

| Setting | Value |
|---|---|
| Initial output | Equilibrium at `CX5=0.30` |
| Initial normalized action | `(0.320519, 0.5)` |
| Initial physical flows | `(163.657, 505) volume/h` |
| Target from reset | `CX5=0.40` |
| Horizon | `100 steps = 10 h` |

The inverse fixes nominal gas flow, solves column equilibrium and liquid flow;
infeasible targets return no operating point.

## Reward and success

Default reward: `regulation`, `-control_dt * mean(((output-reference)/scale)**2)`,
with an additional `100` penalty on a safety violation. See the
[final-window success rule](../user-guide.md#metrics-and-ranking) for how these bands are evaluated.

| Controlled output | Error scale | Success tolerance |
|---|---|---|
| Stage 5 liquid concentration | 0.45 | 0.009 |

Tolerances are 2% of the error scales; safety limits are listed separately below.

## Controller defaults

| Controller setting                | Value                                                                       |
| --------------------------------- | --------------------------------------------------------------------------- |
| PI liquid-flow `(Kp, Ki, Kd)`     | `(1.0, 10.0, 0)`; fixed default-action bias                                 |
| PI gas flow                       | Held at normalized `0.5`, zero feedback gains                               |
| MPC prediction / replanning       | 5 × 0.1 h actions / every step                                              |
| MPC reference                     | Model steady input; [shared RK4 prediction](../user-guide.md#configure-mpc) |
| MPC move weights / steady penalty | `[0.05, 2.0]` / zero                                                        |

PI tuning used tracking seeds `1000–1007`; all three benchmarks use the same
configuration on seeds `0–19`.

## Training variation

| Option | Sampling |
|---|---|
| `randomize=True` | Default interior equilibrium → feasible sampled target; 100 steps |
| `boundary_probability=p` | Fraction `p` uses the low-concentration pre-run below |
| `disturbance=True` | All 3 disturbances; start step `23–40`, duration `33–min(53, 80-start)`, restored by 80 |

Noise, delay, and faults use the [shared options](../user-guide.md#inspect-and-configure-an-environment).

## Benchmarks

| Benchmark | Protocol | Steps / physical duration |
|---|---|---|
| `tracking` | `CX5=0.30` equilibrium → sampled target | 100 / 10 h |
| `disturbance-rejection` | Feed concentrations and mass-transfer changes, then restoration | 580 / 58 h |
| `boundary-safety` | Low-liquid/high-gas pre-run to `CX5=0.01–0.04`; recover toward `0.30` | 100 / 10 h |

Ranking: **unsafe rate → mean cumulative return**. Standard cases: seeds `0–19`.
[Shared evaluation rules](../user-guide.md#metrics-and-ranking).

<details>
<summary>Case sampling and recovery windows</summary>

| Setting | Range / rule |
|---|---|
| Tracking envelope | Target `0.12–0.44`, minimum move `0.05` |
| Equilibrium acceptance | Actions `0.05–0.95`; zero ten-state derivative |
| Disturbance timing | Start `140–240`, duration `200–min(320, 464-start)` steps |
| Disturbance values | Liquid-feed concentration `0.45–0.75`, gas-feed `0.02–0.09`, mass-transfer coefficient `3.5–6.5` |
| Recovery allowance | Restore by step 464 (46.4 h); 58 recovery steps before final 58-step success window |
| Boundary construction | Minimum liquid / near-maximum gas flow from normal state; reachable transient, not an equilibrium |

</details>

## Parameters and model scope

Query `aiogym.list_parameters("extraction")` or the configured `env.describe()["parameters"]`.
Parameter overrides are validated at construction.

| Source | Model content |
|---|---|
| [PC-Gym equations](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py#L1166-L1236) | Five-stage equations and declared parameters |
| [PC-Gym training case](https://github.com/MaximilianB2/pc-gym/blob/main/pc-gym_paper/train_policies/multistage_extraction/me_train.py#L44-L89) | Liquid/gas flow ranges `5–500` / `10–1000`, targets `0.3` / `0.4` |

Literature-derived normalized model; not calibrated against plant measurements.

## Run this scenario

[Train](../user-guide.md#3-train-and-compare) · [Compare a checkpoint](../user-guide.md#evaluate-a-saved-model)
