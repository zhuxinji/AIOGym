# Quadruple-Tank model

`scenario = "quadruple"` · [User guide](../user-guide.md)

## Physical interface

Johansson's nonlinear four-tank, two-pump laboratory process. Pump splits between
upper/lower tanks determine the phase configuration.

| Interface | Ordered values / units |
|---|---|
| State | `[h1, h2, h3, h4]`, cm |
| Output | `[h1, h2]`, cm |
| Action | `[pump_1_voltage, pump_2_voltage]` in `[0, 1]`, scaled by `max_voltage` |
| Observation | `[h1, h2, h3, h4, r1, r2, e1, e2]` |
| Encoding | Levels/references: `0–20 cm → [0, 1]`; errors: `(hi-ri)/20 → [-1, 1]` |
| Other model units | Area `cm^2`, flow `cm^3/s`, voltage `V`, time `s` |
| Timing | Control `1 s`; integration substep `0.1 s` |

Derived errors use the same clipped/noisy/delayed measurements and aligned
references as the policy. All environment modes share the eight-channel interface;
older six-observation checkpoints/datasets require retraining or recollection.

## Default task

| Setting | Value |
|---|---|
| Initial state | Default equilibrium |
| Target from reset | `(8, 18) cm`, with feasible unsaturated steady pump commands |
| Horizon | `180 steps = 180 s` |

## Reward and success

Default reward: `regulation`, `-control_dt * mean(((output-reference)/scale)**2)`,
with an additional `100` penalty on a safety violation. See the
[final-window success rule](../user-guide.md#metrics-and-ranking) for how these bands are evaluated.

| Output | Error scale | Success tolerance | Hard state bounds |
|---|---|---|---|
| Each lower-tank level | `20 cm` | `0.4 cm` | `0–20 cm` |

## Controller defaults

| Controller setting | Value |
|---|---|
| PI lower-tank 1 `(Kp, Ki, Kd)` | `(1.0159437333, 0.1, 0)` |
| PI lower-tank 2 `(Kp, Ki, Kd)` | `(2.0, 0.0900169896, 0)` |
| MPC prediction / replanning | 1 × 1 s action / every step |

Same configuration across all benchmarks. Increase `prediction_horizon` for a
longer [MPC prediction](../user-guide.md#configure-mpc) without changing the environment.

## Training variation

| Option | Sampling |
|---|---|
| `randomize=True` | Feasible interior equilibrium → sampled target; each output moves ≥ `3 cm`; 180 steps |
| `boundary_probability=p` | Fraction `p` uses a pre-run with all levels at `80–92%` of hard maximum |
| `disturbance=True` | Pump-flow factor `0.78–0.94`; start step `36–72`, duration `36–72`, then restore `1.0` |

Training samples new conditions; fixed benchmarks remain evaluation protocols.

## Training settings

Use the [training recipe](../user-guide.md#3-train-and-compare) with
`scenario = "quadruple"`. For the 50,000-step SAC example, set:

```python
training_steps = 50_000
algorithm_settings = {}
record_interval = 500
validation_interval = 5_000
```

This uses SAC's native Stable-Baselines3 initialization and exploration. To use
PPO or DDPG, change `algorithm` in the workflow setup and use separate output
paths. DDPG adds no action noise unless explicitly configured. With PPO's
default `n_steps=2048`, a 50,000-step request completes 51,200 transitions.
These example budgets do not establish performance across training seeds.

## Benchmarks

| Benchmark | Protocol | Steps / physical duration |
|---|---|---|
| `tracking` | Feasible equilibrium start → independent lower-tank target | 180 / 180 s |
| `disturbance-rejection` | Temporary pump-flow loss, then restoration | 500 / 500 s |
| `boundary-safety` | High legal pump voltages until one tank reaches `82–90%` of maximum | 180 / 180 s |

Ranking: **unsafe rate → mean cumulative return**. Standard cases: seeds `0–19`.
[Shared evaluation rules](../user-guide.md#metrics-and-ranking).

<details>
<summary>Case sampling and recovery windows</summary>

| Setting | Range / rule |
|---|---|
| Tracking envelope | Lower-tank initial/target levels `7–16 cm`; each moves ≥ `3 cm` |
| Equilibrium acceptance | Exact lower-tank targets, all 4 levels `0–20 cm`, actions `0.02–0.95`, zero derivative |
| Case construction | Resolve both full equilibria once at reset; no precomputed case table or runtime steady-state solve |
| Disturbance timing / factor | Start `120–220`, duration `150–min(250, 400-start)` steps; pump factor `0.78–0.90` |
| Recovery allowance | Restore by step 400; 50 recovery steps before final 50-step success window |
| Boundary construction | Forward pre-run from normal state under high legal pump voltages |
| Observations | Exact; no measurement noise |

</details>

## Parameters and phase configuration

| `gamma[0] + gamma[1]` | Phase |
|---|---|
| `> 1` | Minimum phase; default `[0.7, 0.6]` |
| `< 1` | Nonminimum phase |
| `= 1` | Transmission zero at the origin |

Formal benchmarks freeze the default parameters/reward. For a custom nonminimum-phase run:

```python
import aiogym

with aiogym.make_env("quadruple", parameters={"gamma": [0.4, 0.4]}) as env:
    print(env.parameters["gamma"])
```

The custom run records its parameters without a formal benchmark label.

## Run this scenario

[Train](../user-guide.md#3-train-and-compare) · [Compare a checkpoint](../user-guide.md#evaluate-a-saved-model)
