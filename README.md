# AIO-Gym

AIO-Gym is a native Python backend for process-control reinforcement learning,
controller benchmarking, offline-data generation, and model-based research. It
provides synchronous, seedable, and vectorizable Gymnasium environments without
requiring a browser or external simulator.

## Features

- Eight process scenarios: `cascade`, `cascade-recirculating`, `quadruple`,
  `cstr`, `hvac`, `extraction`, `heater`, and `crystallization`.
- A generic model contract based on state `x`, action `u`, controlled output
  `y`, setpoint `y_sp`, and declared disturbances.
- PID, linear MPC, nonlinear MPC oracle, Python policy, SB3, and ONNX
  controller support.
- Tracking, economic, KPI, robustness, and safety benchmark protocols.
- Reproducible benchmark suites, reports, plots, leaderboards, and artifact
  validation.
- SB3 and RLPD training workflows with standard benchmark artifacts and ONNX
  export.
- Declarative and Python extension interfaces for custom process models and
  controllers.

## Requirements

- Python 3.10 or newer
- NumPy and Gymnasium
- CasADi for nonlinear MPC
- PyTorch and Stable-Baselines3 for training
- ONNX and ONNX Runtime for policy export and inference

The default installation contains only the core environment stack. Optional
extras install nonlinear optimization, training, and ONNX support when needed.

## Installation

Clone the repository and install it from the repository root:

```bash
python -m pip install -e .
```

Install optional functionality individually:

```bash
python -m pip install -e ".[oracle]"
python -m pip install -e ".[rl]"
python -m pip install -e ".[onnx]"
```

Install the complete development stack:

```bash
python -m pip install -e ".[all,dev]"
```

A non-editable local installation is also supported:

```bash
python -m pip install .
```

## Quick Start

Importing `aiogym` registers all built-in Gymnasium environment IDs:

```python
import gymnasium as gym
import aiogym

env = gym.make("AIOGym/CSTR-v0")
obs, info = env.reset(seed=7)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

The direct factory accepts scenario, task, and benchmark-objective options:

```python
import aiogym

env = aiogym.make_env(
    scenario="quadruple",
    task="minimum-phase",
    objective="tracking",
    seed=7,
    episode_steps=200,
    auto_events=True,
)
```

`auto_events` controls generic automatically generated within-episode events; it
does not switch the process differential equations on or off. Named tasks can
disable generic events and still declare deterministic setpoint or disturbance
schedules. `auto_events` is the only accepted field for this behavior.

`scenario` selects the process, `task` selects the operating point and
experiment, and `objective` selects scoring. Reproducible benchmark artifacts
record all three identities separately.

Every scenario uses the same backend contracts:

```text
observation = [x, y_sp, disturbances]
action      = flat u vector in [0, 1]
```

Set `action_mode="setpoint"` to evaluate supervisory policies over the built-in
PID layer when the selected model declares a supervisory layout.

## Documentation

- Start at the [documentation index](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/index.md).
- Read [concepts](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/concepts.md) for the scenario-to-artifact vocabulary.
- See the [architecture guide](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/architecture.md) for the source tree,
  layer ownership, and extension points.
- Use the [capability matrix](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/capabilities.md) to distinguish interface
  support from bundled task and benchmark evidence.
- See the [public API guide](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/public_api.md) for the model, environment,
  task, objective, controller, and benchmark calling hierarchy.

Discover canonical built-in and registered IDs without reading package files:

```python
aiogym.list_scenarios()
aiogym.list_tasks()
aiogym.list_suites()
aiogym.list_controllers()
```

The same catalog is available from the unified command line:

```bash
aiogym list scenarios
aiogym list tasks --scenario cstr
aiogym list suites
aiogym list controllers
```

Use `aiogym --help` to see the grouped benchmark, training, and artifact
workflows. The unified `aiogym` command is the only installed CLI.

## Benchmarking

The benchmark CLI uses one command shape:

```text
aiogym benchmark suite <scenario-or-suite-id> [task]
```

A scenario without `task` runs all of its registered tasks; adding `task` runs
only that experiment. A named cross-scenario suite runs its declared cases.
Scenarios without registered tasks run one default tracking case.

Run the standard controller-comparison suite:

```bash
aiogym benchmark suite standard-baselines \
  --episodes 3
```

Run a smaller single-scenario comparison:

```bash
aiogym benchmark suite cstr \
  --objective tracking \
  --controllers pid,mpc
```

Commands with a positional task default to `runs/<scenario>/<task>`. Targets
without a task use `runs/<target>_suite`, even when a taskless scenario expands
to only one default case. Re-running replaces the standard managed artifacts in
that directory, so the latest result always has a stable review path. Tracking
objectives automatically save rollout and control figures. Pass
`--no-overwrite` to write a timestamped sibling directory, or `--artifact-dir`
to choose another base directory.

Compare the minimum- and nonminimum-phase quadruple-tank tasks without mixing
their rankings:

```bash
aiogym benchmark suite quadruple-phase-comparison --episodes 1
```

Run one task with its default benchmark conditions, or replace its duration and
absolute setpoint schedule:

```bash
aiogym benchmark suite quadruple minimum-phase

aiogym benchmark suite quadruple minimum-phase \
  --episode-steps 360 \
  --setpoint-step "0:14.2629675195507,12.783158403008972"
```

Use seeded per-episode targets when measuring generalization rather than strict
reference reproduction:

```bash
aiogym benchmark suite quadruple nonminimum-phase \
  --controllers pid,mpc,oracle \
  --episodes 10 \
  --randomize-setpoints
```

Each seed generates its target schedule during `env.reset(seed=...)`. Tasks
with setpoint events retain their initial equilibrium and sample the event
targets; tasks without events sample the initial target. The realized schedule
is recorded in result and rollout metadata.

Run the heated-tank cascade as an explicit continuous-production economic task:

```bash
aiogym benchmark suite cascade continuous-benchmark \
  --objective economic \
  --controllers pid,mpc
```

Direct `cascade` environments use zero throughput targets. The
continuous task's `4.0e-4 m3/s` target is an assumed benchmark throughput, not a
validated equipment rating. There is no separate batch operation mode; setting
`product_flow_sp=min_product_flow=0` disables production value and shortfall
accounting. See [the cascade scenario guide](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/scenarios/cascade.md).

Run the four formal open-cascade control tasks under their task-owned default
objectives:

```bash
aiogym benchmark suite cascade-control \
  --episodes 3
```

The suite covers commissioning, a first-stage temperature step, deterministic
disturbance rejection, and recoverable heater-interlock conditions with PID and
MPC. These tasks reuse the continuous benchmark's assumed throughput but keep
tracking, robustness, and safety rankings separate from production economics.

Run the PDF-derived closed-loop retrofit independently from the historical
open cascade:

```bash
aiogym benchmark suite cascade-recirculating commissioning \
  --objective tracking \
  --controllers pid,mpc,oracle
```

Run all four formal recirculating tasks under their task-owned default
objectives with PID and MPC:

```bash
aiogym benchmark suite cascade-recirculating \
  --episodes 3
```

This scenario has one 2 kW Tank 1 heater, four actuator actions, passive
overflow returns to Tank 3, and no production-economic objective. See the
[recirculating scenario guide](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/scenarios/cascade_recirculating.md) and
[accuracy/authenticity report](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/reports/cascade-recirculating-accuracy-authenticity.md).

Run all four formal quadruple-tank tasks with PID, MPC, and NMPC Oracle on every
task:

```bash
aiogym benchmark suite quadruple --episodes 1
```

Run the zero-boundary or deterministic-disturbance subsets:

```bash
aiogym benchmark suite quadruple zero-boundary-stress \
  --controllers pid,mpc,oracle \
  --episodes 1

aiogym benchmark suite quadruple disturbance-rejection \
  --controllers pid,mpc,oracle \
  --episodes 1
```

Use an ONNX policy:

```bash
aiogym benchmark suite cstr \
  --objective economic \
  --controllers onnx \
  --onnx-path path/to/policy.onnx \
  --episodes 1
```

The same workflow is available through Python:

```python
import aiogym

payload = aiogym.run_benchmark({
    "scenario": "cstr",
    "objective": "tracking",
    "controllers": ["pid", "mpc"],
    "seeds": [7, 8, 9],
    "output_dir": "runs/cstr_tracking",
})
```

## Benchmark Artifacts

Single-task, multi-task, and Python API benchmark runs write the same standard artifact
directory contract:

```text
<artifact_dir>/
  benchmark.json
  config/
  metadata/
  summary/
  results/
  rollouts/
  training/
  figures/
  report.md
```

Generate a Markdown report or validate an artifact directory:

```bash
aiogym artifacts report path/to/artifact_dir
aiogym artifacts check path/to/artifact_dir
```

Default generated outputs use the current working directory's `runs/` folder.
Single-task benchmarks use the stable `runs/<scenario>/<task>` layout and
suite benchmarks use `runs/<suite>_suite`. Both replace their previous managed
artifacts by default. Use `--no-overwrite` to create a timestamped sibling
directory. Set `AIOGYM_RUNS_DIR` to choose another default root, or pass an
explicit base path such as `--out-dir` or `--artifact-dir`; explicit paths still
obey the overwrite switch.

Benchmark commands always write the standard artifact directory contract.
Large rollout trajectories are stored only in `rollouts/rollouts.json`;
`benchmark.json` references that file through its `artifacts` mapping instead
of embedding a duplicate copy.

## Training

Train an SB3 policy and optionally export ONNX:

```bash
aiogym train sb3 \
  --scenario cstr \
  --algo sac \
  --objective tracking \
  --steps 10000 \
  --onnx
```

Train on a named tracking task with task-owned timing, reward, and observation
semantics:

```bash
aiogym train sb3 \
  --scenario quadruple \
  --task minimum-phase \
  --gamma 0.999 \
  --steps 500000 \
  --n-envs 4 \
  --gradient-steps 4 \
  --learning-starts 10000 \
  --learning-rate 1e-4
```

When `--task` is provided, its control interval, episode length, tracking
weights, and observation contract are used unless explicitly overridden. The
quadruple tasks use normalized tracking errors instead of setpoint channels,
exclude disturbance channels, include the previous applied action, and normalize
observations with fixed model-schema bounds. Boolean observation options support
both forms, such as `--disturbance-obs` and `--no-disturbance-obs`, for explicit
ablation runs. Older checkpoints trained under another observation contract
must still be evaluated with that original contract. When periodic
learning-curve evaluation is enabled, the main checkpoint is the best evaluated
policy and a separate `_final.zip` checkpoint preserves the last training state.
Training evaluates every 10,000 steps and saves the final plotting rollout by
default; use `--learning-curve-every 0` or `--no-save-rollout` for a minimal
artifact run.

Run the offline-to-online RLPD workflow:

```bash
aiogym train rlpd \
  --scenario quadruple \
  --task minimum-phase \
  --offline-episodes 20 \
  --online-steps 10000
```

Training runs write checkpoints, exports, learning curves, evaluation results,
and standard benchmark artifacts. High-level training commands accept all five
public objectives; `robustness` and `safety` resolve to the KPI environment
reward while retaining their own evaluation identity. Select it with
`--objective`; training commands do not accept a separate reward-mode option.
SB3 and RLPD resolve `--task`, task-owned timing, tracking weights, observation
contracts, and generated run identities through the same configuration layer.
Their algorithm-specific optimization flags remain separate.

## Custom Models and Controllers

Start with the declarative model example when the dynamics can be expressed as
formulas:

```bash
python aiogym/models/examples/declarative_model.py
```

Use the Python model example for lower-level behavior:

```bash
python aiogym/models/examples/custom_model.py
```

Human-readable documentation for every built-in model lives under
`docs/scenarios/`. Structured model metadata is stored in benchmark artifacts
for reproducibility.

The shared physical-model foundation is described in
[`docs/model_infrastructure.md`](https://github.com/zhuxinji/AIO-gym-temp/blob/main/docs/model_infrastructure.md). It covers
versioned parameter provenance, scenario task profiles, solver metadata, and
generic numerical readiness checks. Unmigrated profiles retain
`legacy-unverified`; `quadruple` is the first `reference-parameterized` physical
model.

## Package Layout

```text
aiogym/
  _internal/      private shared utilities
  cli/            console entrypoints
  controllers/    controller API, algorithms, configs, and tuning
  evaluation/     protocols, objectives, metrics, reports, and artifacts
  models/         model contracts, registry, scenarios, and metadata
  rl/             transition data, RLPD, and training workflows
  tests/          backend contract and regression tests
  env.py          Gymnasium environment
  env_factory.py  public environment factory
```

## Validation

Run the complete backend test suite from the repository root:

```bash
python -m pytest -q
```

Run a short end-to-end benchmark check:

```bash
aiogym benchmark suite standard-baselines \
  --scenarios cstr \
  --objectives tracking \
  --controllers pid,mpc \
  --episodes 1 \
  --episode-steps 2
```

## License

MIT
