# Quickstart

## Inspect and run an environment

```python
import aiogym

print(aiogym.list_scenarios())
print(aiogym.list_benchmarks("quadruple"))
print(aiogym.list_parameters("quadruple"))

env = aiogym.make_env(
    "quadruple",
    parameters={"pump_gain": [3.2, 3.25]},
)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

## Collect varied training data

```python
env = aiogym.make_env(
    "quadruple",
    randomize=True,
    disturbance=True,
    noise=True,
    delay=True,
    fault=True,
)
pid = aiogym.make_controller("pid", env=env)
aiogym.collect(env=env, policy=pid, episodes=2, seed=0, output="runs/data")
env.close()

reader = aiogym.DatasetReader("runs/data")
print(len(reader), reader.transition_count)
```

## Training variation

`randomize=True` samples a new operating condition for one tracking training
task on every reset. 80% of episodes start from a feasible interior steady
state and apply a randomized setpoint step; 20% start near the upper liquid-
level boundary and track a feasible interior target. Physical disturbances are
not part of this distribution. The fixed disturbance-rejection and boundary-
safety Benchmarks remain evaluation-only protocols.

`disturbance=True` independently samples one temporary physical pump-capacity
loss per episode. It may be used with either the fixed training condition or
`randomize=True`; it changes the true process dynamics and is not appended to
the policy observation. Quadruple applies a `0.78–0.94` pump-flow factor from a
sampled step in `120–240` for `120–240` steps. Three-Tank applies a `0.72–0.94`
factor from a sampled step in `450–800` for `300–600` steps. Both restore the
factor to `1.0` afterward.

The optional policy-channel variations can be enabled with `True` or configured
explicitly:

```python
env = aiogym.make_env(
    "quadruple",
    randomize=True,
    disturbance=True,
    noise={"std": 0.01, "bias_std": 0.002},
    delay={"observation_steps": [0, 2], "action_steps": [0, 1]},
    fault={
        "probability": 0.05,
        "severity": [0.2, 0.5],
        "duration_steps": [20, 100],
    },
)
```

- `std` is zero-mean white observation noise sampled on every step;
  `bias_std` controls an observation bias sampled once per episode. Both are
  fractions of each observation channel's range. Noise is not added to known
  reference channels.
- `observation_steps` and `action_steps` are delays sampled once per episode.
  Observation delay applies to the complete policy observation; action delay is
  initially filled with the model's default action.
- `probability` is the per-episode probability of one actuator
  loss-of-effectiveness fault. One action channel is multiplied by
  `1 - severity` for the sampled number of consecutive control steps.

When the options are `True`, the values above are their defaults. The resolved
physical-disturbance schedule is stored in `episode_spec`; noise bias, delay,
and fault values are stored in `runtime_variation`. Workflows record both.
These variations are AIO-Gym environment behavior, not Stable-Baselines3
exploration settings.

## Train, load, evaluate, and compare

Install `aiogym[rl]`, then run:

```python
training_env = aiogym.make_env(
    "quadruple", randomize=True, disturbance=True, noise=True
)
trained = aiogym.train(
    env=training_env,
    algorithm="sac",
    steps=64,
    seed=0,
    demonstrations="runs/data",
    behavior_cloning_epochs=10,
    record_every=500,
    output="runs/sac",
)
aiogym.plot_training_curve(
    trained["training_curve"],
    output="runs/replotted-training-curve.svg",
)
sac = aiogym.load_policy(
    trained["checkpoint"], env=training_env
)
training_env.close()

benchmark_env = aiogym.make_env("quadruple", benchmark="tracking")
pid = aiogym.make_controller("pid", env=benchmark_env)
mpc = aiogym.make_controller("mpc", env=benchmark_env)
evaluation = aiogym.evaluate(
    env=benchmark_env, policy=sac, seeds=[0, 1, 2]
)
comparison = aiogym.compare_policies(
    env=benchmark_env,
    policies={"pid": pid, "mpc": mpc, "sac": sac},
    seeds=[0, 1, 2],
)
benchmark_env.close()
```

For a Benchmark, `seeds` are reproducible case seeds rather than repeated
controller seeds. `compare_policies()` resolves the same ordered `EpisodeSpec`
values for every policy. A Benchmark with a deliberately fixed episode may map
multiple case seeds to that same episode.

Behavior cloning is optional pretraining for DDPG, SAC, and TD3. The Dataset
must match the training Scenario, Reward, parameters, control interval, and
observation/action dimensions. Training uses `observation` as input and
`commanded_action` as the expert label. Channel-delayed, faulted, or
slew-limited actions are retained for analysis but are not imitation targets.
After pretraining, online training continues for `steps`. AIO-Gym stores the
backend payload and algorithm manifest in the common `model.zip` checkpoint.

External algorithms can implement and register the public `AlgorithmBackend`
contract without imitating an SB3 model. They then reuse the same training
curve, periodic evaluation, best checkpoint, loader, evaluation, comparison,
and CLI paths. See [External algorithm backends](external_algorithms.md).

Short RL runs verify the pipeline; they are not performance evidence. The
default comparison directory is `runs/<scenario>/<benchmark>/`; rerunning the
same benchmark replaces its machine-readable `comparison.json` and trajectory
figure `comparison.svg`. Pass an explicit empty `output` directory when those
artifacts must not be overwritten.

The comparison SVG plots trajectories only for the first ordered seed. Its
return panel shows one point for every seed and a median marker; the JSON retains
all per-seed trajectories and exact return values.

## CLI

```bash
aiogym list benchmarks --scenario quadruple
aiogym list parameters --scenario quadruple
aiogym collect quadruple --randomize --disturbance on --noise on \
  --delay on --fault on \
  --controller pid --episodes 2 --output runs/data
aiogym train quadruple sac --randomize --disturbance on --noise on --steps 64 \
  --demonstrations runs/data --behavior-cloning-epochs 10 \
  --record-every 500 --output runs/sac
aiogym evaluate quadruple --benchmark tracking --controller pid --seeds 0 1 2 \
  --output runs/pid-evaluation.json
aiogym compare quadruple --benchmark tracking --controllers pid mpc \
  --checkpoint sac runs/sac/model.zip \
  --seeds 0 1 2
```

Use `--parameters parameters.json` on any workflow command to apply one JSON
object of model parameter overrides. Workflow commands print compact JSON
summaries; evaluation and comparison trajectories remain in their output
artifacts. Use `aiogym <command> --help` for parameter meanings and defaults.
