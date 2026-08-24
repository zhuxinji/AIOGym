# External algorithm backends

AIO-Gym training is not tied to a Stable-Baselines3 model. An external
algorithm implements one `AlgorithmBackend`, registers one algorithm id, and
then uses the same public `train()`, `load_policy()`, `evaluate()`,
`compare_policies()`, training-curve, periodic-evaluation, and best-checkpoint
paths as the built-in algorithms.

The corresponding first-party contracts and framework adapters are grouped in
`aiogym.rl`; workflow code only orchestrates training and artifacts.

The backend is deliberately an adapter around the external implementation. The
external model does not need an SB3 `predict()` method or an SB3 callback.

## Required contract

```python
import aiogym
import my_library


class MyBackend:
    id = "my_algorithm"
    behavior_cloning = None
    requires_dataset = False

    def effective_kwargs(self, *, steps, values):
        # Validate values and return the complete JSON-safe configuration.
        return dict(values)

    def create(self, *, env, seed, algorithm_kwargs):
        return MyModel(env=env, seed=seed, **algorithm_kwargs)

    def learn(self, model, *, steps, dataset, on_step):
        # Run the algorithm's own interaction and optimization loop.
        # `dataset` is a validated DatasetReader or None.
        # Report exactly one event after each env.step().
        for step in range(1, steps + 1):
            reward, terminated, truncated = model.train_one_environment_step()
            on_step(aiogym.TrainingStep(
                step=step,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            ))
        return steps

    def save(self, model, payload):
        # Retain parameters, optimizer state, replay, and learner counters.
        model.save(payload)  # Must create the requested payload.zip file.

    def load(self, payload, *, env=None):
        return MyModel.load(payload, env=env)

    def policy(self, model, *, checkpoint):
        return MyPolicy(model=model, checkpoint=checkpoint)

    def runtime_metadata(self):
        return {"my_library": my_library.__version__}


backend = MyBackend()
aiogym.register_algorithm(backend)
```

`MyPolicy` implements the existing AIO-Gym `Policy` boundary: an `env`
attribute plus `reset(seed)`, `act(observation, context)`, and `metadata()`.
`act()` returns one action directly in the environment action space.

The training callback is strict. `learn()` reports consecutive steps starting
at one for each invocation, reports every environment transition exactly once,
and returns the last reported step. AIO-Gym combines those local steps with the
checkpoint's completed-step count to build the common training curve and to
trigger periodic evaluation. Training currently supports exactly one
environment; batching optimization updates internally does not change this
transition-reporting rule.

The backend owns only model-specific construction, learning, serialization,
and inference adaptation. AIO-Gym owns output-directory validation,
`metadata.json`, `training_curve.json/svg`, periodic evaluation,
`best/model.zip`, final evaluation, comparison, and environment compatibility.
`effective_kwargs()` and `runtime_metadata()` must return JSON-safe mappings.
`save()` must create the exact `payload.zip` path passed by AIO-Gym, and
`load()` must read that payload into a model that can either continue learning
or provide a Policy. The payload must retain the optimizer, replay, and learner
state owned by the algorithm. AIO-Gym wraps it with a self-describing
`manifest.json` in the public `model.zip` checkpoint. The same backend then
supports `train(..., resume_from=checkpoint)` without another workflow.

## Stable-Baselines3 algorithms

An SB3 `BaseAlgorithm` subclass does not need a custom backend. Register the
class directly:

```python
from stable_baselines3 import A2C

aiogym.register_sb3_algorithm("a2c", A2C)
```

It immediately uses the same `train()`, checkpoint, training-curve,
evaluation, comparison, and CLI paths. `SB3AlgorithmBackend` is also public for
installed packages that need to expose a backend instance through an entry
point. Behavior cloning remains disabled unless an explicit compatible hook is
supplied.

Direct registration is process-local. Register the class again before loading
its checkpoint in a fresh Python process, or publish the backend through the
entry point below for automatic CLI and loader discovery.

```python
# my_package/aiogym_backend.py
from stable_baselines3 import A2C
from aiogym import SB3AlgorithmBackend

backend = SB3AlgorithmBackend(id="a2c", model_class=A2C)
```

## Optional behavior cloning

Set `behavior_cloning = None` when the algorithm has no valid supervised actor
update. To enable demonstrations, expose a callable with the signature declared
by `AlgorithmBackend.behavior_cloning`; it receives the live model, validated
observation/action arrays, epochs, batch size, learning rate, seed, and Dataset
source metadata, and returns the behavior-cloning report mapping. This keeps
behavior cloning an explicit algorithm capability rather than assuming every
optimizer has an SB3-style actor.

The returned report is written as `behavior_cloning.json`. It must include the
common `dataset`, `transition_count`, `action_field`, `epochs`, `batch_size`,
and `learning_rate` fields used by training metadata; backend-specific loss or
optimizer fields may be added alongside them.

## Installed-package and CLI discovery

`register_algorithm()` registers a backend in the current Python process. An
installed external package can also expose the backend instance through the
standard Python entry-point group so a fresh `aiogym` CLI process discovers it:

```toml
[project.entry-points."aiogym.algorithms"]
my_algorithm = "my_package.aiogym_backend:backend"
```

The entry-point name must exactly equal `backend.id`. After installation,
`aiogym list algorithms`, `aiogym train ... my_algorithm`, checkpoint loading,
evaluation, and comparison all resolve the same backend. `load_policy()` reads
the algorithm id from the checkpoint manifest and requires the target `env` so
AIO-Gym can validate environment compatibility before calling the backend.
Duplicate ids, incomplete adapters, invalid ids, missing checkpoint files,
malformed training events, and unsupported behavior cloning fail immediately.

Set `requires_dataset = True` only when learning cannot run without a Dataset,
as with RLPD. AIO-Gym then validates `dataset=` before creating the output
directory and passes the resulting `DatasetReader` to `learn()`. A backend that
sets it to `False` may receive a Dataset only when its behavior-cloning hook
consumes that Dataset before online learning.
