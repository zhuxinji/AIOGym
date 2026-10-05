# Train your own algorithm with AIO-Gym

Use AIO-Gym as a Gymnasium simulation environment for your own training code. Follow the [installation guide](user-guide.md#1-install-aio-gym) and install your algorithm's dependencies separately. Then follow [environment API](#1-use-the-simulation-environment), [evaluation](#2-evaluate-the-trained-policy), and [optional data collection](#3-collect-a-dataset-optional).

## 1. Use the simulation environment

`aiogym.make_env()` returns a Gymnasium environment with the standard `reset()` / `step()` API. Pass it directly to your Gymnasium-compatible training code. If you already know Gymnasium, go directly to [simulation behavior](#simulation-behavior) and use the API table as a reference.

| API                                               | Meaning                                                                                                                                               |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env = aiogym.make_env("heater", randomize=True)` | Creates the scenario environment; `randomize=True` samples feasible initial conditions and targets on reset.                                          |
| `env.reset(seed=...)`                             | Starts a new episode and returns `(observation, info)`. Supply a seed to reproduce a case; omit it to continue the random sequence.                   |
| `env.step(action)`                                | Advances one control interval and returns `(observation, reward, terminated, truncated, info)`.                                                       |
| `env.observation_space`, `env.action_space`       | Gymnasium spaces defining array shapes and bounds; the algorithm must support continuous `Box` actions.                                               |
| `env.observations`, `env.actions`                 | Ordered channel definitions, including physical meanings, units, and normalization.                                                                   |
| `env.describe()`                                  | Independent environment and current-episode description; `env.describe()["environment"]` includes the step duration `control_dt` and its `time_unit`. |
| `env.close()`                                     | Releases environment resources; `with aiogym.make_env(...) as env:` calls it on exit.                                                                 |

For `step()` in AIO-Gym, `terminated=True` indicates a safety violation or task-defined batch completion, which can be successful; `truncated=True` indicates a time limit. These use Gymnasium's standard ending flags.

For parameters and variations, see [environment settings](user-guide.md#inspect-and-configure-an-environment).

### Simulation behavior

Here, `info` is the dictionary returned by `reset()` / `step()`; `env.describe()` separately describes the environment and current episode.

| Behavior                       | AIO-Gym rule                                                                                                                                                                                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Episode reset                  | The environment does not automatically reset when an episode ends.                                                                                                                                                                                       |
| Action validation              | Actions must be finite and within `env.action_space`; invalid actions raise an error instead of being clipped.                                                                                                                                           |
| Requested vs. applied action   | After `step(action)`, `info["commanded_action"]` is the command you sent; `info["applied_action"]` is the action actually used by the process model. Delay, faults, and rate limits can make them differ.                                                |
| Observation vs. internal state | The returned `observation` is the algorithm's input, with channels described by `env.observations`. `info["true_state"]` contains the simulator's underlying physical state for diagnostics; observations may be partial, normalized, noisy, or delayed. |

## 2. Evaluate the trained policy

`model` below is your trained model. Its prediction function must return only an action; for SB3's `(action, state)` return value, use the adapter below.

Create a separate `test_env` with the observation/action interface used in training. This example uses the scenario's default parameters and its fixed `tracking` benchmark:

```python
import aiogym

scenario = "heater"
output = f"runs/{scenario}/demos/external-evaluation-001"

with aiogym.make_env(scenario, benchmark="tracking") as test_env:
    result = aiogym.evaluate(
        env=test_env,
        policies={"mine": model.predict},
        seeds=range(20),
        output=output,
    )

print("My metrics:", result["evaluations"]["mine"]["aggregate"])
print("Figure:", f"{output}/comparison.svg")
```

For an SB3 model, define this function before evaluation and pass `policies={"mine": predict_action}`:

```python
def predict_action(observation):
    action, _ = model.predict(observation, deterministic=True)
    return action
```

To compare policies, add entries such as `"other_rl": other_model.predict`, `"pid": "pid"`, or `"mpc": "mpc"`. Each prediction function must return only an action. To include an AIO-Gym checkpoint, add `"saved": aiogym.load_policy(saved_model_path, env=test_env)` inside the `with` block.

If training used custom parameters, create `test_env` with the same parameters and omit `benchmark`; see [test conditions](user-guide.md#evaluate-a-saved-model). Keep final test cases separate from model selection.

Omit `output` to evaluate without writing files. To interpret the results, follow [Read the results](user-guide.md#4-read-the-results).

## 3. Collect a Dataset (optional)

Follow [Collect and use a Dataset](user-guide.md#collect-and-use-a-dataset) with your scenario. Replace `policy="pid"` with `policy=model.predict`, or `policy=predict_action` for the SB3 adapter above. Your model chooses the actions, and `collect()` saves the environment interactions as a Dataset.
