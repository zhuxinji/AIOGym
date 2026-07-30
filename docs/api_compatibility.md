# API compatibility after functional closure

The functional-closure release deliberately narrows the stable surface. It
does not preserve superseded construction, sampling, or trainer entry points.

## Stable environment API

| Before | After | Migration |
| --- | --- | --- |
| Gym IDs such as `AIOGym/...-v0` | no registration side effect | call `aiogym.make_env()` |
| public environment classes and broad kwargs | one factory with direct/config modes | put advanced settings in `config["environment"]` |
| constructor seed | reset seed | call `env.reset(seed=...)` |
| `TrackSpec.make_env()` | internal resolved builder | load a Track, then use benchmark/training services |
| `RLEnvironmentFactory` | removed | use the resolved training plan |

The top-level package exports only:

```text
__version__
evaluate_controller
list_cases
list_controllers
list_scenarios
list_tracks
load_case
load_track
make_controller
make_env
```

## Training and data

| Before | After | Migration |
| --- | --- | --- |
| `aiogym train sb3 ...` | `aiogym train --config FILE` | select `algorithm_id` in `RLTrainingConfig` v2 |
| `aiogym train rlpd ...` | `aiogym train --config FILE` | bind a Dataset v2 path/ID/hash in config |
| `python -m aiogym.rl.train_offline` | unified `train` command with `algorithm_id: "bc"` | use the BC adapter |
| `CaseMixtureEnv` | sampler-driven `EpisodeSpec` injection | use Track distributions and the training runner |
| `aiogym.rl.transitions.TransitionDataset` | Dataset v2 | collect with `aiogym collect --config FILE` |
| RLPD-local replay class | `aiogym.rl.replay.ReplayBuffer` | no user migration needed |

Historical `aiogym.transition.v1` rows have a deliberately narrow, one-way
bridge:

```python
from aiogym.compat.transitions import TransitionDataset
from aiogym.datasets import migrate_transition_dataset
```

The compatibility model validates legacy rows only. It does not collect new
data, create training views, or convert Dataset v2 back to the old format.

## Validation and experimental APIs

Training and HPO use one fixed validation-plan implementation. Test data is
available only through the one-shot `aiogym final-test --config FILE` path.

Constrained RL, recurrent observation contracts, and safety shields remain
explicitly experimental under `aiogym.experimental.rl`; they are not exported
from `aiogym.rl` or the top-level package. Stable algorithm discovery is
`aiogym.rl.list_algorithms()` and returns BC, PPO, RLPD, SAC, and TD3.
