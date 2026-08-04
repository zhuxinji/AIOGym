# API compatibility

The package version is `0.1.0`; there was no stable 1.0 consumer contract to
preserve. The v1 structural freeze therefore removes superseded beta APIs
directly instead of shipping parallel aliases.

| Removed beta surface | v1 path |
|---|---|
| concrete environment classes and Gym registration IDs | `aiogym.make_env(...)` |
| `env.py` / `env_factory.py` imports | top-level `aiogym.make_env(...)` |
| task, suite, and protocol catalogs | Case v2 and official Tracks |
| reward/objective dual configuration | RewardSpec for training, Track goal for ranking |
| `policy`, `sb3`, and `onnx` controller registry IDs | `aiogym.controllers.checkpoints` |
| backend trainer modules and their argument parsers | `aiogym train TARGET ALGORITHM` or `aiogym train --config FILE` |
| legacy Transition v1 and Dataset migration | collect and consume Dataset v2 |
| legacy evaluation artifact migration | regenerate current evaluation artifacts |
| benchmark backend-specific checkpoint flags | `--run`, or `--checkpoint`, `--algorithm`, `--sha256`, `--name` |
| direct scenario/Case benchmark mode | an official Track or benchmark config |

There is deliberately no compatibility package in the wheel. Archived beta
data should be converted with the historical release that produced it, then
recollected or regenerated under the current Dataset v2 and evaluation
contracts.

Checkpoint safety remains strict: pin the digest, load through the canonical
loader, and do not deserialize untrusted files.
