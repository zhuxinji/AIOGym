# RLPD training

RLPD mixes a fixed Dataset with new environment interactions throughout training. It uses the same training, checkpoint, and evaluation workflow as the [user guide](user-guide.md#3-train-and-compare). Method: [Ball et al. (2023)](https://arxiv.org/abs/2302.02948), [authors' implementation](https://github.com/ikostrikov/rlpd).

## Use a Dataset in the main example

First [collect a Dataset](user-guide.md#collect-and-use-a-dataset) compatible with your training environment. Then make these changes to the user guide's complete script:

| Where                            | Change                                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Example settings                 | `algorithm = "rlpd"`, `experiment = "rlpd-demo-1"`, `training_steps = 100_000`                             |
| Before creating the environments | Define `dataset_path = Path("runs/heater/datasets/pid-20/python-demo-1/seed-0")`, or use your Dataset path |
| Arguments to `aiogym.train()`    | Add `dataset=dataset_path`                                                                                 |

Run the modified script. The 100,000 steps count new online transitions, not the offline data; this example is not a performance guarantee. Checkpoint loading and comparison need no further changes.

## Algorithm settings

Set these keys in the main example's `algorithm_settings` dictionary:

| Key               | Default  | Purpose                                            |
| ----------------- | -------- | -------------------------------------------------- |
| `offline_ratio`   | `0.5`    | Fixed-Dataset share of each replay batch, in `0–1` |
| `batch_size`      | `256`    | Transitions per batch                              |
| `n_critics`       | `2`      | Critic ensemble size                               |
| `utd_ratio`       | `1`      | Updates per online transition                      |
| `learning_starts` | `10_000` | Random online transitions before gradient updates  |

For example, `algorithm_settings = {"n_critics": 10, "utd_ratio": 20}` increases ensemble size and computation. Report Dataset source and coverage alongside online training budgets and results.

## Continue an RLPD run

Use the [continuation recipe](user-guide.md#continue-training), setting `algorithm = "rlpd"` and the source/destination paths for your run. Define `dataset_path` and add `dataset=dataset_path` to `train()`. RLPD requires the same offline Dataset content when resuming; algorithm settings are restored from the checkpoint.
