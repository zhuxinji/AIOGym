# RLPD

AIO-Gym's native RLPD backend implements the offline-to-online method from:

- Philip J. Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine,
  [Efficient Online Reinforcement Learning with Offline Data](https://arxiv.org/abs/2302.02948),
  ICML 2023;
- the authors' [reference implementation](https://github.com/ikostrikov/rlpd).

The implementation is a native PyTorch adaptation, not a vendored copy of the
reference JAX runner. It uses AIO-Gym Dataset v2, environment action bounds,
training callbacks, and the common `model.zip` checkpoint.

## Training semantics

RLPD requires one Dataset compatible with the training environment:

```python
import aiogym


trained = aiogym.train(
    env=training_env,
    algorithm="rlpd",
    dataset="runs/data",
    steps=100_000,
    output="runs/quadruple/training/rlpd/seed-0",
)
```

It consumes `observation`, `commanded_action`, `reward`, `next_observation`, and
`terminated`. Truncated time-limit transitions remain eligible for value
bootstrapping. Dataset actions are mapped from physical environment bounds to
the actor's normalized `[-1, 1]` space.

By default every optimizer batch contains exactly half immutable Dataset
transitions and half transitions from the growing online replay. Set
`algorithm_kwargs.offline_ratio` in the inclusive range `[0, 1]` to change the
split. The offline sample count is `int(batch_size * offline_ratio)` and the
rest of the batch comes from online replay. Dataset samples therefore remain
active after online interaction starts whenever the ratio is greater than
zero; this is not behavior-cloning warm-start. Each environment transition is
still reported exactly once to the common training curve and periodic-
evaluation workflow.

The defaults follow the reference locomotion setup's main RLPD choices:

- batch size `256`, split 128/128 between prior and online replay;
- 10 LayerNorm critics and a random minimum over 2 critics;
- update-to-data ratio `20`;
- 10,000 random online exploration transitions before gradient updates;
- two 256-unit actor and critic hidden layers;
- discount `0.99`, target interpolation `0.005`, and learning rate `3e-4`.

Override these values through the ordinary JSON `algorithm_kwargs` mapping.
`offline_ratio=0` produces online-only batches and `offline_ratio=1` produces
offline-only batches after the online collection gate. The default `0.5`
retains the paper's canonical symmetric sampling.

## AIO-Gym-specific scope

The backend uses one mixed Dataset/online-replay training phase and the common
`model.zip` checkpoint path. AIO-Gym owns validation, best-checkpoint selection,
training artifacts, loading, evaluation, and comparison. The saved payload
contains the actor, critics, target critics, all optimizer states, entropy
temperature, online replay, random-generator state, environment-step count, and
sampling/update counters. Continued RLPD training supplies the same Dataset and
uses the ordinary `train(..., resume_from=...)` path.

Short runs and the unit tests establish interface and lifecycle correctness,
not controller performance. A claimed RLPD result still requires a documented
Dataset, meaningful environment-step budget, held-out evaluation, and the same
20-case Benchmark comparisons used for other policies.
