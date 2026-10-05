# AIO-Gym

AIO-Gym is a process-control toolkit for simulation, classical control, reinforcement learning, and reproducible comparison. It lets PID, MPC, and learned policies run on the same environments and test cases, so users can compare stability, tracking quality, disturbance rejection, and safety.

## What you can do

- **Simulate a process:** use Gymnasium-compatible environments to inspect physical variables and advance a model with your own actions.
- **Run controllers:** use built-in PID, MPC, hold, and random policies, or load a trained policy.
- **Train policies:** use DDPG, PPO, SAC, TD3, or RLPD; add randomized operating conditions, physical disturbances, measurement noise, delays, and actuator faults.
- **Use recorded data:** collect a Dataset for behavior cloning or RLPD's offline-to-online learning.
- **Evaluate and inspect results:** compare policies on identical cases and save JSON metrics, SVG figures, and numeric trajectories; continue training from saved checkpoints.

## Start here

**Start with the [User guide](docs/user-guide.md).** One Heater example takes you from installation through SAC training, comparison with PID/MPC, and reading the results. Copy the complete script; training and comparison each run once.

For a specific task, jump directly to its section:

| Task                                                   | Read                                                                                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Train your own algorithm or evaluate an external model | [External algorithms](docs/external-algorithms.md)                                                                                   |
| Train, validate, or continue a policy                  | [Training](docs/user-guide.md#3-train-and-compare), [continuation](docs/user-guide.md#continue-training)                             |
| Compare policies and interpret results                 | [Comparison](docs/user-guide.md#evaluate-a-saved-model), [metrics and figures](docs/user-guide.md#4-read-the-results)                |
| Inspect an environment or configure a controller       | [Environment information](docs/user-guide.md#inspect-and-configure-an-environment), [MPC settings](docs/user-guide.md#configure-mpc) |
| Collect demonstrations                                 | [Dataset](docs/user-guide.md#collect-and-use-a-dataset)                                                                              |
| Run terminal commands or schedule training jobs        | [CLI guide](docs/cli.md)                                                                                                             |
| Combine offline data with online RL                    | [RLPD training](docs/rlpd.md)                                                                                                        |

The CLI runs the same workflows and adds terminal job management; choose it when you prefer commands to Python scripts.

## Included scenarios

Once you know the workflow, choose a process below. Each guide supplies the physical interface, default task, success criteria, controller settings, training differences, and fixed Benchmark protocols for that scenario.

| Scenario id                                            | Process                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| [`cascade`](docs/scenarios/cascade.md)                 | Heated three-tank cascade with configurable heaters and a dynamic closed reservoir |
| [`crystallization`](docs/scenarios/crystallization.md) | Batch crystallization with endpoint quality targets                                |
| [`cstr`](docs/scenarios/cstr.md)                       | Two-input, two-output stirred-tank reactor                                         |
| [`extraction`](docs/scenarios/extraction.md)           | Five-stage counter-current extraction                                              |
| [`heater`](docs/scenarios/heater.md)                   | Fired heater; used in the introductory examples                                    |
| [`hvac`](docs/scenarios/hvac.md)                       | Two-zone HVAC                                                                      |
| [`quadruple`](docs/scenarios/quadruple.md)             | Quadruple-tank laboratory process                                                  |
| [`three_tank`](docs/scenarios/three_tank.md)           | Hydraulic three-tank cascade                                                       |

## Results and support

Your runs generate results locally under the output paths you choose. The [published benchmark snapshot](benchmark-results/README.md) provides existing figures and downloadable artifacts with their recorded provenance. It is a historical snapshot, not a reproduction of the current checkout.

For your own study, follow the [evaluation guidance](docs/user-guide.md#metrics-and-ranking). Short training examples establish that the workflow runs; meaningful RL comparisons also require planned training budgets and independent training seeds.

Report bugs and documentation issues in the [issue tracker](https://github.com/supcon-international/aiogym/issues).
