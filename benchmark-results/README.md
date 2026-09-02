# Published benchmark results

This directory contains the lightweight, reviewable benchmark evidence for
AIO-Gym 0.30.0. The results were regenerated from source commit `bb176a3` with
a clean Git worktree.

Every comparison uses fixed Benchmark cases selected by seeds 0 through 19.
The built-in PID and MPC controllers are included for every scenario. The
Three-Tank comparisons also evaluate the published DDPG, PPO, RLPD, SAC, and
TD3 checkpoints. Across all 24 comparisons, every evaluated policy completed
with an aggregate unsafe rate of zero.

## Files

- `figures/<scenario>/<benchmark>.svg` provides a GitHub-previewable overview.
- `summary.json` contains compact mean and median metrics, policy ordering,
  protocol seeds, and the source commit.
- `manifest.json` records SHA-256 hashes for these files and the corresponding
  GitHub Release assets.

## Policy ordering

The ordering below follows each Benchmark's declared ranking metrics. It is not
a cross-scenario score, and small return differences should be read from the
underlying distributions rather than treated as universal controller quality.

| Scenario | Benchmark | Ordering | Figure |
|---|---|---|---|
| Cascade | Boundary safety | PID > MPC | [View](figures/cascade/boundary-safety.svg) |
| Cascade | Disturbance rejection | PID > MPC | [View](figures/cascade/disturbance-rejection.svg) |
| Cascade | Tracking | MPC > PID | [View](figures/cascade/tracking.svg) |
| Crystallization | Boundary safety | MPC > PID | [View](figures/crystallization/boundary-safety.svg) |
| Crystallization | Disturbance rejection | MPC > PID | [View](figures/crystallization/disturbance-rejection.svg) |
| Crystallization | Tracking | MPC > PID | [View](figures/crystallization/tracking.svg) |
| CSTR | Boundary safety | PID > MPC | [View](figures/cstr/boundary-safety.svg) |
| CSTR | Disturbance rejection | PID > MPC | [View](figures/cstr/disturbance-rejection.svg) |
| CSTR | Tracking | MPC > PID | [View](figures/cstr/tracking.svg) |
| Extraction | Boundary safety | MPC > PID | [View](figures/extraction/boundary-safety.svg) |
| Extraction | Disturbance rejection | PID > MPC | [View](figures/extraction/disturbance-rejection.svg) |
| Extraction | Tracking | MPC > PID | [View](figures/extraction/tracking.svg) |
| Heater | Boundary safety | PID > MPC | [View](figures/heater/boundary-safety.svg) |
| Heater | Disturbance rejection | PID > MPC | [View](figures/heater/disturbance-rejection.svg) |
| Heater | Tracking | MPC > PID | [View](figures/heater/tracking.svg) |
| HVAC | Boundary safety | PID > MPC | [View](figures/hvac/boundary-safety.svg) |
| HVAC | Disturbance rejection | PID > MPC | [View](figures/hvac/disturbance-rejection.svg) |
| HVAC | Tracking | PID > MPC | [View](figures/hvac/tracking.svg) |
| Quadruple tank | Boundary safety | MPC > PID | [View](figures/quadruple/boundary-safety.svg) |
| Quadruple tank | Disturbance rejection | PID > MPC | [View](figures/quadruple/disturbance-rejection.svg) |
| Quadruple tank | Tracking | MPC > PID | [View](figures/quadruple/tracking.svg) |
| Three-Tank | Boundary safety | PID > RLPD > MPC > DDPG > SAC > TD3 > PPO | [View](figures/three-tank/boundary-safety.svg) |
| Three-Tank | Disturbance rejection | PID > RLPD > MPC > DDPG > SAC > TD3 > PPO | [View](figures/three-tank/disturbance-rejection.svg) |
| Three-Tank | Tracking | PID > MPC > RLPD > DDPG > SAC > TD3 > PPO | [View](figures/three-tank/tracking.svg) |

The complete `comparison.json`, `comparison.svg`, and `trajectories.npz`
triplets are distributed in the `aiogym-v0.30.0-benchmarks.tar.gz` asset on the
[v0.30.0 release](https://github.com/zhuxinji/AIO-gym/releases/tag/v0.30.0).
Best Three-Tank checkpoints and the RLPD demonstration dataset are separate
release assets so normal clones do not download generated binary data.

These simulation results describe the included models and fixed protocols.
They are not evidence of plant calibration, PLC safety, commissioning, or
real-equipment validation.
