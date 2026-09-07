# Benchmark results

Prepared snapshot `20260907T100413Z` from the 24 standard comparisons in `runs/<scenario>/benchmarks/`.

Each comparison contains 20 evaluation cases (seeds 0 through 19). Policy coverage and rankings below come directly from each comparison file. Rankings are local to each benchmark and use its recorded ranking metrics.

## Results

| Scenario | Benchmark | Policy ordering | Figure |
|---|---|---|---|
| cascade | boundary-safety | pid > mpc > sac > ppo > td3 > ddpg > rlpd | [View](figures/cascade/boundary-safety.svg) |
| cascade | disturbance-rejection | pid > mpc > ppo > sac > td3 > ddpg > rlpd | [View](figures/cascade/disturbance-rejection.svg) |
| cascade | tracking | mpc > pid > sac > ppo > td3 > ddpg > rlpd | [View](figures/cascade/tracking.svg) |
| crystallization | boundary-safety | mpc > rlpd > td3 > ddpg > ppo > pid > sac | [View](figures/crystallization/boundary-safety.svg) |
| crystallization | disturbance-rejection | mpc > pid > ppo > rlpd > td3 > ddpg > sac | [View](figures/crystallization/disturbance-rejection.svg) |
| crystallization | tracking | ddpg > ppo > td3 > mpc > pid > rlpd > sac | [View](figures/crystallization/tracking.svg) |
| cstr | boundary-safety | ppo > pid > mpc > rlpd > td3 > ddpg > sac | [View](figures/cstr/boundary-safety.svg) |
| cstr | disturbance-rejection | pid > ppo > mpc > rlpd > sac > td3 > ddpg | [View](figures/cstr/disturbance-rejection.svg) |
| cstr | tracking | ppo > mpc > pid > rlpd > sac > td3 > ddpg | [View](figures/cstr/tracking.svg) |
| extraction | boundary-safety | rlpd > sac > mpc > ppo > ddpg > pid > td3 | [View](figures/extraction/boundary-safety.svg) |
| extraction | disturbance-rejection | pid > ppo > rlpd > ddpg > sac > mpc > td3 | [View](figures/extraction/disturbance-rejection.svg) |
| extraction | tracking | rlpd > sac > mpc > ddpg > pid > td3 > ppo | [View](figures/extraction/tracking.svg) |
| heater | boundary-safety | td3 > rlpd > sac > pid > mpc > ddpg > ppo | [View](figures/heater/boundary-safety.svg) |
| heater | disturbance-rejection | pid > mpc > rlpd > sac > td3 > ddpg > ppo | [View](figures/heater/disturbance-rejection.svg) |
| heater | tracking | mpc > pid > sac > rlpd > td3 > ddpg > ppo | [View](figures/heater/tracking.svg) |
| hvac | boundary-safety | pid > ppo > ddpg > td3 > sac > rlpd > mpc | [View](figures/hvac/boundary-safety.svg) |
| hvac | disturbance-rejection | pid > ppo > ddpg > sac > rlpd > td3 > mpc | [View](figures/hvac/disturbance-rejection.svg) |
| hvac | tracking | pid > ppo > sac > rlpd > td3 > ddpg > mpc | [View](figures/hvac/tracking.svg) |
| quadruple | boundary-safety | mpc > pid > ppo > td3 > ddpg > sac > rlpd | [View](figures/quadruple/boundary-safety.svg) |
| quadruple | disturbance-rejection | pid > mpc > ppo > sac > td3 > ddpg > rlpd | [View](figures/quadruple/disturbance-rejection.svg) |
| quadruple | tracking | pid > mpc > ppo > rlpd > td3 > ddpg > sac | [View](figures/quadruple/tracking.svg) |
| three-tank | boundary-safety | pid > rlpd > ddpg > mpc > sac > td3 > ppo | [View](figures/three-tank/boundary-safety.svg) |
| three-tank | disturbance-rejection | pid > mpc > rlpd > ddpg > sac > td3 > ppo | [View](figures/three-tank/disturbance-rejection.svg) |
| three-tank | tracking | pid > mpc > rlpd > ddpg > sac > td3 > ppo | [View](figures/three-tank/tracking.svg) |

## Downloads and provenance

`summary.json` includes mean/median metrics, environment parameters, policy sources and evaluation seeds. `manifest.json` identifies the exact input and published files by SHA-256.

Download the complete comparison JSON, SVG and NPZ files as `aiogym-benchmarks-20260907T100413Z.tar.gz` from the [September 7 benchmark snapshot](https://github.com/supcon-international/aiogym/releases/tag/benchmark-results-20260907).

The comparison files do not record a verified generation-time source commit. The local checkout has uncommitted changes. These results must not be described as reproduced from the packaging HEAD or from a released version. Checkpoint hashes, when available, describe files at packaging time; they do not prove those files were unchanged since evaluation.

This snapshot packages existing results; it does not rerun simulations or training. Training directories, datasets, checkpoints and unrelated run outputs are excluded from the archive.

The previous v0.30.0 results remain available in Git history and the [v0.30.0 release](https://github.com/supcon-international/aiogym/releases/tag/v0.30.0).
