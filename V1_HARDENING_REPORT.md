# AIOGym v1 Hardening Report

Date: 2026-08-03
Repository: `/Users/zhuxinji/Desktop/AIO/AIO-Gym`
Branch / baseline HEAD: `main` / `f9dde638b18fe332ccc01baa981f2937d30136ac`
Runtime used for final validation: Python 3.14.6, pip 26.1.2

## Summary

Phases 0 through 7 of the v1 hardening plan were implemented. The work aligns
source, package resources, runtime contracts, hashes, provenance, CI, and
documentation while preserving the existing public environment and CLI entry
points.

The rebuilt wheel passed the release archive checker and was installed outside
the source tree. The installed package completed import/list/profile/env smoke,
collection and training dry-runs, a 1,000-transition SAC smoke, checkpoint
reload, validation evaluation, and benchmark comparison. There are no local
implementation blockers.

Remote GitHub Actions has not been run for this uncommitted worktree, so the new
Python 3.10-3.13 CI matrix remains to be verified after publication. The origin
still identifies a temporary repository, so project URLs were not guessed or
changed.

The worktree was already dirty with cumulative CLI/module cleanup changes when
this plan started. Those changes were preserved; no reset, stash, commit, or
push was performed.

## Baseline

- `git branch --show-current` reported `main`.
- `git rev-parse HEAD` reported
  `f9dde638b18fe332ccc01baa981f2937d30136ac`.
- `python --version` reported Python 3.14.6.
- `python -m pip --version` reported pip 26.1.2.
- `git remote get-url origin` reported
  `git@github.com:zhuxinji/AIO-gym-temp.git`.
- The initial worktree contained pre-existing, unstaged cumulative changes.
  No files were staged.
- The stale wheel contained `aiogym/_environment/runtime.py`, did not contain
  `aiogym/_environment/env.py`, and omitted current modules including
  `controllers/export.py`, `cli/describe.py`, `datasets/profiles.py`,
  `rl/run_claim.py`, and `rl/run_reference.py`.
- Generated metadata, caches, stale release archives, and ignored run outputs
  were removed before rebuilding. `runs/.gitignore`, golden fixtures, built-in
  cases/tracks/anchors, and package resources were retained.

## Confirmed fixes

| Confirmed issue | Status | Resolution |
|---|---|---|
| Stale wheel differed from source | fixed | Clean build now contains current modules, excludes legacy `runtime.py`, and passes an archive-content gate. |
| macOS metadata, caches, egg-info, runs, and old dist | fixed | Expanded `.gitignore`; removed generated copies and rebuilt from clean source. Final `build/` and `aiogym.egg-info/` were removed; `dist/` is ignored and retained only for local verification. |
| `LagrangianSAC` import/registry mismatch | fixed | Removed from stable config/CLI/SB3 adapter registries; experimental direct-call import remains available. |
| Reward and evaluation action-bound mismatch | fixed | Both use `_internal.control_math.normalized_action()` and canonical `bounds`. |
| Integral observation used one global limit | fixed | Added model-provided per-channel limits with construction-time validation and channel-wise clipping/normalization. |
| Pytest could collect production `final_test.py` | fixed | Restricted discovery to `aiogym/tests/test_*.py`; collect-only verification excludes production modules. |
| Rewards depended on evaluation tracking helpers | fixed | Tracking/action normalization moved to neutral internal control math; AST boundary tests enforce direction. |
| Guided profiles had duplicate Python/root JSON sources | fixed | Eight validated JSON resources are the single packaged source; duplicate root configs were removed. |
| Stable hash/file hash/atomic JSON logic was duplicated | fixed | Centralized serialization helpers and migrated priority call sites with digest regression tests. |
| `BackendResult` lived in dispatcher `__init__.py` | fixed | Moved contract and validation to `rl/backends/contracts.py`, with compatibility re-exports. |
| `spec_hash` mixed MDP and instrumentation identity | fixed | Added `mdp_hash` and `runtime_hash`; retained `spec_hash == runtime_hash` compatibility. |
| Profiler had unconditional Unix dependency and guessed RSS units | fixed | Uses process CPU time, lazy `resource` access, explicit platform conversions, and nullable unavailable memory values. |
| Multiprocessing default was `fork` | fixed | SAC/TD3/PPO now default to `spawn`; `forkserver` and `fork` require an explicit valid selection. |
| Long orchestration functions obscured phase boundaries | fixed | Locally split the five named targets without adding a new public builder or changing signatures. |
| CI/release gates were incomplete | fixed locally | Added core matrix, package/wheel-only, RL smoke, and scheduled/manual optional workflows. Remote execution is deferred until publication. |
| Version/provenance were not fully unified | fixed | Distribution metadata is the version source; provenance records runtime/dependency information without eager optional imports. |

## Implementation details

### Repository and packaging

- Setuptools now discovers `aiogym*` packages automatically and excludes
  `aiogym.tests*` from the wheel.
- Package data includes controllers, cases, tracks, anchors, examples,
  parameters, and recursive profile resources.
- `scripts/check_release_artifacts.py` validates required and forbidden archive
  members and has five focused tests.
- Standard setuptools sdists necessarily contain
  `aiogym.egg-info/SOURCES.txt`. The checker has one narrow sdist-only exception
  for that generated manifest. Any other egg-info member is rejected, and the
  wheel rejects all egg-info members.

### Contracts and identity

- Stable algorithm adapters are exactly `ppo`, `sac`, and `td3`.
- Stable training algorithms remain `bc`, `ppo`, `rlpd`, `sac`, and `td3`.
- Reward and scorecard action normalization now share the same bounds,
  dimensionality checks, finite checks, and documented fallback.
- Cascade and cascade-recirculating integral observations use explicit
  per-channel limits instead of environment-level level/temperature constants.
- `mdp_hash` excludes pure instrumentation fields such as `info_level` and
  `profile_timing`; `runtime_hash` retains the previous full runtime identity.
- Existing `env_spec_hash` semantics and stable artifact schema meanings remain
  unchanged.

### Profiles and serialization

- Package resources now hold six training and two collection quick profiles.
- Loader results retain existing profile contracts; `source` now uses a stable
  `package:aiogym/...` identifier that is valid in an installed wheel.
- Root quickstart `collect`, `sac`, `bc`, and `rlpd` duplicates for cascade and
  quadruple were removed. Repository-only benchmark examples remain.
- Canonical JSON bytes, stable JSON hashing, file SHA-256, and atomic JSON writes
  are centralized while high-level artifact claim/overwrite behavior remains in
  its owning APIs.

### Portability, provenance, and maintenance

- Stable multiprocessing defaults use `spawn`.
- Runtime provenance includes Python, platform, machine, AIOGym, NumPy,
  Gymnasium, optional dependency versions, CUDA/device data when safely
  available, torch thread count, vector backend, and start method.
- Importing core AIOGym does not import Torch, Stable-Baselines3, CasADi, ONNX,
  or ONNX Runtime solely for provenance.
- Ruff now checks `E4`, `E7`, `E9`, `F`, `I`, `UP`, and `B`, including tests.
  A small global historical backlog is explicitly ignored; `SIM` is deferred.
- README and architecture/API documentation now distinguish stable, preview,
  and environment-only scenario maturity and describe packaged profile sources.

## Tests and commands

The following commands were actually run against the final cumulative source
unless a baseline or intermediate failure is explicitly stated.

- PASS: `.venv/bin/python -m compileall -q aiogym`.
- PASS: `.venv/bin/ruff check aiogym` — `All checks passed!`.
- PASS: `.venv/bin/pytest -q -m "not rl and not oracle and not onnx and not e2e"`
  — 574 passed, 37 deselected.
- PASS: `.venv/bin/pytest -q -m rl` — 19 passed, 591 deselected, 2 warnings.
- PASS: `.venv/bin/pytest -q -m oracle` — 11 passed, 599 deselected.
- PASS: `.venv/bin/pytest -q -m onnx` — 3 passed, 607 deselected, 2 warnings.
- PASS: `.venv/bin/pytest -q -m e2e` — 6 passed, 604 deselected.
- PASS: `.venv/bin/pytest -q` outside the restricted sandbox — 611 passed,
  2 warnings in 40.77 seconds.
- ENVIRONMENT-ONLY FAILURE: the same full command inside the restricted sandbox
  produced 610 passed and one `PermissionError` at
  `os.sysconf("SC_SEM_NSEMS_MAX")` in the parallel dataset test. The identical
  suite passed outside the sandbox; this is not treated as a repository failure.
- PASS: `.venv/bin/pytest -q aiogym/tests/test_release_artifacts.py` — 5 passed.
- PASS: `git diff --check`.
- PASS: deterministic same-seed/fixed-action two-run comparisons for cascade,
  CSTR, and quadruple produced identical observations, rewards, termination
  state, and canonical trace digest within each pair.

The two warnings come from PyTorch's legacy TorchScript-based ONNX exporter and
its deprecated logging helper. They do not indicate a failed export or runtime
comparison.

## Hash regression

Official stable identities remained unchanged:

### Reward specs

- `economic-v1`: `c3efc6a2317838f0db3e7f719caf42a097421a51dae77fa80ba4ae8e8d408c0f`
- `regulation-v1`: `fb78375cdbd12dbe4f6544f92d3554f62c1f61510f2e28bf7473590bf33af891`

### Tracks

- `cascade-economic-specialist-v1`: `b1add7956f00b4f11784a18b007f3c695a13250ff83075569a1020adf75bdc9c`
- `cascade-recirculating-recovery-diagnostic-v1`: `cb5a3c23568a407686573912e87ed772f80fd43473809562e91958bb298e90ed`
- `cascade-recirculating-regulation-generalist-v1`: `3b7c280cd79087d2731a38b7baae72d95044aa699a2f5f5d3c48321676b03fbb`
- `cascade-recovery-diagnostic-v1`: `7d2520d10aca5cb23d53ba3349b6b0fa7ebe3cbade74f4e827fb4c0574bd4cbd`
- `cascade-regulation-generalist-v1`: `e078072c8b61e68de84393af360613923fa295b729fa4b94d07a62ed95a957ca`
- `quadruple-regulation-generalist-v1`: `112931a0adab0e01d6d4c26c4187ad05a84d8cf1e6ac1c886350915068d284c6`

### Official anchors

- Cascade economic: `4e1bb74862d681f90bb1638d742e288817273e2c2816e731f129622f93d1c20d`
- Cascade recirculating: `ea229ea4816d1d9e3efce8103ce79de2857cb01cb245611ae8d3acc98a9f1c43`
- Cascade regulation: `c2143b13098a63068b7cdb96c7c258bc3e0a7d80d6223a1289c2f29f44635de7`
- Quadruple regulation: `a41a6806a027a87a22923b631377e865c9bd540e8a23f322c4d874e90ed2db27`

Regression tests also cover distribution, training/final-test config, run-claim,
checkpoint, dataset file, legacy `spec_hash`, and profile resolved-config
identity. No unintended stable digest change was accepted.

## Artifact verification

- Wheel: `dist/aiogym-0.1.0-py3-none-any.whl`
  - SHA-256: `4bb99c629f745b1c44699c43b21e259ea2f12ba7ad80914d1b34c88a17ee43bc`
- Source distribution: `dist/aiogym-0.1.0.tar.gz`
  - SHA-256: `9472eda1a4a915a10f97677165fab69420ba7c2d87d5dc859a1745a31a13e807`
- PASS: `python scripts/check_release_artifacts.py <wheel> <sdist>`.
- Required current modules are present, including `env.py`, controller export,
  CLI describe, dataset profiles, run claims, and run references.
- Forbidden content is absent: legacy `runtime.py`, tests, caches, bytecode,
  runs, datasets, checkpoints, and wheel egg-info.
- PASS: fresh venv installed the exact final wheel outside the source tree;
  `pip check` reported no broken requirements; core import came from the venv and
  did not import Torch or Stable-Baselines3.
- PASS: installed CLI list/describe commands, package profile loading, and
  `make_env/reset/step` smoke.
- PASS: installed-wheel collection and training dry-runs.
- PASS: installed-wheel 1,000-transition SAC training, checkpoint reload,
  evaluation, and PID-vs-SAC benchmark. The AIOGym import path was the fresh
  venv's `site-packages`; locally installed optional RL dependencies were made
  available from the development environment without importing AIOGym source.

The generated `dist/` is ignored and was not staged. It is retained locally for
inspection; release archives should be rebuilt by CI for publication.

## Compatibility

- Stable Python entry points `aiogym.make_env()`, `list_scenarios()`, and
  `list_tracks()` are unchanged.
- Stable CLI command names remain `list`, `describe`, `collect`, `train`,
  `evaluate`, `benchmark`, and `final-test`.
- No Scenario, RewardSpec, KPI, Scorecard metric, Track, Case, distribution,
  public environment builder, or backend-specific CLI was added.
- Reward weights, ranking weights, anchors, safety gates, physical parameters,
  and stable schemas were not changed.
- `LagrangianSAC` remains available from `aiogym.experimental.rl`, but is no
  longer exposed as a stable training/CLI/SB3 adapter choice. This matches its
  intended experimental boundary.
- The default subprocess start method changes from `fork` to `spawn`. Explicit
  `forkserver` and `fork` remain supported where the platform supports them.
- Guided profile source strings migrate from repository-relative config paths to
  packaged `package:aiogym/...` resources. Consumers treating `source` as an
  opaque provenance field need no migration; code opening it as a filesystem
  path must use profile loaders instead.
- `mdp_hash` and `runtime_hash` are additive. Existing `spec_hash` and artifact
  `env_spec_hash` keep their v1 meanings, so no artifact migration is required.

## Deferred

- Confirm the official public repository URL. The configured origin is still
  `zhuxinji/AIO-gym-temp`, so project URL fields were deliberately left intact.
- Run and inspect the new GitHub Actions workflows after the worktree is
  committed and published. Local and static validation cannot claim remote CI
  success.
- Migrate Dataset/policy compatibility decisions from legacy `spec_hash` to
  `mdp_hash` only in a future explicit schema version.
- Further decouple controller inference from RL and dataset rollout/safety
  contracts when a separate task justifies those boundary changes.
- Broaden typing and enable Ruff `SIM` after addressing the historical lint
  backlog. Current historical ignores are `B009`, `B905`, `E402`, `E702`,
  `E731`, `I001`, `UP035`, and `UP037`.
- Scenario L5 physical calibration remains outside this repository-hardening
  task.

## Working tree and cleanup notes

- No commit, tag, push, release, or remote workflow run was created.
- No changes were staged.
- Ignored run results removed during hygiene cleanup were generated local
  artifacts and are not recoverable from Git. Source fixtures and golden
  artifacts were preserved.
- Final build intermediates `build/` and `aiogym.egg-info/` were removed.
- The report and source changes remain unstaged for maintainer review.
