# RL Experiment Readiness Report

> Historical note: the per-experiment `run_*.sh` launchers mentioned in this
> report were later replaced by `aiogym run <experiment.json>`. Recorded results
> and old artifact paths remain historical evidence, not current usage guidance.

Last updated: 2026-08-03

## Baseline

- Repository: `/Users/zhuxinji/Desktop/AIO/AIO-Gym`
- Branch: `main`
- Baseline commit: `f9dde638b18fe332ccc01baa981f2937d30136ac`
- Python: 3.14.6
- pip: 26.1.2
- Worktree: dirty before this plan; all pre-existing changes are preserved and
  no files are staged.

### Available optional dependencies

| Dependency | Version |
|---|---:|
| NumPy | 2.5.0 |
| Gymnasium | 1.3.0 |
| Torch | 2.13.0 |
| Stable-Baselines3 | 2.9.0 |
| CasADi | 3.7.2 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.27.0 |

### Baseline checks

- PASS: `python -m compileall -q aiogym scripts`.
- PASS: `pytest -q -m "not e2e and not rl and not oracle and not onnx"`
  outside the restricted sandbox: 574 passed, 37 deselected.
- ENVIRONMENT LIMITATION: the same core test command inside the sandbox had one
  `PermissionError` while reading `SC_SEM_NSEMS_MAX`; 573 other tests passed.
- PASS: `ruff check aiogym scripts`.
- PASS: `git diff --check`, apart from a pre-existing CRLF conversion warning
  for `aiogym/controllers/tuning/tune_cstr_mpc.py`.

### Final-test lock status

- No final-test lock artifact exists outside tests and the local virtual
  environment.
- No final-test lock was created or consumed by this work.
- Test split evaluation remains prohibited until the explicit Final-Test Gate.

### Known deferred issues

- Existing v1 regulation Tracks remain compatibility/diagnostic artifacts and
  may have validation/test resolved-case overlap. They must remain loadable and
  their hashes must not change.
- V2 anchors are frozen from reviewed calibration candidates; future changes
  to either v2 Track require regenerating the affected anchor manifest.
- Reference budgets and dependency constraints remain unfrozen until the
  Cascade Gate C pilot is run locally and reviewed alongside Gate B evidence.
- Cascade economic control, L5 physical calibration, broad architecture
  refactors, and final-test execution are outside the current gate.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| Phase 0: baseline and change boundary | complete | Baseline checks and lock audit above. |
| Phase 1: final checkpoint selection | complete | Shared callback path plus eight final/resume selector regressions; no optimizer invoked. |
| Phase 2: split-isolated v2 protocols | complete | Two v2 Tracks, split audit, finite Cascade observation bounds, and 18 protocol tests. |
| Local Run Gate A: anchor calibration | complete | Both candidates passed static review, were copied byte-for-byte into builtin resources, and passed 119 focused regression tests. |
| Phase 3: explicit algorithm config | complete | Frozen SAC/TD3/PPO constructor kwargs and 82 no-training adapter/config regressions. |
| Phase 4: training-seed statistics | complete | Strict case matrices, robust statistics, consistency failures, and selected-step reporting. |
| Phase 5: experiment review tooling | complete | Preflight, runtime capture, review, SVG plot, text-only bundle, and pilot config/script. |
| Local Run Gate B: Quadruple SAC pilot | complete | Three 100k-transition seeds reviewed; conclusion A: pipeline/learning signal is sound, proceed to budget extension. |
| Local Run Gate C: Cascade SAC pilot | replacement complete; quality gate failed | The repaired pilot restored 3/3 eligibility and reduced variance, but every selected temperature-step policy remains below the bad anchor. |
| Phase 6: reference budget freeze | blocked by Cascade quality gate | The learning curves do not justify a budget extension; checkpoint selection and the training objective require review. No test rollout is authorized. |

## Protocol constraints

- Training, tuning, checkpoint selection, and pilot review may access only the
  training and validation splits.
- Codex will not call `model.learn()`, `aiogym train`, `run_experiment()`,
  `run_seed_sweep()`, `aiogym final-test`, or `FinalTestLock.consume()` under
  this plan.
- Anchor calibration was run by Codex only after explicit user authorization;
  training and full checkpoint evaluation remain user-local operations.

## Phase 1: final checkpoint selection

- Periodic, resume, and non-boundary final validation now use one
  `evaluate_checkpoint()` implementation.
- A final checkpoint can replace an earlier eligible best; a worse or ineligible
  final checkpoint cannot replace it.
- If the final step was already evaluated, selector and learning-curve records
  are not duplicated.
- Selected policy bytes, SHA-256 manifest, selection record, sidecar state,
  `best_step`, and `best_metric_value` are updated together.
- PASS: eight mock/fake-model selector regressions.
- PASS: 29 resume/checkpoint entrypoint tests and the existing SB3 callback
  episode-spec test.

## Phase 2: split-isolated regulation v2 protocols

- Added `quadruple-regulation-generalist-v2` with Track hash
  `694d966e31a3282124469516f4b2d57360c4212ace928a2b354931f2275125d1`.
- Added `cascade-regulation-generalist-v2`; after the Gate C training-population
  repair its current Track hash is
  `54c02b0c3ef9cce7a5a006b3f9df4b46d9256880cdebb87841468ead72e0a67e`.
- Both protocols use distinct v2 training/validation/test namespaces, omit
  `training.step_budget`, and have disjoint validation/test resolved Case
  hashes.
- Quadruple v2 preserves the v1 policy-facing contract and hides the held-out
  disturbance pair.
- Cascade v2 freezes fixed-schema observation normalization, finite observation
  bounds, and unmeasured disturbances. All declared fixed regulation targets
  pass the model steady-state feasibility check.
- Existing v1 declarations and hashes remain unchanged. Their historical split
  overlap is reported by the audit helper but does not block compatibility
  loading.
- PASS: 18 v2 protocol/schema/identity/feasibility and one-step tests.
- PASS: 65 related existing environment, benchmark, hash, and public-contract
  tests.

## Local Run Gate A: anchor candidate preparation

- `scripts/calibrate_official_anchors.py` retains the v1 suite and adds the
  separately selectable `v2-regulation` suite.
- Candidate generation refuses existing outputs and never writes directly to
  the builtin anchor directory.
- `scripts/experiments/review_anchor_candidates.py` verifies artifact, Track,
  ranking, Case coverage, utility direction/gap, eligibility, dependency
  provenance, and split isolation without running controllers.
- PASS: six candidate generation/review tool tests.
- PASS: pre-anchor core regression, excluding anchor-dependent and final-test
  lock tests: 575 passed, 46 deselected.
- COMPLETE: reviewed candidates and the static report are retained under
  `runs/protocol/` as local review evidence.

### Gate A retry 1

- The first local calibration reported both hold and MPC ineligible for
  `commissioning:validation-offset-v2`.
- Only that Cascade validation variant was revised: its mixed-direction initial
  offset was replaced by a common `+0.01 m / +1 degC` offset, while its event
  times remain v2-specific and its targets reuse the already feasible
  commissioning thermal stages.
- Split isolation, schema, policy contract, finite observation bounds, target
  feasibility, and all 23 focused protocol/tool tests still pass.
- Calibration failures now include per-controller, per-seed safety reasons when
  the evaluator supplies them.
- The already generated Quadruple v3 candidate was statically reviewed and
  passed: 7/7 expected Cases, no extra hashes, all bad/reference eligibility
  vectors true, non-degenerate gaps, valid artifact hash, and complete recorded
  dependency versions. Candidate artifact hash:
  `432e722bd66cc3bd5112519b13a32c23f448dfa22ac9bca348399994c5b0cf42`.
- Added a `v2-cascade` calibration suite so the failed Cascade candidate can be
  regenerated in the existing directory without rerunning or overwriting the
  accepted Quadruple candidate.

### Gate A retry 2 and freeze

- The Cascade-only retry exposed a second safety failure in
  `commissioning:heldout-v2`: both hold and MPC violated the state gate for all
  five evaluation seeds.
- A lower-inventory held-out initial condition also failed. The held-out Case
  therefore retains the validated commissioning initial state while preserving
  distinct v2 event times and target schedule; its resolved hash remains
  disjoint from validation.
- PASS: Cascade calibration for all six expected validation/test Cases.
- PASS: static review for both manifests: no missing/extra Case hashes, all
  bad/reference eligibility vectors true, non-degenerate gaps, matching Track
  and ranking identities, valid artifact hashes, and complete dependency
  provenance.
- Frozen Quadruple v2 anchor artifact hash:
  `432e722bd66cc3bd5112519b13a32c23f448dfa22ac9bca348399994c5b0cf42`.
- Frozen Cascade v2 anchor artifact hash:
  `ae9b0c20147758bc12152cf6ef25103e8fd4e2a1ba687363f1fc60088870fabf`.
- Candidate files were copied byte-for-byte into the builtin anchor directory;
  their utilities were not edited.
- PASS: 119 focused anchor-quality, Track, ranking, safety-gate, CLI, runner,
  and release-artifact tests.
- The final-test lock remains absent and unconsumed.

## Phase 3: explicit effective SB3 hyperparameters

- SAC now materializes and passes `ent_coef`, `target_entropy`, and the frozen
  ReLU `[256, 256]` actor/critic architecture.
- TD3 now materializes and passes normal action noise, sigma, policy delay,
  target policy noise/clip, and the frozen ReLU architecture. Noise arrays are
  derived from the normalized action-space shape.
- PPO now materializes and passes GAE, clipping, epochs, entropy/value
  coefficients, max gradient norm, and the frozen Tanh actor/value architecture.
- Activation names use an `elu`/`relu`/`tanh` whitelist; policy mappings reject
  unknown keys and non-positive layer sizes.
- Resolved config/hash and training metadata contain the serialized effective
  algorithm kwargs actually represented by the adapter.
- PASS: 82 relevant config, fake-constructor, checkpoint, and runner tests; no
  SB3 optimizer or `model.learn()` invocation.

## Phase 4: independent training-seed statistics

- Multi-seed summaries retain the existing `validation_summary` and add
  `training_seed_statistics` with ordered Case matrices, per-run metrics,
  mean/sample-std/median/IQM, stratified bootstrap CI, official-score
  mean/std, eligibility rate, worst seed, and selected checkpoint steps.
- Track/hash, validation plan/hash, metric/direction, Case order/hash, and
  finite-value consistency are mandatory; mismatches fail instead of being
  silently aligned.
- PASS: synthetic minimize/maximize, mismatch, checkpoint-step, bootstrap, and
  legacy-empty-summary coverage.

## Phase 5 and Local Run Gate B preparation

- Added validation-only `preflight_protocol.py`, static
  `review_training_run.py`, text/JSON-only `make_review_bundle.py`,
  dependency-free SVG `plot_learning_curves.py`, and read-only
  `capture_runtime_environment.py`.
- Quadruple v2 Track environments now expose finite physical observation Box
  bounds while v1/direct environment metadata remains unchanged. This closes
  the formal preflight bound check without changing dynamics or observations.
- The real Quadruple v2 preflight passed with three distinct training
  EpisodeSpecs, one neutral step, zero optimizer updates, and zero test
  rollouts. Track hash and frozen anchor hash still match Gate A.
- Added `configs/experiments/quadruple/sac-pilot-v2.json` and the then-current
  guarded pilot launcher. The current generic CLI dry-run resolves seeds 0/1/2
  and 100,000 transitions per seed.
- PASS: 57 focused experiment-tool/config/statistics tests.
- PASS: broad non-E2E/non-RL/non-oracle/non-ONNX regression: 620 passed,
  46 deselected. The sole sandbox multiprocessing semaphore check passed when
  rerun outside the restricted sandbox.
- At the Phase 5 preparation checkpoint no pilot had run; the subsequently
  completed Gate B pilot is reviewed below. Final-test and lock consumption
  remain prohibited.

## Local Run Gate B: Quadruple SAC pilot review

- All three independent training seeds completed exactly 100,000 environment
  transitions with 99,000 optimizer updates each.
- Selected checkpoint steps were 100,000, 80,000, and 70,000. The non-boundary
  final checkpoint participated in each selector; seed 0 selected its final
  checkpoint, while seeds 1 and 2 correctly retained better earlier validation
  checkpoints.
- Every selected policy was validation ranking-eligible. Validation reported no
  state, command, action, controller-created, or aggregate constraint
  violations and no protection, shield, or actuator interventions.
- Each seed recorded 112 unique training EpisodeSpecs. Throughput was stable at
  approximately 318--321 environment transitions/second.
- Validation regulation-cost-rate values by training seed were approximately
  0.000575, 0.012054, and 0.003912. IQM was 0.001706 with a 95% stratified
  bootstrap interval of approximately [0.000545, 0.012055].
- Minimum-phase performance improved substantially for every seed. The
  nonminimum-phase Case remains the bottleneck: all three selected policies
  scored below the bad fixed anchor on that Case, producing zero per-Case
  official score and a near-zero geometric aggregate score.
- Review conclusion: **A -- pipeline/learning signal normal; proceed to budget
  extension**. The evidence indicates under-training/high seed variance, not a
  protocol or implementation failure and not a clearly isolated hyperparameter
  defect.
- The best observed nonminimum-phase validation costs during the 100k curves
  were approximately 0.000918, 0.000956, and 0.001043, versus the fixed bad
  anchor cost of approximately 0.000892. The gap is therefore about 3%, 7%,
  and 17%, respectively: systematic and important, but close enough to support
  the under-training diagnosis rather than immediate hyperparameter tuning.
- The Quadruple pilot validation interval is changed from 10,000 to 2,000
  transitions for future runs, yielding about 51 observations over 100k while
  retaining the declared 10,000-transition checkpointing interval. Existing
  artifacts remain the original 10,000-transition observations and are not
  retroactively interpolated.
- Decision after Gate B: do not ignore the nonminimum-phase bottleneck, but do
  not tune against it yet. Proceed in protocol order to the Cascade pilot, then
  freeze a validation-only Quadruple budget extension. If the longer SAC run
  still fails to produce stable positive nonminimum-phase scores across seeds,
  classify that evidence as a controlled single-factor pilot trigger before
  reference-training freeze; do not proceed directly to final testing.
- The initial post-processing failure was a review-tool false positive: static
  `resolved_cases.test` Track metadata was mistaken for a test rollout. The
  guard now rejects only test evaluation structures with execution evidence
  such as results, aggregates, seeds, or episode-plan hashes. The real artifacts
  contain validation execution only.
- PASS: all three real run reviews and the combined review bundle. Bundle:
  `runs/pilots/quadruple/sac-v2/quadruple-sac-pilot-v2-review.zip`.
- No final-test lock was created or consumed.

## Local Run Gate C: Cascade SAC pilot preparation

- Added the frozen Cascade v2 SAC pilot configuration: 200,000 environment
  transitions, seeds 0/1/2, validation seed 5000, one dummy-vector environment,
  UTD ratio 1.0, replay capacity 500,000, learning starts 2,000, and complete
  validation every 20,000 transitions.
- The Track contract retains fixed-bounds normalized observations and hides the
  disturbance vector from the policy. The preflight performs only resets and
  one neutral action; it records zero optimizer updates and zero test rollouts.
- The historical guarded launcher captured runtime/preflight evidence, trained
  all three seeds, ran static review, rendered Case-aware learning curves, and
  created a text/JSON review bundle. The current `aiogym run` workflow refuses
  existing results unless `--overwrite` is explicitly passed.
- At this preparation checkpoint local training had not run; the subsequently
  completed initial Gate C pilot is reviewed below.

## Local Run Gate C: initial Cascade SAC pilot review

- All three seeds completed 200,000 transitions with stable throughput of about
  275 transitions/second and 144--155 distinct training EpisodeSpecs. Final
  checkpoint participation and artifact integrity passed; no test execution was
  detected and the review bundle passed its ZIP integrity check.
- Selected steps were 200,000, 20,000, and 80,000. Only seeds 1 and 2 selected
  eligible policies; seed 0 had no eligible trained validation checkpoint and
  fell back to an ineligible final policy. Aggregate eligibility was therefore
  2/3, below the reference-training requirement.
- Seed 0 recorded 1,877 state-violation steps across the two Cases, 903
  protection-intervention steps in commissioning, and a hard termination in
  temperature-step. Seeds 1 and 2 had no violations or interventions at their
  selected checkpoints.
- Commissioning learned on the two eligible seeds: costs 0.00937 and 0.01226
  beat the bad anchor cost 0.03626, but remain above the reference cost
  0.00316. Temperature-step is the common bottleneck: selected costs
  0.00301--0.00757 are about 13--33 times the bad anchor cost 0.000232.
- Learning curves are unstable rather than simply unfinished: seed 1 peaked at
  20k and seed 2 at 80k, while most later checkpoints became ineligible. Seed 1
  also placed at least one actuator at an action-space edge throughout each
  selected validation rollout; this saturation pattern was not consistent
  across all seeds, so it is diagnostic rather than a separate tuning target.
- Gate C conclusion: **B -- pipeline normal, current Cascade SAC optimization
  is unstable**. Do not extend the Cascade budget or freeze reference budgets.
  The controlled retry changes only learning rate from 3e-4 to 1e-4; Track,
  reward, network, UTD ratio, replay, budget, seeds, and validation protocol are
  unchanged. A separate output identity preserves the initial pilot evidence.

## Local Run Gate C: learning-rate 1e-4 retry review

- All three seeds completed 200,000 transitions and the review/bundle passed
  without warnings. Selected steps were 60,000, 20,000, and 200,000.
- Eligibility improved from 2/3 to 3/3 and seed-level standard deviation fell
  from approximately 0.0191 to 0.0040. The lower learning rate therefore
  materially improved safety and repeatability.
- Tracking quality did not pass the Case gate. Selected temperature-step costs
  were approximately 0.00997, 0.01725, and 0.01574; even the best safe points
  seen anywhere in the curves were 0.00590, 0.00827, and 0.00440, still far
  above the bad anchor cost 0.000232.
- Phase 6 remains blocked. A final bounded learning-rate midpoint pilot at 2e-4
  is prepared with all other experimental factors unchanged. This is the last
  learning-rate retry: if it does not retain 3/3 eligibility while improving
  temperature-step materially, further learning-rate search stops and the
  training objective/checkpoint-selection design must be reviewed instead.

## Local Run Gate C: learning-rate 2e-4 midpoint review

- All three seeds completed 200,000 transitions and the static review/bundle
  passed. Selected steps were 200,000, 100,000, and 140,000.
- Eligibility regressed from 3/3 at 1e-4 to 2/3. Seed 0 had no eligible trained
  checkpoint and selected an ineligible final policy; the seed-level standard
  deviation increased from approximately 0.0040 to 0.0134.
- Selected temperature-step costs were approximately 0.03769, 0.00380, and
  0.01010. The best safe temperature-step points in the curves were absent for
  seed 0, 0.00256 for seed 1, and 0.00807 for seed 2, still far above the bad
  anchor cost of approximately 0.000232.
- The preset retry gate failed: 2e-4 did not retain 3/3 eligibility and did not
  materially close the common temperature-step gap. Learning-rate search is
  therefore closed; neither a larger Cascade budget nor reference-budget
  freeze is justified by these pilots.
- Review bundle:
  `runs/pilots/cascade/sac-lr2e4/v2/cascade-sac-pilot-lr2e4-v2-review.zip`.

## Gate C root-cause audit and decision

- The three controlled learning-rate trials rule out a simple learning-rate
  defect. Lowering the rate improved safety and repeatability, but no trial
  produced temperature-step tracking near even the bad fixed anchor.
- The Track declares equal-weight `commissioning` and `temperature-step`
  training Cases, but the generalist training path resolves only
  `training.distribution_id`. The second declared Case and both Case weights do
  not participate in runtime episode sampling.
- A deterministic audit of 100 episodes from
  `cascade-regulation-training-l2-v1` found that every episode is 1,600 steps
  with exactly one post-reset feasible-equilibrium reference event. In
  contrast, the fixed temperature-step Case is 2,640 steps with four staged
  reference events. This is a protocol/training-population mismatch, not
  evidence that more transitions alone will solve the Case.
- `regulation-v1` also supplies no dense slew, effort, soft/hard-safety, or
  protection-intervention cost. That sparse safety signal explains why an
  aggressive learning rate can become unsafe, but changing the reward is not
  the first repair because it would introduce a second experimental factor and
  does not address the missing staged-reference population.
- Gate C conclusion is revised to **C -- implementation/protocol error**. The
  existing v1 distribution will remain immutable. A new versioned,
  case-conditioned Cascade distribution must represent commissioning and
  staged temperature-step modes at equal weight; changing the Track training
  distribution reopens Track-hash and anchor consistency checks before any
  replacement pilot.

## Gate C protocol repair and reopened Gate A

- Added `cascade-regulation-training-l2-v2` without modifying the registered v1
  distribution. The v1 distribution hash remains locked at
  `1032928566d244c5bddac8e94254b8edb711e2a8993709814db3f2d25a8d2c11`.
- The v2 distribution is 2,640 steps and samples two explicit modes at equal
  weight: commissioning has two post-reset reference stages and
  temperature-step has four. Event times, feasible targets, plant parameters,
  initial states, and unmeasured disturbances remain procedurally sampled; the
  validation/test fixed Case identities are not copied into training.
- EpisodeSpec difficulty tags now expose the selected training mode. A
  200-episode deterministic audit covered both modes, produced 200 unique
  EpisodeSpec hashes, and passed every Cascade feasibility check.
- Case-conditioned distributions are now required to match the normalized
  training Case weights declared by their Track. This turns the previously
  documentary Case weights into an enforced protocol invariant for the new
  distribution while preserving legacy distributions.
- The Cascade v2 Track now binds the v2 distribution. Its Track hash changed to
  `54c02b0c3ef9cce7a5a006b3f9df4b46d9256880cdebb87841468ead72e0a67e` and
  its distribution hash is
  `ae1437ce8a919c79e2e0a005e351ca70f00af84a3e3bdceeef07070795923137`.
- As required, the Track-hash change initially made the builtin anchor fail
  closed. Cascade-only calibration was rerun in the visible local terminal and
  the new candidate passed static review: six expected Cases, no missing/extra
  hashes, all bad/reference eligibility vectors true, non-degenerate gaps,
  valid provenance, and split isolation.
- The recalibrated Case utilities are byte-for-byte equal to the prior
  candidate; only Track hash, NumPy provenance, and artifact hash changed. The
  reviewed candidate was promoted byte-for-byte. New anchor artifact hash:
  `ac7a378cb4b41ce325ea0d8aafc25fc5603020a72fb14ba82c30f8301ad1abdb`.
- Added a bounded replacement pilot at the safety-stable 1e-4 learning rate:
  `configs/experiments/cascade/sac-pilot-case-conditioned-v2.json`. All other
  algorithm, budget, seed, validation, and checkpoint factors match the prior
  1e-4 retry; output identity is separate.
- PASS: 59 anchor, Track, distribution, split, and training-integration tests;
  PASS: the replacement config dry-run and protocol preflight. Preflight
  observed both training modes, 2,640-step episodes, reference-event counts 3
  and 5, zero optimizer updates, and zero test rollouts.
- PASS: final focused no-training regression before launch: 93 tests, Ruff,
  compileall, candidate/builtin byte equality, and diff whitespace checks.
- LAUNCHED: the historical case-conditioned launcher was started in the
  persistent visible local terminal. Its runtime and protocol-preflight
  artifacts were written before optimizer execution began; completion is
  reviewed below.

## Gate C case-conditioned replacement pilot review

- COMPLETE: all three seeds ran 200,000 transitions and the visible terminal
  returned successfully. The static review passed, the review ZIP passed its
  integrity check, and no test-split execution was detected.
- Selected steps were 120,000, 120,000, and 20,000. All three selected policies
  were ranking-eligible with zero state, command, action, protection,
  intervention, and hard-termination counts on both validation Cases.
- Relative to the prior 1e-4 pilot, aggregate mean cost improved by 23.0%, IQM
  by 34.6%, and seed-level standard deviation fell by 50.3% to approximately
  0.00197. The mean selected temperature-step cost improved by 45.7%.
- Selected temperature-step costs were approximately 0.00935, 0.00903, and
  0.00493. They remain about 40.3x, 38.9x, and 21.3x the bad anchor cost of
  0.000232, so all three per-Case official scores remain zero.
- The best safe temperature-step points in the recorded curves were 0.00651,
  0.00289, and 0.00493, all at 20,000 transitions. Even these remain about
  28.1x, 12.5x, and 21.3x the bad anchor. Later training therefore does not
  show evidence that a 500k--1M budget extension would close the gap.
- Checkpoint selection still favors commissioning once the temperature-step
  fixed-anchor score clips to zero: seeds 0 and 1 selected 120k policies whose
  temperature-step costs are materially worse than their safe 20k points.
  This does not create the underlying underfit, but it can hide the least-bad
  bottleneck checkpoint and must be corrected before reference training.
- Gate C remains closed to Phase 6. The training-population repair was useful
  and materially improved stability, but did not meet the Case quality gate.
  The next bounded work item is a validation-only review of unclipped
  anchor-relative checkpoint selection together with the dense/sparse
  training-objective decomposition; no additional training or test evaluation
  is authorized yet.
- Review bundle:
  `runs/pilots/cascade/sac-case-conditioned/v2/cascade-sac-pilot-case-conditioned-v2-review.zip`.

## Post-Gate C checkpoint-selector repair and reward review

- Official ranking semantics remain unchanged: fixed-anchor Case scores are
  still clipped at zero and official Track aggregation still uses the declared
  weighted geometric mean. A new `case_anchor_margins` field records the same
  fixed-anchor normalization before clipping for validation-time diagnostics
  and checkpoint selection.
- Eligible checkpoint selection now first maximizes the number of Cases above
  their bad anchors. While any Case remains below its bad anchor, it then
  maximizes the worst unclipped anchor margin and the mean margin before using
  the clipped official score. Once every Case clears its bad anchor, official
  score resumes its role as the primary selector key.
- The selector-state schema is bumped from `aiogym.validation_state.v1` to
  `aiogym.validation_state.v2`. Old selector sidecars fail closed instead of
  being silently resumed under changed selection semantics; future runs must
  start a fresh selection history.
- Applying the repaired selector statically to the completed real learning
  curves reselects the safe 20k checkpoint for all three seeds. Their Case cost
  pairs are approximately `(0.01773, 0.00651)`, `(0.01842, 0.00289)`, and
  `(0.02168, 0.00493)` for `(commissioning, temperature-step)`. This fixes the
  commissioning-only bias, but the temperature-step bottleneck still remains
  12.5x--28.1x worse than the bad anchor.
- `regulation-v1` reward audit: the scalar tracking term and official
  `regulation_cost` both use the physical-time integral of mean squared output
  errors normalized by each controlled-output range. A live Cascade transition
  regression confirms that the returned reward equals the negative official
  error integral when no terminal failure occurs. There is no training/evaluation
  objective-scale mismatch.
- Slew, effort, service-shortfall, soft/hard-safety, and protection cost weights
  are explicitly zero in canonical `regulation-v1`; safety remains a separate
  eligibility gate with a remaining-horizon terminal failure cost. Because the
  repaired pilot was 3/3 safe with zero interventions, changing RewardSpec now
  would add an unsupported second factor and force another Track/reward/anchor
  contract revision. No reward change is made.
- PASS: 47 focused ranking/selector tests, 38 reward/scorecard contract tests,
  and a combined 150-test checkpoint/resume/runner/ranking/reward regression.
  Ruff and compileall pass; diff checking reports only the pre-existing CRLF
  warning for `aiogym/controllers/tuning/tune_cstr_mpc.py`.
- Decision: the selector defect is fixed and the reward contract is internally
  aligned, but neither change makes the observed policy quality sufficient for
  Phase 6. No new training is started. Any further pilot must target one new,
  predeclared algorithm/training-stability factor rather than reward semantics
  or a larger budget.

## Cascade early-training stability diagnostic

- Added non-invasive SAC diagnostics to every validation curve point. Recorded
  fields include optimizer updates, SB3 actor/critic loss, entropy coefficient
  and loss, learning rate, replay size/capacity, and deterministic critic-Q
  mean/std/absolute scale over fixed replay-buffer indices.
- Q diagnostics do not call replay sampling and do not advance NumPy or Torch
  RNG state. Missing logger/replay/critic data is represented as unavailable
  rather than changing training behavior.
- Added `configs/experiments/cascade/sac-early-diagnostics-v2.json`: 60,000
  transitions, seeds 0/1/2, validation every 2,000 transitions, resumable
  checkpoint cadence 10,000, and otherwise the exact case-conditioned 1e-4 SAC
  algorithm/replay/Track protocol.
- The historical early-diagnostic launcher used isolated output under
  `runs/diagnostics/cascade/sac-early/v2` and the same guarded preflight,
  static review, plotting, and text-only bundle workflow.
- PASS: 60 diagnostic/config/selector/checkpoint tests, script syntax, Ruff,
  dry-run, and protocol preflight. Preflight records zero optimizer updates and
  zero test rollouts.
- COMPLETE: all three 60,000-transition seeds finished in the persistent
  visible local terminal. Static review and review-ZIP integrity checks pass,
  and no test split was accessed. Each seed produced 31 validation points at
  2,000-transition spacing.
- The repaired selector chose 10k, 18k, and 12k for seeds 0, 1, and 2. All
  selected policies are ranking-eligible. Their `(commissioning,
  temperature-step)` costs are approximately `(0.01983, 0.00374)`, `(0.01927,
  0.00263)`, and `(0.02109, 0.00311)`; aggregate mean cost is approximately
  0.01161, IQM 0.01149, and seed standard deviation 0.00060.
- Dense early validation improves temperature-step selection by approximately
  42.5%, 9.2%, and 36.9% relative to each seed's previously recorded safe 20k
  point. Nevertheless, the selected temperature-step costs remain 16.1x,
  11.3x, and 13.4x the bad-anchor cost, so the Case quality gate still fails.
- All three seeds show the same broad failure chronology: best
  temperature-step behavior occurs at 10k--18k, the first greater-than-50%
  degradation follows at 20k--28k, and the final 60k policies are materially
  worse. The learned entropy coefficient falls from approximately
  0.20--0.45 at the selected points to 0.014--0.017 at 60k. Critic absolute-Q
  scale peaks near 16k at approximately 112--114 and then falls to 41--47.
- Critic-loss spikes do not provide a consistent explanation: the largest
  spikes occur much later for seeds 0 and 1, and deterioration begins before
  them. Entropy-coefficient and Q-scale collapse are therefore the strongest
  shared diagnostic association, but are not yet a proven cause because both
  also covary with training time.
- Decision: do not extend the training budget and do not change RewardSpec.
  Gate C remains closed to Phase 6. The next admissible experiment is one
  bounded, single-factor 60k diagnostic that changes only SAC target entropy
  from the seven-dimensional default of `-7` to `-3.5`, while retaining
  automatic entropy-coefficient adaptation and the same 2k validation cadence.
  Proceed only if it preserves 3/3 eligibility, avoids early coefficient/Q
  collapse, and materially improves both the selected temperature-step floor
  and post-optimum stability. If stability improves without closing a
  substantial part of the anchor gap, stop SAC hyperparameter tuning and move
  the investigation to policy representation or training-task structure.
- Review bundle:
  `runs/diagnostics/cascade/sac-early/v2/cascade-sac-early-diagnostics-v2-review.zip`.
