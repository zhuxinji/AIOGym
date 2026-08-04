#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${AIOGYM_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG="${REPO_ROOT}/configs/experiments/cascade/sac-pilot-lr2e4-v2.json"
RUN_DIR="${REPO_ROOT}/runs/pilots/cascade/sac-lr2e4/v2"
SUMMARY="${RUN_DIR}/cascade-sac-pilot-lr2e4-v2.multi-seed.json"
RUNTIME="${RUN_DIR}/runtime-environment.json"
PREFLIGHT="${RUN_DIR}/preflight.json"
REVIEW_JSON="${RUN_DIR}/training-review.json"
REVIEW_MD="${RUN_DIR}/training-review.md"
CURVES="${RUN_DIR}/learning-curves.svg"
BUNDLE="${RUN_DIR}/cascade-sac-pilot-lr2e4-v2-review.zip"
LOG="${RUN_DIR}/pilot.log"
mkdir -p "${RUN_DIR}"

TRAIN_ARGS=(
  -m aiogym.cli.main train
  --config "${CONFIG}"
  --seeds 0,1,2
)
if [[ "${AIOGYM_PILOT_OVERWRITE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
  rm -f "${RUNTIME}" "${PREFLIGHT}" "${REVIEW_JSON}" "${REVIEW_MD}" "${CURVES}" "${BUNDLE}"
fi

run_pilot() {
if [[ -f "${RUNTIME}" ]]; then
  printf 'Reusing runtime report: %s\n' "${RUNTIME}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/capture_runtime_environment.py" \
    --output "${RUNTIME}"
fi
if [[ -f "${PREFLIGHT}" ]]; then
  printf 'Reusing protocol preflight: %s\n' "${PREFLIGHT}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_protocol.py" \
    --config "${CONFIG}" \
    --output "${PREFLIGHT}"
fi
if [[ -f "${SUMMARY}" ]]; then
  printf 'Reusing completed multi-seed summary: %s\n' "${SUMMARY}"
else
  "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/review_training_run.py" \
  "${SUMMARY}" \
  --output-json "${REVIEW_JSON}" \
  --output-md "${REVIEW_MD}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_learning_curves.py" \
  "${SUMMARY}" \
  --output "${CURVES}" \
  --overwrite
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_review_bundle.py" \
  "${SUMMARY}" \
  --output "${BUNDLE}" \
  --runtime-report "${RUNTIME}" \
  --review-json "${REVIEW_JSON}" \
  --review-markdown "${REVIEW_MD}" \
  --log "${LOG}" \
  --overwrite

printf 'Review bundle: %s\n' "${BUNDLE}"
}

run_pilot 2>&1 | tee -a "${LOG}"
