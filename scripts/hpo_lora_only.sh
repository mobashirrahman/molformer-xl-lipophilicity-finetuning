#!/usr/bin/env bash
# Re-run only the LoRA study, with the widened learning-rate range.
set -uo pipefail
PROJECT="/home/$(id -un)/NNTI_Project"
PY="/scratch/$(id -un)/nnti-venv/bin/python"
"$PY" "${PROJECT}/scripts/hpo.py" \
  --storage "${PROJECT}/experiments/hpo/journal.log" \
  --study stageD-lora --method lora \
  --trials "${TRIALS:-5}" --epochs "${EPOCHS:-12}" --splits 0,1
