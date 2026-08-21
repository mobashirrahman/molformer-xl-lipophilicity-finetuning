#!/usr/bin/env bash
# Tune every PEFT method in sequence on this host.
#
# Stage D compares four methods against each other, so each needs a learning
# rate chosen on equal terms. All hosts run this same script and join the same
# four Optuna studies through the shared journal, so the work distributes
# without any per-host configuration -- which matters because ~/.config is on
# NFS and cannot hold per-host values.
set -uo pipefail

PROJECT="/home/$(id -un)/NNTI_Project"
PY="/scratch/$(id -un)/nnti-venv/bin/python"
TRIALS="${TRIALS:-4}"
EPOCHS="${EPOCHS:-12}"

for method in bitfit lora ia3 full; do
  echo "=== tuning ${method} (${TRIALS} trials on $(hostname -s)) ==="
  "$PY" "${PROJECT}/scripts/hpo.py" \
    --storage "${PROJECT}/experiments/hpo/journal.log" \
    --study "stageD-${method}" \
    --method "${method}" \
    --trials "${TRIALS}" \
    --epochs "${EPOCHS}" \
    --splits 0,1
done
echo "=== all PEFT studies done on $(hostname -s) ==="
