#!/usr/bin/env bash
# Run every experiment for all seeds in configs/base.yaml, then build comparison tables.
# Supervised runs come first because the few-shot runs use their checkpoints.
#
#   bash scripts/run_all.sh                          # everything
#   bash scripts/run_all.sh --set train.epochs=5     # extra overrides go to every run
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=("$@")
for cfg in configs/experiments/supervised/*.yaml; do
  plantdisease run "$cfg" --resume "${EXTRA[@]}"
done
for cfg in configs/experiments/fewshot/[!_]*.yaml; do
  plantdisease run "$cfg" --resume "${EXTRA[@]}"
done

mkdir -p outputs/comparison
plantdisease summarize outputs/exp* --output outputs/comparison/supervised.md
plantdisease summarize outputs/fewshot_* \
  --metrics test/accuracy test/accuracy_ci95 target/accuracy target/accuracy_ci95 target/macro_f1 \
  --output outputs/comparison/fewshot.md
