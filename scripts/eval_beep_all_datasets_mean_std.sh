#!/usr/bin/env bash
set -uo pipefail

GPU="${1:-0}"
DATASETS="${DATASETS:-MUTAG BA3 FC MNIST}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dataset in $DATASETS; do
  echo "===== eval dataset: ${dataset} ====="
  bash "$SCRIPT_DIR/eval_beep_dataset_explainers.sh" "$dataset" "$GPU"
  echo
done

mkdir -p "$SCRIPT_DIR/../results"
combined="$SCRIPT_DIR/../results/beep_all_datasets_all_explainers_mean_std.csv"
printf 'dataset,explainer,metric,mean,std,n\n' > "$combined"

for dataset in $DATASETS; do
  csv="$SCRIPT_DIR/../results/${dataset}/${dataset}_beep_all_explainers_mean_std.csv"
  if [[ -f "$csv" ]]; then
    awk -F, 'NR > 1 {print $0}' "$csv" >> "$combined"
  else
    echo "[warn] missing dataset mean/std CSV: ${csv}"
  fi
done

echo "Wrote all-dataset mean/std summary to ${combined}"
