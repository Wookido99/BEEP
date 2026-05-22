#!/usr/bin/env bash
set -uo pipefail

DATASET="${1:-}"
GPU="${2:-0}"
EXPLAINERS="${EXPLAINERS:-pgexplainer proxyexplainer mixupexplainer gsat confexplainer goat eigsearch}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <MUTAG|BA3|FC|MNIST> [gpu]"
  echo "Optional env: EXPLAINERS=\"...\" STOP_ON_ERROR=1"
  exit 2
fi

case "$DATASET" in
  MUTAG|BA3|FC|MNIST) ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

mkdir -p "logs/${DATASET}/eval" "results/${DATASET}"
summary_csv="results/${DATASET}/${DATASET}_beep_all_explainers_averages.csv"
mean_std_csv="results/${DATASET}/${DATASET}_beep_all_explainers_mean_std.csv"
printf 'dataset,explainer,metric,value\n' > "$summary_csv"
printf 'dataset,explainer,metric,mean,std,n\n' > "$mean_std_csv"

failures=0

for base_explainer in $EXPLAINERS; do
  echo "===== eval ${DATASET} / beep / ${base_explainer} ====="
  log_file="logs/${DATASET}/eval/${base_explainer}.log"

  python eval_saved_beep_models.py \
    --dataset "$DATASET" \
    --explainer_name beep \
    --base_explainer "$base_explainer" \
    --gpu "$GPU" \
    2>&1 | tee "$log_file"

  status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "[fail] eval ${DATASET} / ${base_explainer}; see ${log_file}"
    failures=$((failures + 1))
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit "$status"
    fi
    continue
  fi

  avg_csv="results/${DATASET}/${DATASET}_beep_${base_explainer}_averages.csv"
  if [[ -f "$avg_csv" ]]; then
    awk -F, -v dataset="$DATASET" -v explainer="$base_explainer" 'NR > 1 {print dataset "," explainer "," $1 "," $2}' "$avg_csv" >> "$summary_csv"
  else
    echo "[warn] missing average CSV: ${avg_csv}"
  fi

  stats_csv="results/${DATASET}/${DATASET}_beep_${base_explainer}_mean_std.csv"
  if [[ -f "$stats_csv" ]]; then
    awk -F, -v dataset="$DATASET" -v explainer="$base_explainer" 'NR > 1 {print dataset "," explainer "," $1 "," $2 "," $3 "," $4}' "$stats_csv" >> "$mean_std_csv"
  else
    echo "[warn] missing mean/std CSV: ${stats_csv}"
  fi

  echo
done

echo "Wrote combined summary to ${summary_csv}"
echo "Wrote combined mean/std summary to ${mean_std_csv}"

if [[ "$failures" -ne 0 ]]; then
  echo "Completed with ${failures} failed eval(s)."
  exit 1
fi
