#!/usr/bin/env bash
set -uo pipefail

DATASET="${1:-}"
GPU="${2:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EXPLAINERS="${EXPLAINERS:-pgexplainer proxyexplainer mixupexplainer gsat confexplainer goat eigsearch}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <MUTAG|BA3|FC|MNIST> [gpu]"
  echo "Optional env: SEEDS=\"42 43 44 45 46\" EXPLAINERS=\"...\" SKIP_EXISTING=1 STOP_ON_ERROR=1"
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

echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "Seeds: $SEEDS"
echo "Base explainers: $EXPLAINERS"
echo

failures=0

for base_explainer in $EXPLAINERS; do
  echo "===== ${DATASET} / beep / ${base_explainer} ====="
  log_dir="logs/${DATASET}/beep/${base_explainer}"
  mkdir -p "$log_dir"

  for seed in $SEEDS; do
    seed_dir="param/${DATASET}/beep/${base_explainer}/${seed}"
    p1="${seed_dir}/${DATASET}_beep_phase1_explainer.pth"
    p2="${seed_dir}/${DATASET}_beep_best_model.pth"
    p3="${seed_dir}/${DATASET}_beep_best_model_second.pth"

    if [[ "$SKIP_EXISTING" == "1" && -f "$p1" && -f "$p2" && -f "$p3" ]]; then
      echo "[skip] ${DATASET} / ${base_explainer} / seed ${seed}: checkpoints already exist"
      continue
    fi

    log_file="${log_dir}/seed_${seed}.log"
    echo "[run] ${DATASET} / ${base_explainer} / seed ${seed}"
    python main.py \
      --dataset "$DATASET" \
      --explainer_name beep \
      --base_explainer "$base_explainer" \
      --seed "$seed" \
      --gpu "$GPU" \
      2>&1 | tee "$log_file"

    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
      echo "[fail] ${DATASET} / ${base_explainer} / seed ${seed}; see ${log_file}"
      failures=$((failures + 1))
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        exit "$status"
      fi
    fi
  done

  echo
done

if [[ "$failures" -ne 0 ]]; then
  echo "Completed with ${failures} failed run(s)."
  exit 1
fi

echo "Completed all requested runs."
