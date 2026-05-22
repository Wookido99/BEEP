#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASETS = ["MUTAG", "BA3", "FC", "MNIST"]
DEFAULT_EXPLAINERS = [
    "pgexplainer",
    "proxyexplainer",
    "mixupexplainer",
    "gsat",
    "confexplainer",
    "goat",
    "eigsearch",
]
METRICS = ["AUC", "F1", "Bimod", "Binar", "Fidelity"]
SUMMARY_ROW = re.compile(
    r"^(Base|Round_1|Round_2)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def parse_seed_log(path: Path):
    rows = []
    in_final_summary = False
    for line in path.read_text(errors="replace").splitlines():
        if line.strip() == "===== Final Metrics Summary =====":
            in_final_summary = True
            rows = []
            continue
        if not in_final_summary:
            continue

        match = SUMMARY_ROW.match(line.strip())
        if not match:
            continue

        stage = match.group(1)
        values = [float(match.group(i)) for i in range(2, 7)]
        for metric, value in zip(METRICS, values):
            rows.append((stage, metric, value))

    return rows


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser(description="Summarize BEEP training logs by mean/std across seeds.")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--out", default="results/beep_training_logs_mean_std.csv")
    parser.add_argument("--per-seed-out", default="results/beep_training_logs_per_seed.csv")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--explainers", nargs="+", default=DEFAULT_EXPLAINERS)
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    grouped = defaultdict(list)
    per_seed_rows = []
    missing = []

    for dataset in args.datasets:
        for explainer in args.explainers:
            log_dir = logs_root / dataset / "beep" / explainer
            seed_logs = sorted(log_dir.glob("seed_*.log"))
            if not seed_logs:
                missing.append((dataset, explainer, "no seed logs"))
                continue

            parsed_any = False
            for log_path in seed_logs:
                seed = log_path.stem.replace("seed_", "")
                rows = parse_seed_log(log_path)
                if not rows:
                    missing.append((dataset, explainer, f"{seed}: no final summary"))
                    continue
                parsed_any = True
                for stage, metric, value in rows:
                    key = (dataset, explainer, stage, metric)
                    grouped[key].append(value)
                    per_seed_rows.append(
                        {
                            "dataset": dataset,
                            "explainer": explainer,
                            "seed": seed,
                            "stage": stage,
                            "metric": metric,
                            "value": f"{value:.6f}",
                            "log_path": str(log_path),
                        }
                    )

            if not parsed_any:
                missing.append((dataset, explainer, "no parseable logs"))

    out_path = Path(args.out)
    per_seed_path = Path(args.per_seed_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)

    with per_seed_path.open("w", newline="") as f:
        fieldnames = ["dataset", "explainer", "seed", "stage", "metric", "value", "log_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "explainer", "stage", "metric", "mean", "std", "n"])
        for key in sorted(grouped):
            values = grouped[key]
            writer.writerow([*key, f"{mean(values):.6f}", f"{sample_std(values):.6f}", len(values)])

    print(f"Wrote per-seed metrics to {per_seed_path}")
    print(f"Wrote mean/std metrics to {out_path}")
    if missing:
        print("\nMissing or incomplete combinations:")
        for dataset, explainer, reason in missing:
            print(f"  {dataset} / {explainer}: {reason}")


if __name__ == "__main__":
    main()
