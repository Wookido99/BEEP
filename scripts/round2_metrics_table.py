#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = [
    ("AUC", "round2_auc"),
    ("Binar", "round2_bin"),
    ("Bimod", "round2_bim"),
    ("F1", "round2_f1"),
]


def fmt(mean, std):
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def main():
    parser = argparse.ArgumentParser(description="Build a compact Round_2 metric table.")
    parser.add_argument(
        "--input",
        default="results/beep_training_logs_mean_std.csv",
        help="Input mean/std CSV from summarize_beep_training_logs.py",
    )
    parser.add_argument(
        "--output",
        default="results/round2_auc_bin_bim_f1_summary.csv",
        help="Output wide CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {input_path}. Run scripts/summarize_beep_training_logs.py first."
        )

    values = defaultdict(dict)
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stage"] != "Round_2":
                continue
            metric = row["metric"]
            if metric not in {name for name, _ in METRICS}:
                continue
            values[(row["dataset"], row["explainer"])][metric] = fmt(row["mean"], row["std"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["dataset", "explainer"] + [header for _, header in METRICS]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for dataset, explainer in sorted(values):
            metric_values = values[(dataset, explainer)]
            writer.writerow(
                [dataset, explainer]
                + [metric_values.get(metric, "") for metric, _ in METRICS]
            )

    widths = {header: len(header) for header in headers}
    table_rows = []
    for dataset, explainer in sorted(values):
        metric_values = values[(dataset, explainer)]
        row = {
            "dataset": dataset,
            "explainer": explainer,
            **{header: metric_values.get(metric, "") for metric, header in METRICS},
        }
        table_rows.append(row)
        for header, value in row.items():
            widths[header] = max(widths[header], len(value))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in table_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
