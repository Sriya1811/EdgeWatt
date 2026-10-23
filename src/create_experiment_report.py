
from pathlib import Path
import json
import pandas as pd

RESULTS = Path("results")
INPUT_FILE = RESULTS / "multi_window_benchmark.csv"
SUMMARY_FILE = RESULTS / "multi_window_benchmark_summary.json"
REPORT_FILE = RESULTS / "edgewatt_experiment_report.txt"
CSV_REPORT_FILE = RESULTS / "edgewatt_model_metrics.csv"

print("Loading multi-window benchmark results...")

df = pd.read_csv(INPUT_FILE)

with open(SUMMARY_FILE, encoding="utf-8") as f:
    summary = json.load(f)

lines = [
    "EDGEWATT EXPERIMENT REPORT",
    "=" * 60,
    "",
    "Project: EdgeWatt - Lightweight ML for Anomalous Energy Usage",
    "Experiment: Multi-window synthetic anomaly benchmark",
    "",
    "EVALUATION SETUP",
    "-" * 60,
    f"Contiguous windows: {summary['window_count']}",
    f"Rows scored: {len(df)}",
    f"Window size: {summary['window_size']}",
    f"Autoencoder threshold: {summary['autoencoder_threshold']}",
    "",
    "IMPORTANT LIMITATION",
    "-" * 60,
    "These metrics evaluate injected synthetic anomalies only.",
    "They are not verified real-world fault-detection metrics.",
    "",
    "OVERALL MODEL METRICS",
    "-" * 60,
]

metric_rows = []

for name, metrics in summary["models"].items():
    lines.append(f"\n{name}")

    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")

    metric_rows.append({
        "Model": name,
        **metrics,
    })

lines.extend([
    "",
    "SCENARIO-LEVEL DETECTION",
    "-" * 60,
])

for scenario, metrics in summary["by_scenario"].items():
    lines.append(f"\n{scenario}")

    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")

lines.extend([
    "",
    "INTERPRETATION",
    "-" * 60,
    "The Autoencoder achieved the highest F1-score among the",
    "evaluated configurations in this synthetic experiment.",
    "The OR ensemble did not improve recall over the Autoencoder",
    "and produced more false positives.",
    "",
    "Further evaluation using diverse scenarios and verified",
    "fault labels is required before claiming real-world accuracy.",
    "",
])

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
pd.DataFrame(metric_rows).to_csv(CSV_REPORT_FILE, index=False)

print("\nReport created:", REPORT_FILE)
print("Metrics CSV created:", CSV_REPORT_FILE)
print("\nMODEL SUMMARY")

for name, metrics in summary["models"].items():
    print(
        f"{name}: "
        f"precision={metrics['precision']}, "
        f"recall={metrics['recall']}, "
        f"F1={metrics['F1']}, "
        f"FPR={metrics['false_positive_rate']}"
    )

print("\nReport generation completed.")
