from pathlib import Path
import pandas as pd
import numpy as np

IF_FILE = Path("results/isolation_forest_test_predictions.csv")
AE_FILE = Path("results/autoencoder_test_predictions.csv")

OUTPUT_FILE = Path("results/model_comparison.csv")
SUMMARY_FILE = Path("results/model_comparison_summary.txt")

print("Loading model predictions...")

if_df = pd.read_csv(IF_FILE, parse_dates=["Timestamp"])
ae_df = pd.read_csv(AE_FILE, parse_dates=["Timestamp"])

# Confirm both models evaluated the same timestamps
df = if_df.merge(
    ae_df[["Timestamp", "Reconstruction_Error", "Is_Anomaly"]],
    on="Timestamp",
    how="inner",
    suffixes=("_IF", "_AE"),
    validate="one_to_one"
)

if len(df) != len(if_df) or len(df) != len(ae_df):
    raise ValueError("Prediction files do not contain matching timestamps.")

df = df.rename(columns={
    "Is_Anomaly_IF": "IF_Anomaly",
    "Is_Anomaly_AE": "AE_Anomaly"
})

df["IF_Anomaly"] = df["IF_Anomaly"].astype(int)
df["AE_Anomaly"] = df["AE_Anomaly"].astype(int)

# Compare model agreement
if_flag = df["IF_Anomaly"].eq(1)
ae_flag = df["AE_Anomaly"].eq(1)

both = if_flag & ae_flag
if_only = if_flag & ~ae_flag
ae_only = ae_flag & ~if_flag
neither = ~if_flag & ~ae_flag

union_count = int((if_flag | ae_flag).sum())
intersection_count = int(both.sum())

jaccard = (
    intersection_count / union_count
    if union_count else 0.0
)

# Experimental ensemble strategies, not validated classifiers
df["Ensemble_OR"] = (if_flag | ae_flag).astype(int)
df["Ensemble_AND"] = (if_flag & ae_flag).astype(int)

df.to_csv(OUTPUT_FILE, index=False)

summary = [
    "MODEL AGREEMENT SUMMARY",
    f"Matched test observations: {len(df)}",
    f"Isolation Forest flagged: {int(if_flag.sum())}",
    f"Autoencoder flagged: {int(ae_flag.sum())}",
    f"Flagged by both: {int(both.sum())}",
    f"Isolation Forest only: {int(if_only.sum())}",
    f"Autoencoder only: {int(ae_only.sum())}",
    f"Flagged by neither: {int(neither.sum())}",
    f"OR ensemble flagged: {union_count}",
    f"AND ensemble flagged: {intersection_count}",
    f"Jaccard agreement: {jaccard:.4f}",
    "",
    "OR means either model flags the reading.",
    "AND means both models flag the reading.",
    "Neither strategy is confirmed better without labeled evaluation data.",
]

SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

print("\n" + "\n".join(summary))
print("\nSaved detailed comparison:", OUTPUT_FILE)
print("Saved summary:", SUMMARY_FILE)
