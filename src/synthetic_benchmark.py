
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

DATA_FILE = Path("data/processed/edgewatt_features.csv")
IF_MODEL_FILE = Path("models/isolation_forest.joblib")
AE_MODEL_FILE = Path("models/edgewatt_autoencoder.keras")
SCALER_FILE = Path("models/autoencoder_scaler.joblib")

OUTPUT_FILE = Path("results/synthetic_anomaly_benchmark.csv")
SUMMARY_FILE = Path("results/synthetic_anomaly_benchmark_summary.json")

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "HourSin",
    "HourCos",
    "IsWeekend",
    "Power_Lag_1",
    "Power_Lag_5",
    "Rolling_Mean_5",
    "Rolling_Std_5",
    "Rolling_Mean_30",
    "Power_Change",
    "Power_Relative_Change",
]

SEED = 42
rng = np.random.default_rng(SEED)

print("Loading data and saved models...")

df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

split_index = int(len(df) * 0.80)
test_df = df.iloc[split_index:].copy().reset_index(drop=True)

if_model = joblib.load(IF_MODEL_FILE)
ae_model = keras.models.load_model(AE_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Find a contiguous test window.
window_size = 3000
start = None

for candidate in range(0, len(test_df) - window_size, 1000):
    window = test_df.iloc[candidate:candidate + window_size]
    gaps = window["Timestamp"].diff().dt.total_seconds().dropna()

    if (gaps == 60).all():
        start = candidate
        break

if start is None:
    raise RuntimeError("Could not find a continuous test window.")

original = (
    test_df.iloc[start:start + window_size]
    .copy()
    .reset_index(drop=True)
)

power_col = "Global_active_power"

# Choose four non-overlapping scenarios.
cases = [
    {
        "name": "Point spike",
        "start": 500,
        "length": 1,
        "type": "spike",
        "factor": 3.0,
    },
    {
        "name": "Sustained spike",
        "start": 1000,
        "length": 10,
        "type": "spike",
        "factor": 2.5,
    },
    {
        "name": "Stuck-on pattern",
        "start": 1500,
        "length": 20,
        "type": "stuck",
        "factor": 2.0,
    },
    {
        "name": "Contextual high usage",
        "start": 2200,
        "length": 10,
        "type": "contextual",
        "factor": 2.5,
    },
]

modified = original.copy()
labels = np.zeros(len(modified), dtype=int)
scenario_names = np.full(len(modified), "", dtype=object)

# Estimate typical power for each clock hour using the selected window.
# The contextual case uses the upper end of that hour's observed values
# as a reference and raises power beyond it.
hour_values = original["Timestamp"].dt.hour

for case in cases:
    s = case["start"]
    e = s + case["length"]
    baseline = original.loc[s:e - 1, power_col].to_numpy(dtype=float)

    if case["type"] == "spike":
        modified.loc[s:e - 1, power_col] = baseline * case["factor"]

    elif case["type"] == "stuck":
        level = max(float(np.median(baseline)) * case["factor"], 3.0)
        modified.loc[s:e - 1, power_col] = level

    elif case["type"] == "contextual":
        hours = hour_values.iloc[s:e].to_numpy()
        values = []

        for hour in hours:
            hour_power = original.loc[
                hour_values == hour, power_col
            ].to_numpy(dtype=float)

            # Use a high empirical hourly reference, then exceed it.
            reference = float(np.quantile(hour_power, 0.95))
            values.append(max(reference * case["factor"], reference + 1.0))

        modified.loc[s:e - 1, power_col] = values

    labels[s:e] = 1
    scenario_names[s:e] = case["name"]

# Keep electrical measurements internally more consistent by updating
# active-power-related measurements proportionally where possible.
# This is still a simplified simulation, not a physical appliance model.
original_power = original[power_col].to_numpy(dtype=float)
new_power = modified[power_col].to_numpy(dtype=float)

ratio = np.divide(
    new_power,
    original_power,
    out=np.ones_like(new_power),
    where=original_power > 0.1,
)

modified["Global_intensity"] = (
    original["Global_intensity"].to_numpy(dtype=float) * ratio
)

# Do not scale voltage or reactive power blindly: these measurements
# are not guaranteed to change linearly with active power.
# Sub-metering remains unchanged, so physical consistency is imperfect.

# Recompute all power-derived features for this continuous window.
power = modified[power_col]
previous = power.shift(1)

modified["Power_Lag_1"] = previous
modified["Power_Lag_5"] = power.shift(5)
modified["Rolling_Mean_5"] = previous.rolling(5, min_periods=5).mean()
modified["Rolling_Std_5"] = previous.rolling(5, min_periods=5).std()
modified["Rolling_Mean_30"] = previous.rolling(30, min_periods=30).mean()
modified["Power_Change"] = power.diff()
modified["Power_Relative_Change"] = (
    power.diff() / previous.abs().clip(lower=0.1)
)

# Use the same trained models and saved AE threshold.
valid = modified[FEATURES].notna().all(axis=1).to_numpy()
X = modified.loc[valid, FEATURES].to_numpy(dtype=np.float32)

print("Scoring synthetic observations...")

if_predictions = if_model.predict(X)
if_scores = if_model.decision_function(X)

X_scaled = scaler.transform(X).astype(np.float32)
reconstructed = ae_model.predict(
    X_scaled, batch_size=512, verbose=0
)
ae_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

with open("results/autoencoder_summary.json", encoding="utf-8") as f:
    ae_summary = json.load(f)

ae_threshold = float(ae_summary["validation_error_threshold"])
ae_predictions = (ae_errors > ae_threshold).astype(int)

scored = modified.loc[valid].copy().reset_index(drop=True)
scored["Synthetic_Anomaly"] = labels[valid]
scored["Scenario"] = scenario_names[valid]
scored["IF_Score"] = if_scores
scored["IF_Anomaly"] = (if_predictions == -1).astype(int)
scored["AE_Error"] = ae_errors
scored["AE_Anomaly"] = ae_predictions

scored["Ensemble_OR"] = (
    (scored["IF_Anomaly"] == 1)
    | (scored["AE_Anomaly"] == 1)
).astype(int)

scored["Ensemble_AND"] = (
    (scored["IF_Anomaly"] == 1)
    & (scored["AE_Anomaly"] == 1)
).astype(int)

scored.to_csv(OUTPUT_FILE, index=False)

summary = {
    "benchmark": "Controlled synthetic anomaly experiment",
    "random_seed": SEED,
    "test_window_start": str(original["Timestamp"].iloc[0]),
    "test_window_end": str(original["Timestamp"].iloc[-1]),
    "window_rows": len(original),
    "synthetic_positive_rows": int(labels.sum()),
    "scoring_rows": int(valid.sum()),
    "scenarios": cases,
    "limitations": [
        "Synthetic labels are not real fault labels.",
        "The contextual scenario uses an empirical hourly reference.",
        "Active power and intensity are adjusted proportionally.",
        "Reactive power and sub-metering are not physically simulated.",
        "The benchmark uses one contiguous test window.",
    ],
    "models": {},
}

for model_name, pred_col in [
    ("Isolation Forest", "IF_Anomaly"),
    ("Autoencoder", "AE_Anomaly"),
    ("OR ensemble", "Ensemble_OR"),
    ("AND ensemble", "Ensemble_AND"),
]:
    y_true = scored["Synthetic_Anomaly"].to_numpy()
    y_pred = scored[pred_col].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    summary["models"][model_name] = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "synthetic_precision": round(precision, 4),
        "synthetic_recall": round(recall, 4),
        "synthetic_F1": round(f1, 4),
        "synthetic_detection_rate_percent": round(recall * 100, 2),
    }

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSYNTHETIC ANOMALY BENCHMARK")
print("Window start:", summary["test_window_start"])
print("Window end:", summary["test_window_end"])
print("Window rows:", len(original))
print("Synthetic anomalous readings:", int(labels.sum()))
print("Scored readings:", int(valid.sum()))

for name, metrics in summary["models"].items():
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

print("\nSaved predictions:", OUTPUT_FILE)
print("Saved summary:", SUMMARY_FILE)
print("\nCAUTION: Results apply only to the synthetic scenarios.")
