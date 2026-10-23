
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
AE_SUMMARY_FILE = Path("results/autoencoder_summary.json")

OUTPUT_FILE = Path("results/multi_window_benchmark.csv")
SUMMARY_FILE = Path("results/multi_window_benchmark_summary.json")

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

WINDOW_SIZE = 3000
WINDOW_STEP = 5000
SEED = 42

print("Loading test data and saved models...")

df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

split = int(len(df) * 0.80)
test = df.iloc[split:].copy().reset_index(drop=True)

if_model = joblib.load(IF_MODEL_FILE)
ae_model = keras.models.load_model(AE_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

with open(AE_SUMMARY_FILE, encoding="utf-8") as f:
    ae_summary = json.load(f)

ae_threshold = float(ae_summary["validation_error_threshold"])

# Find separate, contiguous windows from the held-out test period.
windows = []
candidate = 0

while candidate + WINDOW_SIZE <= len(test) and len(windows) < 8:
    block = test.iloc[candidate:candidate + WINDOW_SIZE]
    intervals = block["Timestamp"].diff().dt.total_seconds().dropna()

    if len(intervals) == WINDOW_SIZE - 1 and (intervals == 60).all():
        windows.append(block.copy().reset_index(drop=True))
        candidate += WINDOW_SIZE + WINDOW_STEP
    else:
        candidate += 1000

if len(windows) < 3:
    raise RuntimeError(
        f"Only {len(windows)} valid windows found; need at least 3."
    )

print("Contiguous windows selected:", len(windows))

all_results = []

for window_id, original in enumerate(windows, start=1):
    print(f"Processing window {window_id}/{len(windows)}...")

    modified = original.copy()
    labels = np.zeros(len(modified), dtype=int)
    scenarios = np.full(len(modified), "", dtype=object)

    # Non-overlapping synthetic cases within each window.
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

    original_power = original["Global_active_power"].to_numpy(dtype=float)
    hour = original["Timestamp"].dt.hour

    for case in cases:
        s = case["start"]
        e = s + case["length"]
        baseline = original_power[s:e]

        if case["type"] == "spike":
            injected = baseline * case["factor"]

        elif case["type"] == "stuck":
            level = max(float(np.median(baseline)) * case["factor"], 3.0)
            injected = np.full(case["length"], level)

        else:
            injected = []

            for h in hour.iloc[s:e]:
                hourly = original.loc[
                    hour == h, "Global_active_power"
                ].to_numpy(dtype=float)

                reference = float(np.quantile(hourly, 0.95))
                injected.append(
                    max(reference * case["factor"], reference + 1.0)
                )

            injected = np.asarray(injected, dtype=float)

        modified.loc[s:e - 1, "Global_active_power"] = injected
        labels[s:e] = 1
        scenarios[s:e] = case["name"]

    # Update intensity proportionally as a simplified approximation.
    # Other electrical measurements are not physically simulated.
    modified["Global_intensity"] = (
        original["Global_intensity"].to_numpy(dtype=float)
        * np.divide(
            modified["Global_active_power"].to_numpy(dtype=float),
            original_power,
            out=np.ones_like(original_power),
            where=original_power > 0.1,
        )
    )

    # Recompute active-power-derived features.
    power = modified["Global_active_power"]
    previous = power.shift(1)

    modified["Power_Lag_1"] = previous
    modified["Power_Lag_5"] = power.shift(5)
    modified["Rolling_Mean_5"] = previous.rolling(
        5, min_periods=5
    ).mean()
    modified["Rolling_Std_5"] = previous.rolling(
        5, min_periods=5
    ).std()
    modified["Rolling_Mean_30"] = previous.rolling(
        30, min_periods=30
    ).mean()
    modified["Power_Change"] = power.diff()
    modified["Power_Relative_Change"] = (
        power.diff() / previous.abs().clip(lower=0.1)
    )

    valid = modified[FEATURES].notna().all(axis=1).to_numpy()
    X = modified.loc[valid, FEATURES].to_numpy(dtype=np.float32)

    if_pred = (if_model.predict(X) == -1).astype(int)

    X_scaled = scaler.transform(X).astype(np.float32)
    reconstructed = ae_model.predict(
        X_scaled, batch_size=512, verbose=0
    )
    errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
    ae_pred = (errors > ae_threshold).astype(int)

    result = modified.loc[valid].copy().reset_index(drop=True)
    result["Window_ID"] = window_id
    result["Synthetic_Anomaly"] = labels[valid]
    result["Scenario"] = scenarios[valid]
    result["IF_Anomaly"] = if_pred
    result["AE_Error"] = errors
    result["AE_Anomaly"] = ae_pred
    result["Ensemble_OR"] = (
        (result["IF_Anomaly"] == 1) | (result["AE_Anomaly"] == 1)
    ).astype(int)

    all_results.append(result)

results = pd.concat(all_results, ignore_index=True)
results.to_csv(OUTPUT_FILE, index=False)

summary = {
    "window_count": len(windows),
    "window_size": WINDOW_SIZE,
    "synthetic_anomaly_rows_per_window": 41,
    "autoencoder_threshold": ae_threshold,
    "limitations": [
        "Synthetic anomalies are not real fault labels.",
        "Electrical measurements are only partially adjusted.",
        "All windows use the same four injection positions.",
        "The contextual case uses an empirical hourly reference.",
    ],
    "models": {},
    "by_scenario": {},
}

for name, col in [
    ("Isolation Forest", "IF_Anomaly"),
    ("Autoencoder", "AE_Anomaly"),
    ("OR ensemble", "Ensemble_OR"),
]:
    y = results["Synthetic_Anomaly"].to_numpy()
    p = results[col].to_numpy()

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    summary["models"][name] = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
    }

for scenario in results.loc[
    results["Synthetic_Anomaly"] == 1, "Scenario"
].dropna().unique():
    subset = results[results["Scenario"] == scenario]
    summary["by_scenario"][scenario] = {
        "injected_rows": len(subset),
        "windows": int(subset["Window_ID"].nunique()),
        "IF_detection_rate": round(
            subset["IF_Anomaly"].mean(), 4
        ),
        "AE_detection_rate": round(
            subset["AE_Anomaly"].mean(), 4
        ),
        "OR_detection_rate": round(
            subset["Ensemble_OR"].mean(), 4
        ),
    }

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nMULTI-WINDOW SYNTHETIC BENCHMARK")
print("Windows:", len(windows))
print("Rows scored:", len(results))

for name, metrics in summary["models"].items():
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

print("\nSCENARIO DETECTION RATES")
for scenario, metrics in summary["by_scenario"].items():
    print(f"\n{scenario}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

print("\nSaved:", OUTPUT_FILE)
print("Saved:", SUMMARY_FILE)
print("\nCaution: These are synthetic benchmark results, not real-world accuracy.")
