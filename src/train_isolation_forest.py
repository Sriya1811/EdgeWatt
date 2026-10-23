from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_FILE = Path("data/processed/edgewatt_features.csv")
MODEL_FILE = Path("models/isolation_forest.joblib")
RESULT_FILE = Path("results/isolation_forest_test_predictions.csv")
SUMMARY_FILE = Path("results/isolation_forest_summary.json")

TRAIN_FRACTION = 0.80
TRAIN_SAMPLE_SIZE = 100_000
N_ESTIMATORS = 100
CONTAMINATION = 0.01
RANDOM_STATE = 42

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

# --------------------------------------------------
# Validate files and prepare output directories
# --------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading selected features from dataset...")

columns = ["Timestamp"] + FEATURES
df = pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    parse_dates=["Timestamp"]
)

if df[FEATURES].isna().any().any():
    raise ValueError("Feature dataset contains missing values.")

if not df["Timestamp"].is_monotonic_increasing:
    raise ValueError("Timestamps are not sorted chronologically.")

n_rows = len(df)
split_index = int(n_rows * TRAIN_FRACTION)

if split_index <= TRAIN_SAMPLE_SIZE:
    raise ValueError("Not enough training rows for the configured sample.")

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("\nDATA SPLIT")
print("Total observations:", n_rows)
print("Training-period rows:", len(train_df))
print("Test-period rows:", len(test_df))
print("Training period:", train_df["Timestamp"].iloc[0],
      "to", train_df["Timestamp"].iloc[-1])
print("Test period:", test_df["Timestamp"].iloc[0],
      "to", test_df["Timestamp"].iloc[-1])

# Random sampling is restricted to the training period.
sample_df = train_df.sample(
    n=TRAIN_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

X_train = sample_df[FEATURES].to_numpy(dtype=np.float32)
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

# --------------------------------------------------
# Train the baseline
# --------------------------------------------------
model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    max_samples=256,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining Isolation Forest...")
start_train = time.perf_counter()
model.fit(X_train)
training_seconds = time.perf_counter() - start_train

# --------------------------------------------------
# Score the later test period
# --------------------------------------------------
print("Scoring test observations...")
start_score = time.perf_counter()

# Lower decision_function values indicate more anomalous points.
decision_scores = model.decision_function(X_test)
predictions = model.predict(X_test)

scoring_seconds = time.perf_counter() - start_score
anomaly_mask = predictions == -1
anomaly_count = int(anomaly_mask.sum())
anomaly_rate = float(anomaly_mask.mean())

# Save compact predictions for further analysis.
output = pd.DataFrame({
    "Timestamp": test_df["Timestamp"].to_numpy(),
    "Global_active_power": test_df["Global_active_power"].to_numpy(),
    "Anomaly_Score": decision_scores,
    "Is_Anomaly": anomaly_mask.astype(int)
})

output.to_csv(RESULT_FILE, index=False)

# --------------------------------------------------
# Save model and experiment summary
# --------------------------------------------------
joblib.dump(model, MODEL_FILE)

model_size_mb = MODEL_FILE.stat().st_size / (1024 ** 2)
process = psutil.Process()
memory_mb = process.memory_info().rss / (1024 ** 2)

summary = {
    "model": "IsolationForest",
    "random_state": RANDOM_STATE,
    "training_fraction": TRAIN_FRACTION,
    "training_sample_size": TRAIN_SAMPLE_SIZE,
    "test_rows": len(test_df),
    "n_estimators": N_ESTIMATORS,
    "max_samples": 256,
    "contamination_assumption": CONTAMINATION,
    "training_seconds": round(training_seconds, 3),
    "scoring_seconds": round(scoring_seconds, 3),
    "scoring_ms_per_observation": round(
        scoring_seconds / len(test_df) * 1000, 6
    ),
    "test_anomaly_count": anomaly_count,
    "test_anomaly_rate_percent": round(anomaly_rate * 100, 4),
    "model_size_mb": round(model_size_mb, 3),
    "process_rss_mb_after_scoring": round(memory_mb, 2),
    "training_start": str(train_df["Timestamp"].iloc[0]),
    "training_end": str(train_df["Timestamp"].iloc[-1]),
    "test_start": str(test_df["Timestamp"].iloc[0]),
    "test_end": str(test_df["Timestamp"].iloc[-1]),
    "features": FEATURES,
    "evaluation_note": (
        "Unsupervised candidate anomaly flags; no ground-truth "
        "labels are available, so precision, recall, F1, and "
        "real-world detection accuracy are not established."
    )
}

with SUMMARY_FILE.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nISOLATION FOREST BASELINE SUMMARY")
print("Training time (seconds):", round(training_seconds, 3))
print("Scoring time (seconds):", round(scoring_seconds, 3))
print("Scoring latency (ms/observation):", round(
    scoring_seconds / len(test_df) * 1000, 6
))
print("Candidate anomalies:", anomaly_count)
print("Candidate anomaly rate (%):", round(anomaly_rate * 100, 4))
print("Model size (MB):", round(model_size_mb, 3))
print("Process memory after scoring (MB):", round(memory_mb, 2))
print("Saved model:", MODEL_FILE)
print("Saved predictions:", RESULT_FILE)
print("Saved summary:", SUMMARY_FILE)

print("\nMost anomalous 10 test observations:")
print(
    output.nsmallest(10, "Anomaly_Score").to_string(index=False)
)
