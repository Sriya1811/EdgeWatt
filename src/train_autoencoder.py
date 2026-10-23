from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_FILE = Path("data/processed/edgewatt_features.csv")
MODEL_FILE = Path("models/edgewatt_autoencoder.keras")
SCALER_FILE = Path("models/autoencoder_scaler.joblib")
PREDICTIONS_FILE = Path("results/autoencoder_test_predictions.csv")
SUMMARY_FILE = Path("results/autoencoder_summary.json")

TRAIN_FRACTION = 0.80
TRAIN_SAMPLE_SIZE = 100_000
EPOCHS = 30
BATCH_SIZE = 512
THRESHOLD_PERCENTILE = 99

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

for folder in ["models", "results"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("Loading feature dataset...")
df = pd.read_csv(
    INPUT_FILE,
    usecols=["Timestamp"] + FEATURES,
    parse_dates=["Timestamp"]
)

if df[FEATURES].isna().any().any():
    raise ValueError("Missing values detected in model features.")

if not df["Timestamp"].is_monotonic_increasing:
    raise ValueError("Timestamps must be sorted.")

split_index = int(len(df) * TRAIN_FRACTION)
train_period = df.iloc[:split_index]
test_period = df.iloc[split_index:]

# Sample only from the earlier training period
sample = train_period.sample(
    n=min(TRAIN_SAMPLE_SIZE, len(train_period)),
    random_state=SEED
)

# Separate training and validation data
train_part, val_part = train_test_split(
    sample,
    test_size=0.15,
    random_state=SEED,
    shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(
    train_part[FEATURES].to_numpy(dtype=np.float32)
).astype(np.float32)

X_val = scaler.transform(
    val_part[FEATURES].to_numpy(dtype=np.float32)
).astype(np.float32)

X_test = scaler.transform(
    test_period[FEATURES].to_numpy(dtype=np.float32)
).astype(np.float32)

print("\nDATA SPLIT")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_val))
print("Test observations:", len(X_test))
print("Features:", len(FEATURES))

# Build a small dense autoencoder
inputs = keras.Input(shape=(len(FEATURES),), name="input_features")

x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
encoded = layers.Dense(4, activation="relu", name="bottleneck")(x)

x = layers.Dense(8, activation="relu")(encoded)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(len(FEATURES), activation="linear")(x)

autoencoder = keras.Model(inputs, outputs, name="edgewatt_autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

print("\nTraining autoencoder...")
start_train = time.perf_counter()

history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stop],
    verbose=2
)

training_seconds = time.perf_counter() - start_train

# Establish the threshold from validation reconstruction errors
print("\nCalculating validation reconstruction errors...")
val_reconstructed = autoencoder.predict(
    X_val, batch_size=BATCH_SIZE, verbose=0
)
val_errors = np.mean(np.square(X_val - val_reconstructed), axis=1)

threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))

# Evaluate the later test period
print("Scoring test observations...")
start_score = time.perf_counter()

test_reconstructed = autoencoder.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
)
test_errors = np.mean(
    np.square(X_test - test_reconstructed), axis=1
)

scoring_seconds = time.perf_counter() - start_score
anomaly_mask = test_errors > threshold
anomaly_count = int(anomaly_mask.sum())
anomaly_rate = float(anomaly_mask.mean())

predictions = pd.DataFrame({
    "Timestamp": test_period["Timestamp"].to_numpy(),
    "Global_active_power": test_period["Global_active_power"].to_numpy(),
    "Reconstruction_Error": test_errors,
    "Is_Anomaly": anomaly_mask.astype(int)
})
predictions.to_csv(PREDICTIONS_FILE, index=False)

autoencoder.save(MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

model_size_mb = MODEL_FILE.stat().st_size / (1024 ** 2)
scaler_size_mb = SCALER_FILE.stat().st_size / (1024 ** 2)
rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)

summary = {
    "model": "Dense Autoencoder",
    "random_seed": SEED,
    "training_period_rows": len(train_period),
    "training_sample_size": len(sample),
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(test_period),
    "features": FEATURES,
    "epochs_completed": len(history.history["loss"]),
    "best_validation_loss": float(min(history.history["val_loss"])),
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "validation_error_threshold": threshold,
    "training_seconds": round(training_seconds, 3),
    "scoring_seconds": round(scoring_seconds, 3),
    "batch_scoring_ms_per_observation": round(
        scoring_seconds / len(test_period) * 1000, 6
    ),
    "test_candidate_anomalies": anomaly_count,
    "test_candidate_anomaly_rate_percent": round(
        anomaly_rate * 100, 4
    ),
    "model_size_mb": round(model_size_mb, 3),
    "scaler_size_mb": round(scaler_size_mb, 3),
    "process_rss_after_scoring_mb": round(rss_mb, 2),
    "training_start": str(train_period["Timestamp"].iloc[0]),
    "training_end": str(train_period["Timestamp"].iloc[-1]),
    "test_start": str(test_period["Timestamp"].iloc[0]),
    "test_end": str(test_period["Timestamp"].iloc[-1]),
    "evaluation_note": (
        "Unsupervised candidate anomaly flags using a validation-derived "
        "reconstruction-error threshold. Ground-truth labels are not "
        "available; real-world precision, recall, F1, and accuracy "
        "are not established."
    )
}

with SUMMARY_FILE.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nAUTOENCODER BASELINE SUMMARY")
print("Epochs completed:", len(history.history["loss"]))
print("Training time (seconds):", round(training_seconds, 3))
print("Scoring time (seconds):", round(scoring_seconds, 3))
print("Batch scoring latency (ms/observation):", round(
    scoring_seconds / len(test_period) * 1000, 6
))
print("Validation threshold:", round(threshold, 6))
print("Candidate anomalies:", anomaly_count)
print("Candidate anomaly rate (%):", round(anomaly_rate * 100, 4))
print("Model size (MB):", round(model_size_mb, 3))
print("Scaler size (MB):", round(scaler_size_mb, 3))
print("Process memory after scoring (MB):", round(rss_mb, 2))
print("Saved model:", MODEL_FILE)
print("Saved scaler:", SCALER_FILE)
print("Saved predictions:", PREDICTIONS_FILE)
print("Saved summary:", SUMMARY_FILE)
