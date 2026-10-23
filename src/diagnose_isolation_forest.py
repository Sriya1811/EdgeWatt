from pathlib import Path
import joblib
import pandas as pd
import numpy as np

DATA_FILE = Path("data/processed/edgewatt_features.csv")
MODEL_FILE = Path("models/isolation_forest.joblib")

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

print("Loading data and saved model...")
df = pd.read_csv(DATA_FILE, usecols=["Timestamp"] + FEATURES,
                 parse_dates=["Timestamp"])
model = joblib.load(MODEL_FILE)

split_index = int(len(df) * 0.80)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

train_sample = train_df.sample(n=100_000, random_state=42)

X_train = train_sample[FEATURES].to_numpy(dtype=np.float32)
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

print("Scoring training sample...")
train_scores = model.decision_function(X_train)
train_predictions = model.predict(X_train)

print("Scoring test period...")
test_scores = model.decision_function(X_test)
test_predictions = model.predict(X_test)

print("\nTHRESHOLD DIAGNOSTICS")
print("Model decision threshold (offset):", round(float(model.offset_), 6))
print("Training sample flagged (%):",
      round(float((train_predictions == -1).mean() * 100), 4))
print("Test period flagged (%):",
      round(float((test_predictions == -1).mean() * 100), 4))

print("\nDECISION SCORE DISTRIBUTION")
print("Training sample:")
print(pd.Series(train_scores).describe(
    percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
).round(6).to_string())

print("\nTest period:")
print(pd.Series(test_scores).describe(
    percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
).round(6).to_string())

print("\nInterpretation:")
print("Scores below the model threshold are flagged as anomalies.")
print("Differences between training and test rates can indicate")
print("distribution shift; they do not by themselves establish")
print("whether the flagged observations are true anomalies.")
