from pathlib import Path
import pandas as pd

INPUT_FILE = Path("results/isolation_forest_test_predictions.csv")
OUTPUT_FILE = Path("results/isolation_forest_anomaly_events.csv")

df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Identify consecutive flagged readings.
# A timestamp gap always starts a new event.
is_anomaly = df["Is_Anomaly"].eq(1)
previous_is_anomaly = is_anomaly.shift(1, fill_value=False)
one_minute_gap = df["Timestamp"].diff().dt.total_seconds().eq(60)

starts_event = is_anomaly & ~(previous_is_anomaly & one_minute_gap)
event_ids = starts_event.cumsum().where(is_anomaly)

anomalies = df.loc[is_anomaly].copy()
anomalies["Event_ID"] = event_ids.loc[is_anomaly].astype(int)

if anomalies.empty:
    print("No candidate anomalies found.")
    raise SystemExit(0)

events = (
    anomalies.groupby("Event_ID")
    .agg(
        Start=("Timestamp", "min"),
        End=("Timestamp", "max"),
        Anomaly_Readings=("Timestamp", "size"),
        Mean_Active_Power=("Global_active_power", "mean"),
        Peak_Active_Power=("Global_active_power", "max"),
        Lowest_Anomaly_Score=("Anomaly_Score", "min"),
    )
    .reset_index()
)

events["Duration_Minutes"] = (
    (events["End"] - events["Start"]).dt.total_seconds() / 60 + 1
).astype(int)

events = events.sort_values(
    ["Anomaly_Readings", "Lowest_Anomaly_Score"],
    ascending=[False, True]
)

events.to_csv(OUTPUT_FILE, index=False)

print("\nANOMALY EVENT ANALYSIS")
print("Test observations:", len(df))
print("Flagged observations:", int(is_anomaly.sum()))
print("Distinct anomaly events:", len(events))
print("Single-reading events:", int((events["Anomaly_Readings"] == 1).sum()))
print("Multi-reading events:", int((events["Anomaly_Readings"] > 1).sum()))
print("Longest event (readings):", int(events["Anomaly_Readings"].max()))
print("Saved event report:", OUTPUT_FILE)

print("\nTOP 15 ANOMALY EVENTS")
print(events.head(15).round(4).to_string(index=False))

print("\nPOWER DISTRIBUTION")
print("Median test active power:", round(df["Global_active_power"].median(), 3))
print("Median flagged active power:", round(anomalies["Global_active_power"].median(), 3))
print("Maximum flagged active power:", round(anomalies["Global_active_power"].max(), 3))
