from pathlib import Path
import pandas as pd

INPUT_FILE = Path("results/model_comparison.csv")
OUTPUT_FILE = Path("results/ensemble_event_summary.csv")

df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

def summarize_events(data, flag_col, strategy):
    flagged = data[flag_col].eq(1)
    previous_flagged = flagged.shift(1, fill_value=False)
    continuous = data["Timestamp"].diff().dt.total_seconds().eq(60)

    event_start = flagged & ~(previous_flagged & continuous)
    event_id = event_start.cumsum().where(flagged)

    selected = data.loc[flagged].copy()
    selected["Event_ID"] = event_id.loc[flagged].astype(int)

    if selected.empty:
        return []

    events = selected.groupby("Event_ID").agg(
        Start=("Timestamp", "min"),
        End=("Timestamp", "max"),
        Readings=("Timestamp", "size"),
        Mean_Active_Power=("Global_active_power", "mean"),
        Peak_Active_Power=("Global_active_power", "max"),
    ).reset_index()

    events["Duration_Minutes"] = (
        (events["End"] - events["Start"]).dt.total_seconds() / 60 + 1
    ).astype(int)

    events["Strategy"] = strategy
    return events.to_dict("records")

all_events = []

for col, name in [
    ("IF_Anomaly", "Isolation Forest"),
    ("AE_Anomaly", "Autoencoder"),
    ("Ensemble_OR", "OR ensemble"),
    ("Ensemble_AND", "AND ensemble"),
]:
    events = summarize_events(df, col, name)
    all_events.extend(events)

    print(f"\n{name}")
    print("Flagged readings:", int(df[col].sum()))
    print("Distinct events:", len(events))
    if events:
        event_df = pd.DataFrame(events)
        print("Single-reading events:", int((event_df["Readings"] == 1).sum()))
        print("Multi-reading events:", int((event_df["Readings"] > 1).sum()))
        print("Longest event (readings):", int(event_df["Readings"].max()))

result = pd.DataFrame(all_events)
result.to_csv(OUTPUT_FILE, index=False)

print("\nSaved event summary:", OUTPUT_FILE)
