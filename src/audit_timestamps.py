import pandas as pd
from pathlib import Path

path = Path("data/processed/edgewatt_features.csv")

print("Loading feature dataset for timestamp audit...")
df = pd.read_csv(path, usecols=["Timestamp"], parse_dates=["Timestamp"])

seconds = df["Timestamp"].diff().dt.total_seconds()
gaps = seconds[seconds != 60].dropna()

print("\nTIMESTAMP AUDIT")
print("Total rows:", len(df))
print("Exactly one-minute intervals:", int((seconds == 60).sum()))
print("Intervals longer or shorter than one minute:", len(gaps))

if len(gaps):
    print("\nFirst 10 irregular intervals:")
    print(pd.DataFrame({
        "Previous_Timestamp": df["Timestamp"].shift(1)[gaps.index],
        "Current_Timestamp": df["Timestamp"][gaps.index],
        "Elapsed_Seconds": gaps
    }).head(10).to_string(index=False))
else:
    print("All consecutive saved timestamps are one minute apart.")
