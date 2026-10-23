from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path("data/processed/household_power_consumption_clean.csv")
OUTPUT_FILE = Path("data/processed/edgewatt_features.csv")

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

power_col = "Global_active_power"
power = df[power_col]

# Identify continuous one-minute segments
elapsed = df["Timestamp"].diff().dt.total_seconds()
new_segment = elapsed.ne(60)
df["_Segment"] = new_segment.cumsum()

# Time-of-day context
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.day_of_week
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# Lag features restart at each segment boundary
groups = df.groupby("_Segment", sort=False)[power_col]

df["Power_Lag_1"] = groups.shift(1)
df["Power_Lag_5"] = groups.shift(5)

# Rolling statistics use previous observations only
previous_power = df.groupby("_Segment", sort=False)[power_col].shift(1)

df["Rolling_Mean_5"] = (
    previous_power.groupby(df["_Segment"], sort=False)
    .transform(lambda s: s.rolling(5, min_periods=5).mean())
)

df["Rolling_Std_5"] = (
    previous_power.groupby(df["_Segment"], sort=False)
    .transform(lambda s: s.rolling(5, min_periods=5).std())
)

df["Rolling_Mean_30"] = (
    previous_power.groupby(df["_Segment"], sort=False)
    .transform(lambda s: s.rolling(30, min_periods=30).mean())
)

# Changes calculated only within continuous segments
df["Power_Change"] = groups.diff()
df["Power_Relative_Change"] = (
    df["Power_Change"] / previous_power.abs().clip(lower=0.1)
)

# Timing diagnostics
df["Elapsed_Seconds"] = elapsed
df["Is_One_Minute_Interval"] = elapsed.eq(60).astype(int)

# Remove rows without the required temporal history
feature_cols = [
    "Power_Lag_1",
    "Power_Lag_5",
    "Rolling_Mean_5",
    "Rolling_Std_5",
    "Rolling_Mean_30",
    "Power_Change",
    "Power_Relative_Change",
]

df[feature_cols] = df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
df = df.dropna(subset=feature_cols).copy()

# Segment ID is an internal processing field, not a model feature
df = df.drop(columns=["_Segment"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nCORRECTED FEATURE ENGINEERING SUMMARY")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing cells:", int(df.isna().sum().sum()))
print("Output file:", OUTPUT_FILE)
print("Output size (MB):", round(
    OUTPUT_FILE.stat().st_size / (1024 ** 2), 2
))
print("Non-one-minute intervals:", int(
    (df["Is_One_Minute_Interval"] == 0).sum()
))
print("\nFeature columns:")
print(df.columns.tolist())
