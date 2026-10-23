from pathlib import Path
import pandas as pd

# File paths
INPUT_FILE = Path(
    "data/raw/extracted/household_power_consumption.txt"
)
OUTPUT_FILE = Path(
    "data/processed/household_power_consumption_clean.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Original shape:", df.shape)

# Convert measurement columns to numeric
measurement_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

for col in measurement_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Combine date and time into a timestamp
df["Timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Remove invalid timestamps and sort chronologically
df = df.dropna(subset=["Timestamp"])
df = df.sort_values("Timestamp")
df = df.drop_duplicates(subset=["Timestamp"], keep="first")
df = df.set_index("Timestamp")

print("\nMissing measurements before cleaning:")
print(df[measurement_cols].isna().sum())

# Interpolate only short gaps in each measurement series.
# The time limit prevents interpolation through long missing periods.
df[measurement_cols] = df[measurement_cols].interpolate(
    method="time",
    limit=5,
    limit_area="inside"
)

# Keep rows with all required measurements available
before_drop = len(df)
df = df.dropna(subset=measurement_cols)
removed_rows = before_drop - len(df)

# Save cleaned data with a timestamp column
df = df.reset_index()
df = df[["Timestamp"] + measurement_cols]

df.to_csv(OUTPUT_FILE, index=False)

print("\nCLEANING SUMMARY")
print("Rows after timestamp validation and deduplication:", before_drop)
print("Rows removed due to remaining missing measurements:", removed_rows)
print("Final shape:", df.shape)
print("Remaining missing cells:", int(df.isna().sum().sum()))
print("Output file:", OUTPUT_FILE)
print("Output size (MB):", round(OUTPUT_FILE.stat().st_size / (1024 ** 2), 2))
print("\nFirst 5 cleaned rows:")
print(df.head().to_string(index=False))
