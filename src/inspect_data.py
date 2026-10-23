from pathlib import Path
import pandas as pd

file_path = Path("data/raw/extracted/household_power_consumption.txt")

print("File exists:", file_path.exists())
print("File size (MB):", round(file_path.stat().st_size / (1024 ** 2), 2))

df = pd.read_csv(
    file_path,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST 5 ROWS")
print(df.head().to_string())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDATE RANGE")
print("Start:", df["Date"].iloc[0])
print("End:", df["Date"].iloc[-1])

print("\nTOTAL MISSING CELLS")
print(int(df.isna().sum().sum()))
