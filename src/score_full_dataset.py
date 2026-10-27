
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "edgewatt_features.parquet"
IF_MODEL = ROOT / "models" / "isolation_forest.joblib"
AE_MODEL = ROOT / "models" / "edgewatt_autoencoder.keras"
AE_SCALER = ROOT / "models" / "autoencoder_scaler.joblib"

OUTPUT_FILE = ROOT / "results" / "full_dataset_predictions.parquet"
TEMP_FILE = ROOT / "results" / "full_dataset_predictions.tmp.parquet"

BATCH_SIZE = 5000
AE_THRESHOLD = 1.335868

FEATURE_COLUMNS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2",
    "Sub_metering_3", "HourSin", "HourCos", "IsWeekend",
    "Power_Lag_1", "Power_Lag_5", "Rolling_Mean_5",
    "Rolling_Std_5", "Rolling_Mean_30", "Power_Change",
    "Power_Relative_Change",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    for path in (IF_MODEL, AE_MODEL, AE_SCALER):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load the locally stored Parquet dataset.
    print("Loading dataset...")
    df = pd.read_parquet(DATA_FILE)

    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column is missing.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Dataset is empty.")

    print(f"Input readings: {total_rows:,}")
    print("Loading trained models...")

    isolation_forest = joblib.load(IF_MODEL)
    autoencoder = tf.keras.models.load_model(AE_MODEL, compile=False)
    scaler = joblib.load(AE_SCALER)

    # Write to a temporary file first, so a failed run does not
    # replace an existing completed output.
    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    writer = None
    processed = 0
    if_count = 0
    ae_count = 0
    started = time.perf_counter()

    try:
        for start in range(0, total_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_rows)
            batch = df.iloc[start:end]

            x = batch[FEATURE_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            )
            x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Isolation Forest
            if_flags = isolation_forest.predict(x) == -1

            # Autoencoder
            scaled = scaler.transform(x)
            reconstructed = autoencoder.predict(
                scaled, batch_size=BATCH_SIZE, verbose=0
            )
            errors = np.mean(
                np.square(scaled - reconstructed), axis=1
            )
            ae_flags = errors > AE_THRESHOLD

            out = pd.DataFrame({
                "Timestamp": batch["Timestamp"].to_numpy(),
                "Global_active_power": pd.to_numeric(
                    batch["Global_active_power"], errors="coerce"
                ).to_numpy(),
                "IF_Anomaly": if_flags,
                "AE_Error": errors,
                "AE_Anomaly": ae_flags,
            })

            table = pa.Table.from_pandas(out, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    TEMP_FILE, table.schema, compression="snappy"
                )

            writer.write_table(table)
            processed += len(out)
            if_count += int(if_flags.sum())
            ae_count += int(ae_flags.sum())

            print(
                f"Processed {processed:,}/{total_rows:,} "
                f"| IF flags: {if_count:,} "
                f"| AE flags: {ae_count:,}"
            )

            del batch, x, scaled, reconstructed, errors, out, table

    finally:
        if writer is not None:
            writer.close()

    # Validate the finished file before publishing it as the final output.
    if processed != total_rows or not TEMP_FILE.exists():
        raise RuntimeError(
            f"Incomplete scoring: processed {processed:,} "
            f"of {total_rows:,} readings."
        )

    saved_rows = pq.ParquetFile(TEMP_FILE).metadata.num_rows
    if saved_rows != total_rows:
        raise RuntimeError(
            f"Output verification failed: found {saved_rows:,} "
            f"rows, expected {total_rows:,}."
        )

    TEMP_FILE.replace(OUTPUT_FILE)

    elapsed = time.perf_counter() - started

    print("\nFull-dataset scoring completed successfully.")
    print(f"Input readings: {total_rows:,}")
    print(f"Saved predictions: {saved_rows:,}")
    print(f"Isolation Forest flagged: {if_count:,}")
    print(f"Autoencoder flagged: {ae_count:,}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
