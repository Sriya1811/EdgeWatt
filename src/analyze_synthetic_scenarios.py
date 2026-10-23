from pathlib import Path
import pandas as pd

INPUT_FILE = Path("results/synthetic_anomaly_benchmark.csv")

df = pd.read_csv(INPUT_FILE)

models = {
    "Isolation Forest": "IF_Anomaly",
    "Autoencoder": "AE_Anomaly",
    "OR ensemble": "Ensemble_OR",
    "AND ensemble": "Ensemble_AND",
}

print("SCENARIO-LEVEL SYNTHETIC RESULTS")
print("Note: these are controlled synthetic cases, not real-world labels.")

for scenario in df.loc[
    df["Synthetic_Anomaly"] == 1, "Scenario"
].dropna().unique():

    subset = df[df["Scenario"] == scenario]
    print(f"\nScenario: {scenario}")
    print("Injected readings:", len(subset))

    for name, col in models.items():
        detected = int(subset[col].sum())
        total = len(subset)
        recall = detected / total if total else 0

        print(
            f"{name}: detected {detected}/{total} "
            f"({recall * 100:.2f}%)"
        )

normal = df[df["Synthetic_Anomaly"] == 0]

print("\nFALSE-POSITIVE COUNTS ON UNMODIFIED READINGS")
print("Unmodified readings:", len(normal))

for name, col in models.items():
    flagged = int(normal[col].sum())
    rate = flagged / len(normal) if len(normal) else 0

    print(
        f"{name}: {flagged} flagged "
        f"({rate * 100:.2f}%)"
    )
