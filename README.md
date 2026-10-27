# EdgeWatt

**A Lightweight Machine Learning Framework for Real-Time Anomalous Energy Usage Detection in Smart Homes**

EdgeWatt is a machine-learning project that identifies unusual household electricity consumption patterns using Isolation Forest and a Dense Autoencoder. It includes a Streamlit dashboard for exploring energy usage, reviewing candidate anomalies, visualizing results, and exporting data.

## Live Demo

- **Streamlit Dashboard:** https://edgewatt-uime8fbvhktbpyoumzx9zv.streamlit.app/
- **GitHub Repository:** https://github.com/Sriya1811/EdgeWatt

The hosted dashboard uses a lightweight demonstration dataset. The full processed dataset is maintained locally and is required to reproduce the full-dataset experiments.

## Features

- Household electricity consumption analysis
- Isolation Forest anomaly detection
- Dense Autoencoder reconstruction-error detection
- Candidate anomaly comparison and ensemble experiments
- Interactive Streamlit dashboard
- Date-range filtering and visualization
- Anomaly analysis and CSV export
- Synthetic anomaly benchmarking
- Model runtime and storage measurements

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Joblib

## Dataset

The project uses the **UCI Individual Household Electric Power Consumption** dataset.

The processing pipeline creates time-series and engineered features from household electricity measurements.

The model uses 17 features:

- Global active power
- Global reactive power
- Voltage
- Global intensity
- Sub-metering 1, 2, and 3
- Hour sine and cosine
- Weekend indicator
- Lagged power values
- Rolling mean and standard deviation
- Power change and relative power change

The full processed dataset is stored locally at:

`data/processed/edgewatt_features.csv`

The hosted demonstration uses:

`data/processed/edgewatt_features_demo.csv`

The demonstration dataset is a reduced sample and should not be treated as the full experimental dataset.

## Methodology

### 1. Data preparation

The electricity dataset is processed into chronological observations with engineered time and power-consumption features.

### 2. Chronological evaluation split

The first 80% of observations is used as the training period. The final 20% is held out as a later test period.

Both models were evaluated on the same **409,789 test observations**, covering:

`2010-02-05 04:29:00` to `2010-11-26 21:02:00`

### 3. Isolation Forest

Isolation Forest learns patterns in the training data and assigns anomaly flags to observations that appear unusual. The configured contamination assumption is 0.01.

### 4. Dense Autoencoder

The Dense Autoencoder learns to reconstruct the input features. A reconstruction-error threshold is derived from the 99th percentile of validation errors. Observations exceeding that threshold are flagged as candidate anomalies.

The scaler is fitted on the training subset and then applied to validation and test data.

### 5. Ensemble experiments

Two experimental strategies are evaluated:

- **OR:** Flag an observation if either model identifies it.
- **AND:** Flag an observation only if both models identify it.

These strategies are exploratory and have not been validated against verified real-world fault labels.

## Operational Evaluation

The following results were recorded in the current experimental environment.

| Metric | Isolation Forest | Dense Autoencoder |
|---|---:|---:|
| Test observations | 409,789 | 409,789 |
| Training sample | 100,000 | 100,000 |
| Training time | 3.158 s | 22.409 s |
| Test scoring time | 6.828 s | 1.405 s |
| Reported scoring latency | 0.016663 ms/observation | 0.003429 ms/observation |
| Model size | 0.967 MB | 0.052 MB |
| Candidate anomalies | 1,609 | 1,278 |
| Candidate anomaly rate | 0.3926% | 0.3119% |

The Autoencoder had lower reported scoring time and model size in this run. These measurements depend on the execution environment and do not establish superior detection accuracy.

## Model Agreement

| Result | Count |
|---|---:|
| Flagged by both models | 361 |
| Isolation Forest only | 1,248 |
| Autoencoder only | 917 |
| Flagged by neither | 407,263 |
| OR ensemble | 2,526 |
| AND ensemble | 361 |

The Jaccard agreement between the two models was **0.1429**, indicating limited overlap between their candidate anomaly sets.

## Synthetic Anomaly Benchmark

A separate experiment evaluated eight contiguous windows of 3,000 observations each, with injected synthetic anomaly scenarios.

| Model | Precision | Recall | F1-score | False-positive rate |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.1111 | 0.0457 | 0.0648 | 0.0051 |
| Dense Autoencoder | 0.3660 | 0.2957 | 0.3272 | 0.0072 |
| OR ensemble | 0.2636 | 0.2957 | 0.2787 | 0.0116 |

In this synthetic experiment, the Autoencoder achieved the highest F1-score among the evaluated configurations. The OR ensemble did not improve recall over the Autoencoder and produced more false positives.

**These metrics measure performance against injected synthetic anomalies only. They are not verified real-world fault-detection metrics.**

## Installation and Local Execution

### Prerequisites

- Python environment compatible with the packages in `requirements.txt`
- Git
- The processed dataset for full-dataset experiments

### Setup

Clone the repository:

```bash
git clone https://github.com/Sriya1811/EdgeWatt.git
cd EdgeWatt
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
python -m streamlit run src/app.py
```

The hosted app uses the demonstration dataset. To rerun the full-dataset training and evaluation scripts, place the full processed CSV at `data/processed/edgewatt_features.csv`.

### Training and Evaluation Scripts

The repository includes scripts for:

- Feature engineering and data inspection
- Isolation Forest training
- Autoencoder training
- Model comparison
- Anomaly analysis
- Synthetic anomaly benchmarking
- Experiment report generation

Run the appropriate scripts from the project root after ensuring that the required datasets and dependencies are available.

## Project Structure

```text
EdgeWatt/
├── data/
│   └── processed/
├── models/
├── results/
├── src/
│   ├── app.py
│   ├── train_isolation_forest.py
│   ├── train_autoencoder.py
│   ├── compare_models.py
│   └── create_experiment_report.py
├── requirements.txt
└── README.md
```

Additional analysis and reporting scripts are available in `src/`.

## Limitations

- Ground-truth anomaly labels are unavailable for the real held-out test period.
- Candidate anomaly flags should not be interpreted as confirmed faults.
- Synthetic benchmark metrics apply only to the injected scenarios evaluated.
- The models use different thresholding methods.
- The OR and AND ensembles have not been validated against verified fault labels.
- Runtime and memory measurements are environment-dependent.
- The hosted demonstration dataset is smaller than the full processed dataset.

## Future Work

- Evaluate against verified fault labels when available.
- Expand synthetic testing to more diverse anomaly patterns.
- Investigate threshold calibration and false-alarm reduction.
- Assess detection performance across different household usage conditions.
- Evaluate resource consumption and latency on target edge hardware.

## Conclusion

EdgeWatt demonstrates a lightweight anomaly-detection workflow using Isolation Forest and a Dense Autoencoder, supported by an interactive dashboard and experimental benchmarking. The current results characterize candidate anomaly behavior, model agreement, and operational costs. Further evaluation using verified fault labels is needed before making claims about real-world detection accuracy.

## License

Add a license file if you intend to distribute the project under a specific open-source license.
