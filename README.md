# EdgeWatt

A Lightweight Machine Learning Framework for Real-Time Anomalous Energy Usage Detection in Smart Homes.

## Features
- Smart-home energy consumption analysis
- Isolation Forest anomaly detection
- Autoencoder-based anomaly detection
- Interactive Streamlit dashboard
- Configurable date ranges and chart styles
- Anomaly analysis and CSV export

## Technology Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, Streamlit, Plotly.

## Setup
1. Install Python 3.12.
2. Install dependencies:
   pip install -r requirements.txt
3. Place the processed dataset at data/processed/edgewatt_features.csv.
4. Place the trained models in the models directory.
5. Start the dashboard:
   python -m streamlit run src/app.py

## Models
- Isolation Forest
- Autoencoder

Anomaly flags are candidate events and are not confirmed faults. Synthetic benchmark results should not be interpreted as verified real-world accuracy.

## Dataset
The project uses the UCI Individual Household Electric Power Consumption dataset.
