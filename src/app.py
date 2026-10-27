from __future__ import annotations

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# EdgeWatt: Lightweight ML-based anomalous energy usage dashboard
# Place this file at: EdgeWatt/src/app.py
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "edgewatt_features_demo.csv"
FEATURES_PARQUET = ROOT / "data" / "processed" / "edgewatt_features_demo.parquet"

FULL_PREDICTIONS_FILE = "full_dataset_predictions.parquet"
HF_DATASET_REPO = "Hyndhavichakka/edgewatt-full-dataset"


@st.cache_data(show_spinner="Loading saved full-dataset predictions...")
def load_full_predictions() -> pd.DataFrame:
    """Download and cache precomputed predictions from Hugging Face."""
    import os
    from huggingface_hub import hf_hub_download

    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not configured.")

    prediction_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=FULL_PREDICTIONS_FILE,
        repo_type="dataset",
        token=hf_token,
    )

    predictions = pd.read_parquet(prediction_path)
    required = {
        "Timestamp",
        "Global_active_power",
        "IF_Anomaly",
        "AE_Error",
        "AE_Anomaly",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Prediction file is missing columns: " + ", ".join(sorted(missing))
        )

    predictions["Timestamp"] = pd.to_datetime(
        predictions["Timestamp"], errors="coerce"
    )
    predictions = predictions.dropna(subset=["Timestamp"])

    if predictions["Timestamp"].duplicated().any():
        raise ValueError("Prediction file contains duplicate timestamps.")

    return predictions



st.set_page_config(
    page_title="EdgeWatt | Smart Energy Intelligence",
    page_icon="âš¡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Styling --------------------------------
def apply_theme(theme: str) -> None:
    dark = theme == "Dark"
    bg = "#0b1220" if dark else "#f4f7fb"
    panel = "#111c2e" if dark else "#ffffff"
    text = "#e7eef9" if dark else "#172033"
    muted = "#98a8bf" if dark else "#627089"
    border = "#26364d" if dark else "#e0e7f0"
    accent = "#53d8c5"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
        .block-container {{ padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1500px; }}
        .hero {{ padding: 1.15rem 1.35rem; border: 1px solid {border}; border-radius: 18px;
                 background: linear-gradient(120deg, {panel}, {bg}); margin-bottom: 1rem; }}
        .hero-title {{ font-size: 1.75rem; font-weight: 750; letter-spacing: -0.04em; color: {text}; }}
        .hero-sub {{ color: {muted}; font-size: .93rem; margin-top: .3rem; }}
        .eyebrow {{ color: {accent}; font-weight: 700; text-transform: uppercase;
                    letter-spacing: .12em; font-size: .72rem; }}
        [data-testid="stMetric"] {{ background: {panel}; border: 1px solid {border};
            border-radius: 14px; padding: 15px 16px; }}
        [data-testid="stMetricLabel"] {{ color: {muted}; }}
        div[data-testid="stTabs"] button {{ font-weight: 650; }}
        div[data-testid="stDataFrame"] {{ border: 1px solid {border}; border-radius: 12px; overflow: hidden; }}
        .section-note {{ color: {muted}; font-size: .85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------- Data -----------------------------------

@st.cache_data(show_spinner="Loading energy data (first run may take a little while)...")
def load_dataset() -> pd.DataFrame:
    """Load the private full dataset when configured; otherwise use the demo CSV."""
    import os
    from huggingface_hub import hf_hub_download

    full_dataset_path = None
    dataset_source = "Demo dataset"

    # Prefer the full dataset stored in the private Hugging Face repository.
    try:
        hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

        if hf_token:
            full_dataset_path = hf_hub_download(
                repo_id="Hyndhavichakka/edgewatt-full-dataset",
                filename="edgewatt_features.csv",
                repo_type="dataset",
                token=hf_token,
            )
    except Exception as exc:
        st.warning(
            "Could not download the full dataset from Hugging Face. "
            "Using the demo dataset instead. "
            f"Details: {type(exc).__name__}"
        )

    # Use the full dataset if available; otherwise use the local demo dataset.
    if full_dataset_path:
        df = pd.read_csv(full_dataset_path, low_memory=False)
        dataset_source = "Full Hugging Face dataset"
    elif FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
    elif FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV, low_memory=False)
    else:
        raise FileNotFoundError(
            "Neither the full Hugging Face dataset nor the local demo dataset "
            "could be loaded."
        )

    if "Timestamp" not in df.columns:
        raise ValueError("The dataset does not contain a 'Timestamp' column.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return df, dataset_source


def choose_chart_data(df: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    """Downsample chart points for responsiveness without changing table/scoring data."""
    if len(df) <= max_points:
        return df
    interval = max(1, int(np.ceil(len(df) / max_points)))
    sampled = df.iloc[::interval].copy()
    # Preserve peaks and anomaly points in the visualization.
    flag_cols = [c for c in ("IF_Anomaly", "AE_Anomaly") if c in df.columns]
    if flag_cols:
        flagged = df.loc[df[flag_cols].any(axis=1)]
        sampled = pd.concat([sampled, flagged]).drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp")
    return sampled


# ---------------------------- Sidebar --------------------------------
with st.sidebar:
    st.markdown("## âš¡ EdgeWatt")
    st.caption("Smart-home energy anomaly detection")
    theme = st.selectbox("Appearance", ["Dark", "Light"], index=0)
    period = st.radio(
        "Time range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range", "All data"],
        index=0,
    )
    layout = st.radio(
        "Dashboard view",
        ["Overview", "Analytics", "Anomaly monitoring"],
        index=0,
    )
    selected_models = st.radio(
        "Models",
        ["Both models", "Autoencoder only", "Isolation Forest only"],
        index=0,
    )
    chart_style = st.radio("Energy chart", ["Line", "Area", "Bar"], index=0)
    st.divider()
    st.caption("Inference is performed only for the selected date range.")
    st.caption("Chart points are downsampled for speed; anomaly table retains scored rows.")

apply_theme(theme)

# ---------------------------- Main -----------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Energy intelligence platform</div>
      <div class="hero-title">EdgeWatt Dashboard</div>
      <div class="hero-sub">Monitor household consumption, compare anomaly detectors, and inspect unusual readings.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    full_df, dataset_source = load_dataset()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.info("Confirm that data/processed/edgewatt_features.csv exists in your EdgeWatt project.")
    st.stop()

try:
    predictions = load_full_predictions()
    prediction_columns = [
        "Timestamp", "IF_Anomaly", "AE_Error", "AE_Anomaly"
    ]

    full_df = full_df.merge(
        predictions[prediction_columns],
        on="Timestamp",
        how="left",
        validate="one_to_one",
    )

    if full_df[["IF_Anomaly", "AE_Error", "AE_Anomaly"]].isna().any().any():
        raise ValueError(
            "Some dataset readings do not have matching predictions."
        )

    predictions_ready = True
    st.success(
        f"Loaded saved predictions for all {len(full_df):,} readings."
    )
except Exception as exc:
    predictions_ready = False
    st.error(f"Could not load or match full-dataset predictions: {exc}")
    st.stop()



# Precomputed predictions are used; model files are not required at runtime.

data_min = full_df["Timestamp"].min()
data_max = full_df["Timestamp"].max()
st.caption(f"Dataset coverage: **{data_min:%d %b %Y} â€“ {data_max:%d %b %Y}** Â· {len(full_df):,} readings")
st.caption(f"Dataset source: {dataset_source}")

now_end = data_max
if period == "Last 24 hours":
    range_start, range_end = now_end - pd.Timedelta(hours=24), now_end
elif period == "Last 7 days":
    range_start, range_end = now_end - pd.Timedelta(days=7), now_end
elif period == "Last 30 days":
    range_start, range_end = now_end - pd.Timedelta(days=30), now_end
elif period == "Custom range":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=max(data_min.date(), (now_end - pd.Timedelta(days=1)).date()),
                                   min_value=data_min.date(), max_value=data_max.date())
    with col2:
        end_date = st.date_input("End date", value=now_end.date(),
                                 min_value=data_min.date(), max_value=data_max.date())
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
    range_start = pd.Timestamp(start_date)
    range_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
else:
    range_start, range_end = data_min, data_max

filtered = full_df.loc[
    full_df["Timestamp"].between(range_start, range_end, inclusive="both")
].copy()
if filtered.empty:
    st.warning("No readings were found for this period. Choose a longer time range.")
    st.stop()


# Use precomputed predictions for every reading in the selected period.
# No model inference is needed during dashboard reruns.
started = time.perf_counter()
scored = filtered.copy()

if selected_models == "Isolation Forest only":
    scored = scored.drop(
        columns=["AE_Anomaly", "AE_Error"], errors="ignore"
    )
elif selected_models == "Autoencoder only":
    scored = scored.drop(
        columns=["IF_Anomaly"], errors="ignore"
    )

# Predictions are already calculated; this measures selection/data handling.
scoring_seconds = time.perf_counter() - started

if "IF_Anomaly" in scored:
    if_anomalies = int(scored["IF_Anomaly"].sum())
else:
    if_anomalies = None
if "AE_Anomaly" in scored:
    ae_anomalies = int(scored["AE_Anomaly"].sum())
else:
    ae_anomalies = None
available_flags = [c for c in ("IF_Anomaly", "AE_Anomaly") if c in scored]
if available_flags:
    scored["Any_Anomaly"] = scored[available_flags].any(axis=1)
    total_anomalies = int(scored["Any_Anomaly"].sum())
else:
    total_anomalies = 0
mean_power = pd.to_numeric(scored["Global_active_power"], errors="coerce").mean()
estimated_kwh = pd.to_numeric(scored["Global_active_power"], errors="coerce").sum() / 60.0

if layout == "Overview":
    st.markdown("### At a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Readings analyzed", f"{len(scored):,}")
    m2.metric("Average active power", f"{mean_power:.3f} kW" if pd.notna(mean_power) else "â€”")
    m3.metric("Candidate anomaly readings", f"{total_anomalies:,}", help="Model flags are candidates, not confirmed faults.")
    m4.metric("Estimated energy", f"{estimated_kwh:,.1f} kWh", help="Approximation: sum of kW readings divided by 60, assuming minute-level samples.")

    st.markdown("### Energy consumption")
    plot_df = choose_chart_data(scored)
    fig = go.Figure()
    if chart_style == "Line":
        fig.add_trace(go.Scatter(x=plot_df["Timestamp"], y=plot_df["Global_active_power"],
                                 mode="lines", name="Active power", line=dict(width=1.6, color="#53d8c5")))
    elif chart_style == "Area":
        fig.add_trace(go.Scatter(x=plot_df["Timestamp"], y=plot_df["Global_active_power"],
                                 mode="lines", fill="tozeroy", name="Active power",
                                 line=dict(width=1.5, color="#53d8c5")))
    else:
        # Aggregate into hourly means for readable bars.
        hourly = scored.set_index("Timestamp")["Global_active_power"].resample("h").mean().dropna().reset_index()
        fig.add_trace(go.Bar(x=hourly["Timestamp"], y=hourly["Global_active_power"], name="Hourly mean power",
                             marker_color="#53d8c5"))
    if available_flags:
        flagged = plot_df.loc[plot_df["Any_Anomaly"]]
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged["Timestamp"], y=flagged["Global_active_power"],
                                     mode="markers", name="Flagged readings",
                                     marker=dict(color="#ff6b6b", size=7, symbol="circle-open")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title=None, yaxis_title="Active power (kW)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      template="plotly_dark" if theme == "Dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model status")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Isolation Forest**")
        if if_anomalies is None:
            st.info("Not selected.")
        else:
            st.metric("Flagged readings", f"{if_anomalies:,}", f"{if_anomalies / len(scored) * 100:.2f}% of analyzed")
    with c2:
        st.markdown("**Autoencoder**")
        if ae_anomalies is None:
            st.info("Not selected.")
        else:
            st.metric("Flagged readings", f"{ae_anomalies:,}", f"{ae_anomalies / len(scored) * 100:.2f}% of analyzed")
    st.caption(f"Scoring time for this view: {scoring_seconds:.2f} seconds. Flags are unsupervised model outputs, not verified faults.")

elif layout == "Analytics":
    st.markdown("### Model analytics")
    a, b, c = st.columns(3)
    a.metric("Analyzed readings", f"{len(scored):,}")
    b.metric("Scoring time", f"{scoring_seconds:.2f} s")
    c.metric("Throughput", f"{len(scored) / max(scoring_seconds, 1e-9):,.0f} rows/s")
    tabs = st.tabs(["Consumption", "Model scores", "Flag distribution"])
    with tabs[0]:
        plot_df = choose_chart_data(scored)
        fig = px.line(plot_df, x="Timestamp", y="Global_active_power", title="Active power over time")
        fig.update_layout(height=380, template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        if "IF_Score" in scored:
            fig = px.line(choose_chart_data(scored), x="Timestamp", y="IF_Score", title="Isolation Forest anomaly score")
            fig.update_layout(height=320, template="plotly_dark" if theme == "Dark" else "plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        if "AE_Error" in scored:
            err_df = choose_chart_data(scored)
            fig = px.line(err_df, x="Timestamp", y="AE_Error", title="Autoencoder reconstruction error")
            fig.add_hline(y=1.335868, line_dash="dash", annotation_text="Validation 99th-percentile threshold")
            fig.update_layout(height=320, template="plotly_dark" if theme == "Dark" else "plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        if "IF_Score" not in scored and "AE_Error" not in scored:
            st.info("Choose at least one model.")
    with tabs[2]:
        labels, counts = [], []
        if if_anomalies is not None:
            labels.append("Isolation Forest"); counts.append(if_anomalies)
        if ae_anomalies is not None:
            labels.append("Autoencoder"); counts.append(ae_anomalies)
        fig = px.bar(x=labels, y=counts, labels={"x": "Model", "y": "Flagged readings"}, title="Flagged readings by model")
        fig.update_layout(height=320, template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("### Anomaly monitoring")
    st.caption("Inspect individual model flags. A flag indicates an unusual pattern, not a confirmed appliance fault.")
    if available_flags:
        only_flags = scored.loc[scored["Any_Anomaly"]].copy()
        if only_flags.empty:
            st.success("No candidate anomalies were flagged in this period.")
        else:
            sort_col = "AE_Error" if "AE_Error" in only_flags.columns else (
                "IF_Score" if "IF_Score" in only_flags.columns else "Global_active_power"
            )
            only_flags = only_flags.sort_values(sort_col, ascending=False)
            display_cols = ["Timestamp", "Global_active_power"] + available_flags
            for optional in ["IF_Score", "AE_Error"]:
                if optional in only_flags.columns:
                    display_cols.append(optional)
            st.metric("Candidate anomaly readings", f"{len(only_flags):,}")
            st.dataframe(only_flags[display_cols], width="stretch", height=420)
            st.download_button(
                "Download flagged readings (CSV)",
                data=only_flags[display_cols].to_csv(index=False).encode("utf-8"),
                file_name="edgewatt_flagged_readings.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No model flags are available. Select a model in the sidebar.")

# Always include a compact data table/download at the bottom.
with st.expander("View analyzed readings and export results"):
    default_cols = ["Timestamp", "Global_active_power"]
    for col in ("IF_Anomaly", "AE_Anomaly", "IF_Score", "AE_Error"):
        if col in scored.columns:
            default_cols.append(col)
    st.dataframe(scored[default_cols].tail(500), width="stretch", height=300)
    st.caption("The table shows the latest 500 analyzed rows. CSV export includes all analyzed rows.")
    st.download_button(
        "Download analyzed readings (CSV)",
        data=scored[default_cols].to_csv(index=False).encode("utf-8"),
        file_name="edgewatt_analyzed_readings.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_all_rows",
    )

st.divider()
st.caption("EdgeWatt Â· Research prototype Â· Historical UCI household power data Â· Model outputs require validation against labeled events.")
