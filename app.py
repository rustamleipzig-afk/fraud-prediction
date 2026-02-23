"""Fraud Prediction — Streamlit application."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import predict, train_model
from src.preprocessing import FEATURE_COLUMNS, load_data, preprocess_input

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")

st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _ensure_model():
    """Train and cache the model in session state if not already done."""
    if "model" not in st.session_state:
        with st.spinner("Training model on sample data…"):
            df = load_data(DATA_PATH)
            model, scaler, report, cm, importances = train_model(df)
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["report"] = report
            st.session_state["cm"] = cm
            st.session_state["importances"] = importances
            st.session_state["df"] = df


# ---------------------------------------------------------------------------
# Sidebar — transaction input
# ---------------------------------------------------------------------------

def _sidebar_inputs() -> dict:
    st.sidebar.header("Transaction Details")

    amount = st.sidebar.number_input(
        "Transaction Amount (€)", min_value=0.01, max_value=50000.0, value=120.0, step=1.0
    )
    time_of_day = st.sidebar.slider("Time of Day (hour)", 0.0, 23.99, 14.0, step=0.25)
    day_of_week = st.sidebar.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    )
    distance_from_home = st.sidebar.number_input(
        "Distance from Home (km)", min_value=0.0, max_value=1000.0, value=5.0, step=0.5
    )
    distance_from_last = st.sidebar.number_input(
        "Distance from Last Transaction (km)", min_value=0.0, max_value=500.0, value=2.0, step=0.5
    )
    ratio_to_median = st.sidebar.number_input(
        "Ratio to Median Purchase Price", min_value=0.0, max_value=20.0, value=1.0, step=0.1
    )
    repeat_retailer = st.sidebar.checkbox("Repeat Retailer", value=True)
    used_chip = st.sidebar.checkbox("Chip Used", value=True)
    used_pin_online = st.sidebar.checkbox("PIN Used Online", value=False)
    online_order = st.sidebar.checkbox("Online Order", value=False)

    return {
        "amount": amount,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "distance_from_home": distance_from_home,
        "distance_from_last_transaction": distance_from_last,
        "ratio_to_median_purchase_price": ratio_to_median,
        "repeat_retailer": int(repeat_retailer),
        "used_chip": int(used_chip),
        "used_pin_online": int(used_pin_online),
        "online_order": int(online_order),
    }


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def _section_prediction(inputs: dict, model, scaler):
    st.header("🔍 Fraud Prediction")
    input_df = preprocess_input(inputs)
    prediction, probability = predict(model, scaler, input_df)

    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.error(f"⚠️ **Fraudulent Transaction** detected ({probability:.1%} confidence)")
    else:
        col1.success(f"✅ **Legitimate Transaction** ({1 - probability:.1%} confidence)")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson" if prediction == 1 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightsalmon"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    gauge.update_layout(height=300)
    col2.plotly_chart(gauge, use_container_width=True)

    with st.expander("Input summary"):
        st.json(inputs)


def _section_dataset(df: pd.DataFrame):
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    fraud_count = int(df["fraud"].sum())
    col2.metric("Fraudulent", f"{fraud_count:,}", f"{fraud_count / len(df):.1%}")
    col3.metric("Legitimate", f"{len(df) - fraud_count:,}")

    col_a, col_b = st.columns(2)

    fig_dist = px.histogram(
        df,
        x="amount",
        color="fraud",
        barmode="overlay",
        nbins=50,
        title="Transaction Amount Distribution",
        labels={"fraud": "Fraud", "amount": "Amount (€)"},
        color_discrete_map={0: "steelblue", 1: "crimson"},
    )
    col_a.plotly_chart(fig_dist, use_container_width=True)

    fig_pie = px.pie(
        df,
        names=df["fraud"].map({0: "Legitimate", 1: "Fraudulent"}),
        title="Class Distribution",
        color_discrete_sequence=["steelblue", "crimson"],
    )
    col_b.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("View raw data (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)


def _section_model_performance(report: dict, cm: np.ndarray, importances: dict):
    st.header("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.2%}")
    col2.metric("Precision (fraud)", f"{report['1']['precision']:.2%}")
    col3.metric("Recall (fraud)", f"{report['1']['recall']:.2%}")
    col4.metric("F1-Score (fraud)", f"{report['1']['f1-score']:.2%}")

    col_a, col_b = st.columns(2)

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual"},
        x=["Legitimate", "Fraudulent"],
        y=["Legitimate", "Fraudulent"],
        title="Confusion Matrix",
        color_continuous_scale="Blues",
    )
    col_a.plotly_chart(cm_fig, use_container_width=True)

    imp_df = (
        pd.DataFrame.from_dict(importances, orient="index", columns=["importance"])
        .sort_values("importance", ascending=True)
    )
    imp_fig = px.bar(
        imp_df,
        x="importance",
        y=imp_df.index,
        orientation="h",
        title="Feature Importances",
        labels={"importance": "Importance", "y": "Feature"},
    )
    col_b.plotly_chart(imp_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("💳 Fraud Prediction App")
    st.caption("Powered by a Random Forest classifier trained on transaction data.")

    _ensure_model()

    inputs = _sidebar_inputs()

    tabs = st.tabs(["Prediction", "Dataset", "Model Performance"])

    with tabs[0]:
        _section_prediction(inputs, st.session_state["model"], st.session_state["scaler"])

    with tabs[1]:
        _section_dataset(st.session_state["df"])

    with tabs[2]:
        _section_model_performance(
            st.session_state["report"],
            st.session_state["cm"],
            st.session_state["importances"],
        )


if __name__ == "__main__":
    main()
