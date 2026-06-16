"""
Interactive dashboard for the neural backtester.
Visualises equity curves, signals and metrics. Includes a threshold slider
that recomputes signals and metrics on the fly.
"""

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import download_data, preprocess_data
from features import prepare_data
from model import MLPPredictor


# ---------------------------------------------------------------------------
# Helper functions (reuse logic from backtest.py to keep dashboard standalone)
# ---------------------------------------------------------------------------
def get_probabilities(model, X_test, seq_len, device):
    """Return array of predicted probabilities for each valid window position."""
    model.to(device)
    model.eval()
    probs = np.full(len(X_test), np.nan)

    with torch.no_grad():
        for i in range(len(X_test) - seq_len + 1):
            window = X_test[i : i + seq_len]
            window_flat = torch.tensor(window.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            probs[i + seq_len - 1] = model(window_flat).item()
    return probs


def signals_from_prob(probs, threshold):
    """Convert probabilities to binary signals (1 = long, 0 = flat)."""
    s = np.zeros(len(probs), dtype=int)
    mask = ~np.isnan(probs)
    s[mask] = (probs[mask] > threshold).astype(int)
    return s


def compute_metrics(prices, signals, initial_capital=10_000, commission=0.001):
    """Vectorised backtest (same logic as backtest.py)."""
    n = len(prices)
    equity = np.full(n, initial_capital, dtype=float)
    position = 0
    prev_sig = 0

    for t in range(1, n):
        if signals[t] != prev_sig:
            # Transaction cost
            equity[t] = equity[t-1] * (1 - commission)
            prev_sig = signals[t]
            position = signals[t]
        else:
            equity[t] = equity[t-1]

        if position == 1:
            equity[t] *= prices[t] / prices[t-1]

    # Metrics
    daily_ret = np.diff(equity) / equity[:-1]
    total_ret = (equity[-1] / initial_capital - 1) * 100
    sharpe = (np.mean(daily_ret) / np.std(daily_ret)) * np.sqrt(252) if np.std(daily_ret) > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = np.min(drawdown) * 100
    return total_ret, sharpe, max_dd, equity


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Neural Backtester", layout="wide")
st.title("📈 Neural Network Trading Strategy Backtest")
st.markdown(
    "This dashboard compares an MLP‑based strategy (predicting 5‑day price direction) "
    "with a simple buy‑and‑hold benchmark. Adjust the **threshold** slider in the sidebar "
    "to change how confident the model must be before taking a long position."
)

# ----------------------------- Sidebar --------------------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Signal threshold",
    min_value=0.50,
    max_value=0.80,
    value=0.55,
    step=0.01,
    help="Only take a long position when predicted probability exceeds this value."
)

# ----------------------------- Load data & model once -----------------------
@st.cache_resource
def load_model_and_data():
    """Load or download data, preprocess, build test set, and load trained model."""
    raw = download_data(ticker="AAPL", period="10y", save_path="data/raw/AAPL.csv")
    clean = preprocess_data(raw)
    X_train, X_test, y_train, y_test, _ = prepare_data(clean, horizon=5)

    seq_len = 10
    input_dim = seq_len * X_train.shape[1]
    model = MLPPredictor(input_dim=input_dim, hidden_dims=[128, 64])
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()

    return clean, X_test, seq_len, model

clean, X_test, seq_len, model = load_model_and_data()

# Compute probabilities once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
probs = get_probabilities(model, X_test, seq_len, device)

# Derive signals based on current threshold
signals = signals_from_prob(probs, threshold)

# Extract prices for the test period
prices_test = clean['Close'].values[-len(X_test):]

# Strategy metrics
strat_ret, strat_sharpe, strat_dd, equity = compute_metrics(prices_test, signals)

# Buy & hold metrics
bh_ret = (prices_test[-1] / prices_test[0] - 1) * 100
bh_daily = np.diff(prices_test) / prices_test[:-1]
bh_sharpe = (np.mean(bh_daily) / np.std(bh_daily)) * np.sqrt(252) if np.std(bh_daily) > 0 else 0.0
bh_peak = np.maximum.accumulate(prices_test)
bh_drawdown = (prices_test - bh_peak) / bh_peak
bh_dd = np.min(bh_drawdown) * 100

# ----------------------------- Metrics row ----------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy Return", f"{strat_ret:.2f}%")
col2.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
col3.metric("Strategy Max DD", f"{strat_dd:.2f}%")
col4.metric("Buy & Hold Return", f"{bh_ret:.2f}%")

# ----------------------------- Charts ---------------------------------------
# 1. Equity curves
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    y=equity, name="Strategy", line=dict(color="blue")
))
fig1.add_trace(go.Scatter(
    y=prices_test / prices_test[0] * 10_000,
    name="Buy & Hold",
    line=dict(color="gray", dash="dash")
))
fig1.update_layout(
    title="Equity Curve (starting capital = $10,000)",
    xaxis_title="Trading days (test set)",
    yaxis_title="Portfolio value ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Price with signals
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(y=prices_test, name="Close Price", line=dict(color="black")))
buy_mask = signals == 1
fig2.add_trace(go.Scatter(
    x=np.where(buy_mask)[0], y=prices_test[buy_mask],
    mode="markers", name="Long signal",
    marker=dict(color="green", symbol="triangle-up", size=10)
))
fig2.update_layout(
    title="Price and Long Signals",
    xaxis_title="Trading days (test set)",
    yaxis_title="Price ($)"
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Probability distribution
fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=probs[~np.isnan(probs)], nbinsx=30, name="Predicted probability"))
fig3.add_vline(x=threshold, line_dash="dot", line_color="red", annotation_text="Threshold")
fig3.update_layout(
    title="Distribution of Predicted Probabilities",
    xaxis_title="P(up)",
    yaxis_title="Count"
)
st.plotly_chart(fig3, use_container_width=True)

st.caption("Model: MLP trained on 10 years of AAPL data, 5‑day forecast horizon, 10‑day lookback window.")