"""
Backtesting module.
Simulates a trading strategy based on model predictions and computes
performance metrics (total return, Sharpe ratio, max drawdown).
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple

from data_loader import download_data, preprocess_data
from features import prepare_data
from model import StockDataset, MLPPredictor


def generate_signals(
    model: torch.nn.Module,
    X_test: np.ndarray,
    seq_len: int,
    threshold: float = 0.55,
    device: torch.device = None
) -> np.ndarray:
    """
    Generate binary trading signals (1 = long, 0 = flat) from model predictions.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    X_test : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix (chronological order).
    seq_len : int
        Window length used during training.
    threshold : float
        Probability above which a buy signal is triggered.
    device : torch.device, optional
        Device for inference.

    Returns
    -------
    np.ndarray of int (0 or 1), same length as X_test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    signals = np.zeros(len(X_test), dtype=int)

    with torch.no_grad():
        for i in range(len(X_test) - seq_len + 1):
            window = X_test[i:i + seq_len]                # (seq_len, n_features)
            window_flat = torch.tensor(window.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            prob = model(window_flat).item()
            signals[i + seq_len - 1] = 1 if prob > threshold else 0

    return signals


def compute_backtest_metrics(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 10_000.0,
    commission: float = 0.001
) -> Tuple[float, float, float, np.ndarray]:
    """
    Simulate a simple long-only strategy and compute performance metrics.

    Parameters
    ----------
    prices : np.ndarray
        Array of closing prices.
    signals : np.ndarray
        Binary array (1 = invested, 0 = out of market).
    initial_capital : float
        Starting capital.
    commission : float
        Transaction cost as a fraction (e.g. 0.001 = 0.1%).

    Returns
    -------
    total_return : float
        Total percentage return of the strategy.
    sharpe_ratio : float
        Annualised Sharpe ratio (assuming 252 trading days, 0% risk-free rate).
    max_drawdown : float
        Maximum drawdown as a negative fraction (e.g. -0.25 = -25%).
    equity_curve : np.ndarray
        Daily portfolio value.
    """
    n = len(prices)
    equity = np.zeros(n)
    equity[0] = initial_capital

    position = 0          # 0 = out, 1 = invested
    prev_signal = 0

    daily_returns = np.zeros(n)

    for t in range(1, n):
        # Check if signal changed → apply transaction cost
        if signals[t] != prev_signal:
            if signals[t] == 1:   # enter position
                cost = equity[t-1] * commission
                equity[t] = equity[t-1] - cost
            else:                 # exit position
                cost = equity[t-1] * commission
                equity[t] = equity[t-1] - cost
            prev_signal = signals[t]
            position = signals[t]
        else:
            equity[t] = equity[t-1]

        # Apply market return if invested
        if position == 1:
            daily_return = (prices[t] / prices[t-1] - 1)
            equity[t] = equity[t] * (1 + daily_return)

        daily_returns[t] = (equity[t] / equity[t-1]) - 1 if equity[t-1] > 0 else 0.0

    # Total return
    total_return = (equity[-1] / initial_capital - 1) * 100

    # Sharpe ratio (annualised)
    mean_ret = np.mean(daily_returns[1:])
    std_ret = np.std(daily_returns[1:])
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    return total_return, sharpe_ratio, max_drawdown, equity


def buy_and_hold_metrics(prices: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute buy & hold metrics for comparison.

    Returns
    -------
    total_return : float (%)
    sharpe_ratio : float
    max_drawdown : float (%)
    """
    n = len(prices)
    daily_returns = prices[1:] / prices[:-1] - 1

    total_return = (prices[-1] / prices[0] - 1) * 100
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    max_dd = np.min(drawdown) * 100

    return total_return, sharpe, max_dd


def run_backtest(
    ticker: str = "AAPL",
    period: str = "10y",
    horizon: int = 5,
    seq_len: int = 10,
    model_path: str = "models/best_model.pth",
    hidden_dims: list = [128, 64]
) -> dict:
    """
    Run the full backtesting pipeline.

    Parameters
    ----------
    ticker, period, horizon, seq_len : same as during training.
    model_path : str
        Path to the saved model state dict.
    hidden_dims : list
        Must match the architecture of the saved model.

    Returns
    -------
    dict with all metrics and equity curves.
    """
    print(f"\n{'='*50}")
    print(f"Running backtest for {ticker} ({period}, horizon={horizon}d)")
    print(f"{'='*50}")

    # 1. Load and prepare data
    raw = download_data(ticker=ticker, period=period, save_path=f"data/raw/{ticker}.csv")
    clean = preprocess_data(raw)
    X_train, X_test, y_train, y_test, _ = prepare_data(clean, horizon=horizon)

    # 2. Load model
    input_dim = seq_len * X_train.shape[1]
    model = MLPPredictor(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Generate signals on test set
    signals = generate_signals(model, X_test, seq_len=seq_len, device=device)
    print(f"Signals generated: {np.sum(signals)} long / {len(signals)} total "
          f"({100 * np.sum(signals) / len(signals):.1f}% invested)")

    # 4. Get prices for test period (use raw data aligned with X_test)
    #    The test period starts right after the train split
    prices_test = clean['Close'].values[-len(X_test):]

    # 5. Compute metrics
    strat_return, strat_sharpe, strat_dd, equity = compute_backtest_metrics(prices_test, signals)
    bh_return, bh_sharpe, bh_dd = buy_and_hold_metrics(prices_test)

    # 6. Summary
    results = {
        "strategy_total_return_%": round(strat_return, 2),
        "strategy_sharpe": round(strat_sharpe, 2),
        "strategy_max_drawdown_%": round(strat_dd, 2),
        "buyhold_total_return_%": round(bh_return, 2),
        "buyhold_sharpe": round(bh_sharpe, 2),
        "buyhold_max_drawdown_%": round(bh_dd, 2),
        "equity_curve": equity,
        "prices_test": prices_test
    }

    print(f"\n{'Strategy':<25} {'Buy & Hold'}")
    print(f"{'-'*40}")
    print(f"Return:     {strat_return:>8.2f}%       {bh_return:>8.2f}%")
    print(f"Sharpe:     {strat_sharpe:>8.2f}        {bh_sharpe:>8.2f}")
    print(f"Max DD:     {strat_dd:>8.2f}%       {bh_dd:>8.2f}%")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    results = run_backtest(
        ticker="AAPL",
        period="10y",
        horizon=5,
        seq_len=10,
        model_path="models/best_model.pth"
    )