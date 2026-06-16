"""
Feature engineering module.
Creates target labels, technical indicators, performs train/test split
(temporally ordered) and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Adds a binary target column indicating whether the price will be higher
    after `horizon` days.

    Target: 1 if Close[t+horizon] > Close[t], else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' column.
    horizon : int
        Number of days ahead for prediction.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'target' column, rows with NaN target removed.
    """
    df = df.copy()
    future_close = df['Close'].shift(-horizon)
    df['target'] = (future_close > df['Close']).astype(int)
    df = df.dropna(subset=['target']).copy()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators and returns a DataFrame with only the features
    and target (if present). Drops rows with any NaN created by rolling windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close', 'log_return'.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus new feature columns, NaN rows removed.
    """
    df = df.copy()

    # Simple Moving Averages
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()

    # RSI (Relative Strength Index) – 14-day
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # Historical volatility (standard deviation of log returns over 20 days)
    df['volatility_20'] = df['log_return'].rolling(window=20).std()

    # Difference between short and long SMAs (momentum signal)
    df['sma_diff_5_20'] = df['sma_5'] - df['sma_20']

    # Lagged log returns (autoregressive terms)
    df['lag_return_1'] = df['log_return'].shift(1)
    df['lag_return_2'] = df['log_return'].shift(2)
    df['lag_return_3'] = df['log_return'].shift(3)

    # Drop rows with NaN produced by rolling/shift operations
    df = df.dropna().copy()
    return df


def prepare_data(
    df: pd.DataFrame,
    horizon: int = 5,
    test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Full data preparation pipeline: adds labels, features, splits
    chronologically, and scales the features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with at least 'Close' and 'log_return'.
    horizon : int
        Forecast horizon (days ahead).
    test_size : float
        Fraction of the data to use for testing (from the end).

    Returns
    -------
    X_train : np.ndarray
    X_test : np.ndarray
    y_train : np.ndarray
    y_test : np.ndarray
    scaler : StandardScaler
        Fitted scaler (only on training data) for potential inverse transform.
    """
    # 1. Create target
    df = add_labels(df, horizon=horizon)

    # 2. Add technical features
    df = add_features(df)

    # Feature columns (excluding non-numeric / non-feature columns)
    feature_cols = [
        'log_return', 'sma_5', 'sma_10', 'sma_20',
        'rsi_14', 'volatility_20', 'sma_diff_5_20',
        'lag_return_1', 'lag_return_2', 'lag_return_3'
    ]

    X = df[feature_cols].values
    y = df['target'].values

    # 3. Chronological split (no shuffling!)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4. Scale features using only training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data prepared: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples. "
          f"Feature count: {X_train.shape[1]}.")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Quick test using the data loader
    from data_loader import download_data, preprocess_data

    raw = download_data(ticker="AAPL", period="10y", save_path="data/raw/AAPL.csv")
    clean = preprocess_data(raw)
    X_tr, X_te, y_tr, y_te, scaler = prepare_data(clean, horizon=5, test_size=0.2)
    print("Feature means (train):", X_tr.mean(axis=0))
    print("Target distribution (train):", np.bincount(y_tr))
    print("Target distribution (test):", np.bincount(y_te))