"""
Module for downloading and preprocessing stock market data.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


def download_data(
    ticker: str = "AAPL",
    period: str = "10y",
    save_path: str = "data/raw/AAPL.csv"
) -> pd.DataFrame:
    """
    Downloads historical daily data for a given ticker and saves it as a CSV file.

    Parameters
    ----------
    ticker : str
        Ticker symbol (default 'AAPL').
    period : str
        Data period in yfinance format (e.g. '10y', '5y', 'max').
    save_path : str
        Output file path (directories will be created if they don't exist).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close, etc.
    """
    print(f"Downloading data for {ticker} over the last {period}...")
    df = yf.download(ticker, period=period, auto_adjust=False)

    if df.empty:
        raise ValueError(f"Failed to download data for {ticker}. Check the symbol or your connection.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Data saved to {save_path}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data and adds a log return column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional 'log_return' column.
    """
    # Flatten columns if yfinance returned a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Forward fill missing values
    df = df.ffill()

    # Logarithmic returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Drop the first row (NaN log return)
    df = df.dropna(subset=['log_return']).copy()

    print(f"Data after cleaning: {len(df)} rows, columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    raw_df = download_data(ticker="AAPL", period="10y", save_path="data/raw/AAPL.csv")
    clean_df = preprocess_data(raw_df)
    print(clean_df.head())