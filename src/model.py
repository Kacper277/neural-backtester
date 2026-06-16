"""
PyTorch Dataset and model definitions for the trading predictor.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    """
    Dataset that slices a flat feature array into overlapping windows
    and returns (window_flat, target) pairs.

    Each sample is a sequence of `seq_len` consecutive days
    flattened into a 1D vector (suitable for an MLP).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 10):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Scaled feature matrix ordered chronologically.
        y : np.ndarray, shape (n_samples,)
            Binary targets.
        seq_len : int
            Number of time steps in each window.
        """
        if len(X) < seq_len:
            raise ValueError(f"Not enough data ({len(X)} rows) for seq_len={seq_len}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        # Extract window of features and flatten
        window = self.X[idx:idx + self.seq_len]          # shape: (seq_len, n_features)
        window_flat = window.flatten()                   # shape: (seq_len * n_features,)
        target = self.y[idx + self.seq_len - 1]           # target at the last timestep
        return window_flat, target


class MLPPredictor(nn.Module):
    """
    Simple multi‑layer perceptron for binary classification.
    Input is the flattened window (seq_len * n_features).
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], dropout: float = 0.3):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (seq_len * n_features).
        hidden_dims : list of int
            Sizes of hidden layers.
        dropout : float
            Dropout probability after each hidden layer.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())    # output probability

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)