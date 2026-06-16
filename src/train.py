"""
Training script with MLflow experiment tracking.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from data_loader import download_data, preprocess_data
from features import prepare_data
from model import StockDataset, MLPPredictor


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / total, correct / total


def main(args):
    # 1. Prepare data
    raw = download_data(ticker=args.ticker, period=args.period,
                        save_path=f"data/raw/{args.ticker}.csv")
    clean = preprocess_data(raw)
    X_train, X_test, y_train, y_test, _ = prepare_data(clean, horizon=args.horizon,
                                                       test_size=args.test_size)

    # 2. Create datasets and loaders
    train_ds = StockDataset(X_train, y_train, seq_len=args.seq_len)
    test_ds = StockDataset(X_test, y_test, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Model, loss, optimizer
    input_dim = args.seq_len * X_train.shape[1]   # n_features * seq_len
    model = MLPPredictor(input_dim, hidden_dims=args.hidden_dims, dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. MLflow tracking
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("input_dim", input_dim)
        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            print(f"Epoch {epoch:2d} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "models/best_model.pth")
                mlflow.log_artifact("models/best_model.pth")

    print("Training complete. Best model saved at models/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP trading predictor.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--period", type=str, default="10y", help="Data period")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (days)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seq_len", type=int, default=10, help="Window length (days)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64],
                        help="Hidden layer sizes (e.g., 128 64)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--experiment_name", type=str, default="neural_backtester")
    args = parser.parse_args()

    main(args)