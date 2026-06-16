"""
Automatic hyperparameter tuning – logs all runs to MLflow,
saves only the globally best model after all runs complete.
"""

import itertools
import torch
import mlflow
import numpy as np

from data_loader import download_data, preprocess_data
from features import prepare_data
from model import StockDataset, MLPPredictor
from train import train_one_epoch, evaluate

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
GRID = {
    "seq_len": [5, 10, 20],
    "hidden_dims": [[64, 32], [128, 64], [256, 128, 64]],
    "lr": [0.001, 0.0005],
    "dropout": [0.3, 0.5],
    "batch_size": [32, 64],
}

TICKER = "AAPL"
PERIOD = "10y"
HORIZON = 5
TEST_SIZE = 0.2
MAX_EPOCHS = 50
EXPERIMENT_NAME = "neural_backtester_tuning"
EARLY_STOP_PATIENCE = 10

# Global variables to track the best model seen so far
best_global_val_loss = float("inf")
best_global_state = None
best_global_params = None


def run_single_experiment(params, X_train, X_test, y_train, y_test):
    """Train one model, log to MLflow (no artifact), and update global best."""
    global best_global_val_loss, best_global_state, best_global_params

    # Datasets
    train_ds = StockDataset(X_train, y_train, seq_len=params["seq_len"])
    test_ds = StockDataset(X_test, y_test, seq_len=params["seq_len"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=params["batch_size"], shuffle=False
    )

    # Model
    input_dim = params["seq_len"] * X_train.shape[1]
    model = MLPPredictor(
        input_dim,
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict_local = None

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("input_dim", input_dim)

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save model state to memory
                best_state_dict_local = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {EARLY_STOP_PATIENCE} epochs)."
                )
                break

        # Update global best if this run is better
        if best_state_dict_local is not None and best_val_loss < best_global_val_loss:
            best_global_val_loss = best_val_loss
            best_global_state = best_state_dict_local
            best_global_params = params.copy()
            print(f"  => New global best: val_loss = {best_global_val_loss:.4f}")

        print(f"  Local best val_loss: {best_val_loss:.4f}")


def main():
    # ------------------------------------------------------------------
    # 1. Load data ONCE
    # ------------------------------------------------------------------
    print("Loading and preparing data...")
    raw = download_data(
        ticker=TICKER, period=PERIOD, save_path=f"data/raw/{TICKER}.csv"
    )
    clean = preprocess_data(raw)
    X_train, X_test, y_train, y_test, _ = prepare_data(
        clean, horizon=HORIZON, test_size=TEST_SIZE
    )
    print(
        f"Data ready: {X_train.shape[0]} train, {X_test.shape[0]} test, "
        f"{X_train.shape[1]} features."
    )

    # ------------------------------------------------------------------
    # 2. Generate all hyperparameter combinations
    # ------------------------------------------------------------------
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    print(f"Total runs: {len(combinations)}")

    # ------------------------------------------------------------------
    # 3. Run experiments (no model files saved yet)
    # ------------------------------------------------------------------
    for i, params in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{len(combinations)} – {params}")
        print(f"{'='*60}")
        try:
            run_single_experiment(params, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Run failed with error: {e}")
            continue

    # ------------------------------------------------------------------
    # 4. Save the globally best model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("All runs finished. Saving the best model...")
    print(f"Best params: {best_global_params}")
    print(f"Best val_loss: {best_global_val_loss:.4f}")
    print(f"{'='*60}")

    if best_global_state is not None:
        # Save to disk
        torch.save(best_global_state, "models/best_model.pth")

        # Also log as an artifact under a new MLflow run (optional but nice)
        with mlflow.start_run(run_name="best_model"):
            mlflow.log_params(best_global_params or {})
            mlflow.log_metric("val_loss", best_global_val_loss)
            mlflow.log_artifact("models/best_model.pth")
        print("Best model saved as models/best_model.pth and logged to MLflow.")
    else:
        print("No model was saved – all runs may have failed.")


if __name__ == "__main__":
    main()