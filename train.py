import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    drop_constant_sensors,
    fit_minmax_scaler,
    last_window_per_engine,
    load_dataset,
    sliding_windows,
)
from src.metrics import nasa_score, rmse
from src.model import LSTMRUL
from src.plots import (
    classification_accuracy,
    layer_mse_dict,
    plot_learning_curves,
    plot_mse_per_layer,
    plot_rul_and_classification,
    plot_weights,
    predict_from_loader,
)


def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, x: np.ndarray, device) -> np.ndarray:
    model.eval()
    return np.maximum(model(torch.from_numpy(x).to(device)).cpu().numpy(), 0.0)


@torch.no_grad()
def mse_on_array(model, x: np.ndarray, y: np.ndarray, device, batch_size: int = 512) -> float:
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        preds.append(model(xb).cpu().numpy())
    preds = np.maximum(np.concatenate(preds), 0.0)
    return mse_np(y, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--plot-sample", type=int, default=800, help="Rozmiar próbki do wykresów MSE warstw")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    data_path = Path(cfg["data_dir"])
    window_size = cfg["window_size"]
    max_rul = cfg["max_rul"]
    rul_critical = cfg.get("rul_critical_cycles", 30)
    var_thresh = cfg["variance_threshold"]
    seed = cfg["random_seed"]
    val_ratio = cfg["val_ratio"]
    degradation_only = cfg.get("train_degradation_only", True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading {dataset_id}...")
    train, test, rul_test = load_dataset(data_path, dataset_id)

    feature_cols, dropped = drop_constant_sensors(train, var_thresh)
    print(f"Dropping {len(dropped)} constant sensors: {dropped}")

    train, test, scaler = fit_minmax_scaler(train, test, feature_cols)

    train_units, val_units = train_test_split(
        train["unit_id"].unique(), test_size=val_ratio, random_state=seed
    )

    print(f"Building windows (size={window_size}, stride={args.stride})...")
    x_train, y_train = sliding_windows(
        train, feature_cols, window_size, max_rul, train_units, args.stride
    )
    x_val, y_val = sliding_windows(
        train, feature_cols, window_size, max_rul, val_units, args.stride
    )

    if degradation_only:
        m_tr, m_va = y_train < max_rul, y_val < max_rul
        x_train, y_train = x_train[m_tr], y_train[m_tr]
        x_val, y_val = x_val[m_va], y_val[m_va]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | train samples: {len(x_train)} | features: {len(feature_cols)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
    )

    model = LSTMRUL(len(feature_cols), args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history = {
        "train_mse": [],
        "train_mse_full": [],
        "val_mse": [],
        "train_acc": [],
        "val_acc": [],
        "rul_critical_cycles": rul_critical,
    }

    n_plot_sample = min(args.plot_sample, len(x_train))
    plot_idx = np.random.default_rng(seed).choice(len(x_train), n_plot_sample, replace=False)
    x_plot_sample = x_train[plot_idx]
    y_plot_sample = y_train[plot_idx]

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, train_loader, optimizer, criterion, device)
        train_mse_full = mse_on_array(model, x_train, y_train, device)
        y_v, y_vp = predict_from_loader(model, val_loader, device)
        val_mse = mse_np(y_v, y_vp)

        y_tr_p = predict(model, x_plot_sample, device)
        train_acc = classification_accuracy(y_plot_sample, y_tr_p, rul_critical)
        val_acc = classification_accuracy(y_v, y_vp, rul_critical)

        history["train_mse"].append(train_mse)
        history["train_mse_full"].append(train_mse_full)
        history["val_mse"].append(val_mse)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | MSE train {train_mse:.2f} | MSE train (cały) {train_mse_full:.2f} | "
                f"MSE val {val_mse:.2f} | dokł. kl. {val_acc * 100:.1f}%"
            )

    out = Path("results") / dataset_id / "lstm"
    plots_dir = out / "wykresy"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Generowanie wykresów...")
    plot_learning_curves(history, plots_dir)

    mse_sample = layer_mse_dict(model, x_plot_sample, y_plot_sample, device)
    mse_full = layer_mse_dict(model, x_train, y_train, device)
    plot_mse_per_layer(mse_sample, mse_full, plots_dir)

    y_v, y_vp = predict_from_loader(model, val_loader, device)
    plot_rul_and_classification(y_v, y_vp, rul_critical, plots_dir, "walidacja")

    x_test = last_window_per_engine(test, feature_cols, window_size)
    pred_test = predict(model, x_test, device)
    y_test = rul_test.to_numpy(dtype=np.float32)
    print(f"Test RMSE (per engine): {rmse(y_test, pred_test):.2f}")
    print(f"Test NASA score: {nasa_score(y_test, pred_test):.2f}")

    plot_rul_and_classification(y_test, pred_test, rul_critical, plots_dir, "test")
    plot_weights(model, plots_dir, hist_dx=cfg.get("weight_hist_dx", 0.05))

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "bundle.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_cols": feature_cols,
                "dropped_sensors": dropped,
                "window_size": window_size,
                "hidden": args.hidden,
                "layers": args.layers,
            },
            f,
        )
    with open(out / "metrics.json", "w") as f:
        json.dump(
            {
                "dataset": dataset_id,
                "dropped_sensors": dropped,
                "n_features": len(feature_cols),
                "val_rmse_windows": rmse(y_v, y_vp),
                "val_nasa_windows": nasa_score(y_v, y_vp),
                "test_rmse": rmse(y_test, pred_test),
                "test_nasa": nasa_score(y_test, pred_test),
                "rul_critical_cycles": rul_critical,
                "test_classification_accuracy": classification_accuracy(
                    y_test, pred_test, rul_critical
                ),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Zapisano model i wykresy w {out}/ (wykresy: {plots_dir}/)")


if __name__ == "__main__":
    main()
