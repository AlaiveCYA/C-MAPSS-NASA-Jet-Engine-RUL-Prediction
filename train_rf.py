#!/usr/bin/env python3
"""Las losowy (Random Forest) — osobny pipeline RUL, niezależny od LSTM."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.data import (
    drop_constant_sensors,
    fit_minmax_scaler,
    load_dataset,
)
from src.metrics import nasa_score, rmse
from src.plots import classification_accuracy, plot_rul_and_classification
from src.rf_features import build_rf_samples, last_row_per_engine


def make_rf(n_estimators: int, max_depth: int | None, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="CMAPSS RUL — Random Forest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    data_path = Path(cfg["data_dir"])
    window_size = cfg["window_size"]
    max_rul = cfg["max_rul"]
    var_thresh = cfg["variance_threshold"]
    seed = cfg["random_seed"]
    val_ratio = cfg["val_ratio"]
    degradation_only = cfg.get("train_degradation_only", True)
    rul_critical = cfg.get("rul_critical_cycles", 30)

    print(f"[RF] Ładowanie {dataset_id}...")
    train, test, rul_test = load_dataset(data_path, dataset_id)

    feature_cols, dropped = drop_constant_sensors(train, var_thresh)
    print(f"[RF] Usunięto {len(dropped)} stałych sensorów: {dropped}")

    train, test, scaler = fit_minmax_scaler(train, test, feature_cols)

    train_units, val_units = train_test_split(
        train["unit_id"].unique(), test_size=val_ratio, random_state=seed
    )

    print(f"[RF] Budowanie cech tabularnych (okno={window_size}, stride={args.stride})...")
    x_train, y_train = build_rf_samples(
        train, feature_cols, window_size, max_rul, train_units, args.stride
    )
    x_val, y_val = build_rf_samples(
        train, feature_cols, window_size, max_rul, val_units, args.stride
    )

    if degradation_only:
        m_tr, m_va = y_train < max_rul, y_val < max_rul
        x_train, y_train = x_train[m_tr], y_train[m_tr]
        x_val, y_val = x_val[m_va], y_val[m_va]

    print(f"[RF] Trening: {len(x_train)} próbek, {x_train.shape[1]} cech...")
    model = make_rf(args.n_estimators, args.max_depth, seed)
    model.fit(x_train, y_train)

    pred_val = np.maximum(model.predict(x_val), 0.0)
    print(f"[RF] Val RMSE (okna): {rmse(y_val, pred_val):.2f}")
    print(f"[RF] Val NASA (okna): {nasa_score(y_val, pred_val):.2f}")

    x_test = last_row_per_engine(test, feature_cols, window_size)
    pred_test = np.maximum(model.predict(x_test), 0.0)
    y_test = rul_test.to_numpy(dtype=np.float32)
    print(f"[RF] Test RMSE (per silnik): {rmse(y_test, pred_test):.2f}")
    print(f"[RF] Test NASA: {nasa_score(y_test, pred_test):.2f}")
    print(f"[RF] Test dokładność dyskryminacji (RUL≤{rul_critical}): "
          f"{classification_accuracy(y_test, pred_test, rul_critical) * 100:.1f}%")

    out = Path("results") / dataset_id / "rf"
    plots_dir = out / "wykresy"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Walidacja: wiele okien z różnym RUL (nie ostatni cykl — tam zawsze RUL=0, bo train → awaria)
    plot_rul_and_classification(y_val, pred_val, rul_critical, plots_dir, "walidacja")
    plot_rul_and_classification(y_test, pred_test, rul_critical, plots_dir, "test")

    with open(out / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "dropped_sensors": dropped,
                "window_size": window_size,
            },
            f,
        )

    importances = model.feature_importances_
    top_k = min(15, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_k]
    with open(out / "metrics.json", "w") as f:
        json.dump(
            {
                "model": "random_forest",
                "dataset": dataset_id,
                "dropped_sensors": dropped,
                "n_features": int(x_train.shape[1]),
                "n_train_samples": int(len(x_train)),
                "val_rmse_windows": rmse(y_val, pred_val),
                "val_nasa_windows": nasa_score(y_val, pred_val),
                "val_classification_accuracy": classification_accuracy(
                    y_val, pred_val, rul_critical
                ),
                "test_rmse": rmse(y_test, pred_test),
                "test_nasa": nasa_score(y_test, pred_test),
                "test_classification_accuracy": classification_accuracy(
                    y_test, pred_test, rul_critical
                ),
                "top_feature_indices": top_idx.tolist(),
                "top_feature_importances": importances[top_idx].tolist(),
            },
            f,
            indent=2,
        )

    print(f"[RF] Zapisano: {out}/")


if __name__ == "__main__":
    main()
