"""Load, clean, scale, and window CMAPSS data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMNS = (
    ["unit_id", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def load_dataset(
    data_path: Path, dataset_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(
        data_path / f"train_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    test = pd.read_csv(
        data_path / f"test_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    rul = pd.read_csv(
        data_path / f"RUL_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )["RUL"]
    return train, test, rul


def drop_constant_sensors(
    train: pd.DataFrame, variance_threshold: float = 0.01
) -> tuple[list[str], list[str]]:
    variances = train[SENSOR_COLUMNS].var()
    dropped = variances[variances < variance_threshold].index.tolist()
    feature_cols = [c for c in SENSOR_COLUMNS if c not in dropped]
    return feature_cols, dropped


def fit_minmax_scaler(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train = train.copy()
    test = test.copy()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return train, test, scaler


def _pad_window(chunk: np.ndarray, window_size: int) -> np.ndarray:
    if len(chunk) >= window_size:
        return chunk[-window_size:]
    pad = np.zeros((window_size - len(chunk), chunk.shape[1]), dtype=np.float32)
    return np.vstack([pad, chunk]).astype(np.float32)


def sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    max_rul: int,
    units: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    subset = df if units is None else df[df["unit_id"].isin(units)]
    for _, g in subset.groupby("unit_id"):
        g = g.sort_values("time_cycles")
        values = g[feature_cols].to_numpy(dtype=np.float32)
        labels = np.minimum(g["time_cycles"].max() - g["time_cycles"].to_numpy(), max_rul)
        for i in range(0, len(g), max(1, stride)):
            xs.append(_pad_window(values[: i + 1], window_size))
            ys.append(labels[i])
    return np.stack(xs), np.array(ys, dtype=np.float32)


def last_window_per_engine(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> np.ndarray:
    windows = []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("time_cycles")
        windows.append(_pad_window(g[feature_cols].to_numpy(dtype=np.float32), window_size))
    return np.stack(windows)


def last_cycle_rul(df: pd.DataFrame, max_rul: int) -> np.ndarray:
    """
    RUL w ostatnim wierszu każdego silnika.
    Dla zbioru TRENINGOWEGO (run-to-failure) ostatni cykl = awaria → zawsze 0.
    Do wykresów walidacji używaj wielu okien (build_rf_samples / sliding_windows), nie tej funkcji.
    """
    labels = []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("time_cycles")
        labels.append(min(g["time_cycles"].max() - g["time_cycles"].iloc[-1], max_rul))
    return np.array(labels, dtype=np.float32)
