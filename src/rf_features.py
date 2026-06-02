"""Cechy tabularne dla lasów losowych (okno czasowe → wektor)."""

import numpy as np
import pandas as pd

from src.data import _pad_window


def window_stats(window: np.ndarray) -> np.ndarray:
    """Średnia, odch. std. i ostatni cykl dla każdego sensora."""
    return np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0),
            window[-1],
        ]
    ).astype(np.float32)


def build_rf_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    max_rul: int,
    units: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Jedna próbka tabularna na cykl (okno kończące się w tym cyklu)."""
    xs, ys = [], []
    subset = df if units is None else df[df["unit_id"].isin(units)]
    for _, g in subset.groupby("unit_id"):
        g = g.sort_values("time_cycles")
        values = g[feature_cols].to_numpy(dtype=np.float32)
        labels = np.minimum(g["time_cycles"].max() - g["time_cycles"].to_numpy(), max_rul)
        for i in range(0, len(g), max(1, stride)):
            window = _pad_window(values[: i + 1], window_size)
            xs.append(window_stats(window))
            ys.append(labels[i])
    return np.stack(xs), np.array(ys, dtype=np.float32)


def last_row_per_engine(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> np.ndarray:
    """Jedna próbka tabularna na silnik — okno kończące się w ostatnim cyklu."""
    rows = []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("time_cycles")
        window = _pad_window(g[feature_cols].to_numpy(dtype=np.float32), window_size)
        rows.append(window_stats(window))
    return np.stack(rows)
