"""Wykresy treningu — każdy wykres zapisany osobno (LaTeX)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.model import LSTMRUL

# Wysoka rozdzielczość pod \includegraphics w LaTeX
SAVE_KW = dict(dpi=150, bbox_inches="tight")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@torch.no_grad()
def predict_from_loader(model: LSTMRUL, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(targets), np.maximum(np.concatenate(preds), 0.0)


@torch.no_grad()
def _pred_from_hidden(model: LSTMRUL, h: torch.Tensor) -> np.ndarray:
    z = model.head[0](h)
    return model.head[2](z).squeeze(-1).cpu().numpy()


@torch.no_grad()
def layer_mse_dict(
    model: LSTMRUL,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    t = torch.from_numpy(x).to(device)
    y_true = y.astype(np.float32)
    acts = model.forward_layers(t)
    result = {}
    for name, h in acts.items():
        if name == "wyjscie":
            pred = h.cpu().numpy()
        elif name == "FC_1":
            pred = model.head[2](h).squeeze(-1).cpu().numpy()
        else:
            pred = _pred_from_hidden(model, h)
        result[name] = mse(y_true, pred)
    return result


def rul_to_binary(rul: np.ndarray, critical_cycles: float = 30) -> np.ndarray:
    return (rul <= critical_cycles).astype(int)


def classification_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, critical_cycles: float = 30
) -> float:
    yt = rul_to_binary(y_true, critical_cycles)
    yp = rul_to_binary(y_pred, critical_cycles)
    return float(np.mean(yt == yp))


def plot_learning_curves(history: dict, out_dir: Path) -> None:
    epochs = range(1, len(history["train_mse"]) + 1)
    crit = history.get("rul_critical_cycles", 30)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, history["train_mse"], label="MSE — batch (uczenie)", marker="o", ms=3)
    ax1.plot(epochs, history["train_mse_full"], label="MSE — cały zbiór uczący", marker="s", ms=3)
    ax1.plot(epochs, history["val_mse"], label="MSE — walidacja", marker="^", ms=3)
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("MSE")
    ax1.set_title("Błąd MSE w trakcie uczenia")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    _save(fig1, out_dir / "uczenie_mse_epoki.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(epochs, history["train_acc"], label="Uczenie (próbka)", marker="o", ms=3)
    ax2.plot(epochs, history["val_acc"], label="Walidacja", marker="^", ms=3)
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Dokładność")
    ax2.set_title(f"Dyskryminacja RUL ≤ {crit} cykli — dokładność w epokach")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    _save(fig2, out_dir / "uczenie_dokladnosc_dyskryminacji_epoki.png")


def plot_mse_per_layer(
    mse_sample: dict[str, float],
    mse_full: dict[str, float],
    out_dir: Path,
) -> None:
    layers = list(mse_sample.keys())
    x = np.arange(len(layers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [mse_sample[k] for k in layers], w, label="Próbka ucząca")
    ax.bar(x + w / 2, [mse_full[k] for k in layers], w, label="Cały zbiór uczący")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=15)
    ax.set_ylabel("MSE")
    ax.set_title("MSE we wszystkich warstwach sieci")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir / "mse_warstwy.png")


def plot_rul_and_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_cycles: float,
    out_dir: Path,
    prefix: str,
) -> None:
    """Zapisuje 4 osobne pliki: scatter, histogram, macierz, słupki."""
    yt = rul_to_binary(y_true, critical_cycles)
    yp = rul_to_binary(y_pred, critical_cycles)
    acc = classification_accuracy(y_true, y_pred, critical_cycles)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.55, s=22, edgecolors="none")
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    ax.axvline(critical_cycles, color="orange", linestyle=":", lw=1, label=f"Próg {critical_cycles} cykli")
    ax.axhline(critical_cycles, color="orange", linestyle=":", lw=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="Idealna predykcja")
    ax.set_xlabel("Rzeczywiste RUL [cykle]")
    ax.set_ylabel("Przewidywane RUL [cykle]")
    ax.set_title("RUL: rzeczywiste vs przewidywane")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / f"{prefix}_rul_scatter.png")

    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(err, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Błąd predykcji [cykle]")
    ax.set_ylabel("Liczba próbek")
    ax.set_title("Rozkład błędu RUL")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / f"{prefix}_rul_histogram_blad.png")

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=[
            f"0: RUL > {critical_cycles}",
            f"1: RUL ≤ {critical_cycles}",
        ],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Macierz pomyłek (RUL ≤ {critical_cycles} cykli)")
    _save(fig, out_dir / f"{prefix}_macierz_pomylek.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(
        ["Poprawne", "Błędne"],
        [np.mean(yt == yp), 1 - np.mean(yt == yp)],
        color=["seagreen", "indianred"],
    )
    ax.set_ylabel("Udział próbek")
    ax.set_ylim(0, 1)
    ax.set_title(f"Dyskryminacja — dokładność {acc * 100:.1f}%")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir / f"{prefix}_dyskryminacja_slupki.png")


def _plot_weight_histogram(
    weights: np.ndarray,
    path: Path,
    title: str,
    dx: float = 0.05,
) -> None:
    """Histogram: ile połączeń ma wagę w przedziale [w, w+dx)."""
    w = weights.ravel()
    w_min = np.floor(w.min() / dx) * dx
    w_max = np.ceil(w.max() / dx) * dx
    if w_max <= w_min:
        w_max = w_min + dx
    bins = np.arange(w_min, w_max + dx, dx)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts, edges, _ = ax.hist(w, bins=bins, edgecolor="white", color="steelblue")
    ax.set_xlabel("Wartość wagi")
    ax.set_ylabel("Liczba połączeń")
    ax.set_title(f"{title}\n(szerokość przedziału Δ={dx})")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)


def plot_weights(model: LSTMRUL, out_dir: Path, hist_dx: float = 0.05) -> None:
    for layer_i in range(model.num_layers):
        w_ih = getattr(model.lstm, f"weight_ih_l{layer_i}").detach().cpu().numpy()
        w_hh = getattr(model.lstm, f"weight_hh_l{layer_i}").detach().cpu().numpy()
        w_all = np.concatenate([w_ih.ravel(), w_hh.ravel()])
        _plot_weight_histogram(
            w_all,
            out_dir / f"wagi_hist_lstm_{layer_i + 1}.png",
            f"Histogram wag LSTM — warstwa {layer_i + 1}",
            dx=hist_dx,
        )

        w = w_ih
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(w, aspect="auto", cmap="RdBu_r")
        ax.set_title(f"Wagi LSTM — warstwa {layer_i + 1}")
        ax.set_xlabel("Wejście")
        ax.set_ylabel("Ukryty")
        plt.colorbar(im, ax=ax, fraction=0.046)
        _save(fig, out_dir / f"wagi_lstm_{layer_i + 1}.png")

    w_fc1 = model.head[0].weight.detach().cpu().numpy()
    _plot_weight_histogram(
        w_fc1,
        out_dir / "wagi_hist_fc_1.png",
        "Histogram wag FC — warstwa 1",
        dx=hist_dx,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(w_fc1, aspect="auto", cmap="RdBu_r")
    ax.set_title("Wagi FC — warstwa 1 (hidden → 32)")
    ax.set_xlabel("Neuron wyjściowy")
    ax.set_ylabel("Neuron wejściowy")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_dir / "wagi_fc_1.png")

    w_fc2 = model.head[2].weight.detach().cpu().numpy()
    _plot_weight_histogram(
        w_fc2,
        out_dir / "wagi_hist_fc_2.png",
        "Histogram wag FC — warstwa 2",
        dx=hist_dx,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(w_fc2, aspect="auto", cmap="RdBu_r")
    ax.set_title("Wagi FC — warstwa 2 (32 → RUL)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_dir / "wagi_fc_2.png")

    all_w, labels = [], []
    for layer_i in range(model.num_layers):
        all_w.append(getattr(model.lstm, f"weight_ih_l{layer_i}").detach().cpu().numpy().ravel())
        labels.append(f"LSTM {layer_i + 1}")
    all_w.append(model.head[0].weight.detach().cpu().numpy().ravel())
    labels.append("FC 1")
    all_w.append(model.head[2].weight.detach().cpu().numpy().ravel())
    labels.append("FC 2")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(all_w, tick_labels=labels)
    ax.set_ylabel("Wartość wagi")
    ax.set_title("Rozkład wag w warstwach")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir / "wagi_rozklad_boxplot.png")
