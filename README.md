# CMAPSS — Remaining Useful Life (RUL) Prediction

Predict how many operational cycles remain before jet engine failure using the NASA **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset.

Two **independent** pipelines share `config.yaml` and the same preprocessing (drop near-constant sensors, MinMax scaling):

| Script | Model | Output directory |
|--------|-------|------------------|
| `train.py` | LSTM | `results/<FD00x>/lstm/` |
| `train_rf.py` | Random Forest | `results/<FD00x>/rf/` |

## Problem

Each engine is a multivariate time series (one row per **cycle**). Training engines run until failure; test engines stop before failure. The task is to predict **RUL** (remaining useful life in cycles) at the **last observed cycle** of each test engine and compare against `RUL_FD00x.txt`.

| Subset | Training engines | Test engines | Notes |
|--------|------------------|--------------|--------|
| **FD001** | 100 | 100 | 1 operating condition, 1 fault — easiest |
| FD002 | 260 | 259 | 6 conditions, 1 fault |
| FD003 | 100 | 100 | 1 condition, 2 fault modes |
| FD004 | 248 | 249 | 6 conditions, 2 fault modes — hardest |

## Project layout

```
dataset/              # train_*, test_*, RUL_*, readme.txt
config.yaml           # shared experiment settings
train.py              # LSTM training & evaluation
train_rf.py           # Random Forest training & evaluation
requirements.txt
src/
  data.py             # load, sensor filtering, MinMax scale, sliding windows
  rf_features.py      # tabular features for Random Forest
  model.py            # LSTM architecture
  metrics.py          # RMSE, NASA PHM08 score
  plots.py            # evaluation plots (one PNG per figure)
results/              # created after training
  FD001/
    lstm/             # model.pt, bundle.pkl, metrics.json, wykresy/
    rf/               # model.pkl, metrics.json, wykresy/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (`config.yaml`)

| Key | Description |
|-----|-------------|
| `dataset_id` | Subset to use (`FD001` … `FD004`) |
| `data_dir` | Path to raw data (`dataset`) |
| `max_rul` | Piecewise RUL cap (labels clipped at this value) |
| `window_size` | Sliding window length (default: 30 cycles) |
| `variance_threshold` | Drop sensors with training variance below this (default: 0.01) |
| `train_degradation_only` | Skip training rows where RUL equals `max_rul` |
| `val_ratio` | Fraction of engines held out for validation (by `unit_id`) |
| `rul_critical_cycles` | Binary discrimination threshold: class 1 if RUL ≤ this value (default: 30) |
| `random_seed` | Reproducibility seed |
| `weight_hist_dx` | Bin width for weight histogram plots (default: 0.05) |

**Note:** `RUL_FD00x.txt` is used only for **final test evaluation**, not for validation during training.

## Train — LSTM

```bash
python train.py
python train.py --epochs 100 --batch-size 256 --stride 1
python train.py --epochs 50 --stride 5    # faster on CPU (fewer window samples)
```

| CLI argument | Default | Description |
|--------------|---------|-------------|
| `--epochs` | 40 | Training epochs |
| `--batch-size` | 256 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--hidden` | 64 | LSTM hidden size |
| `--layers` | 2 | Number of LSTM layers |
| `--stride` | 1 | Sample every N-th cycle when building windows |

**Outputs:** `results/<FD00x>/lstm/model.pt`, `bundle.pkl`, `metrics.json`, plots in `lstm/wykresy/`.

## Train — Random Forest

Separate script; same preprocessing and `variance_threshold` as LSTM.

```bash
python train_rf.py
python train_rf.py --n-estimators 400 --max-depth 28 --stride 1
```

| CLI argument | Default | Description |
|--------------|---------|-------------|
| `--n-estimators` | 300 | Number of trees |
| `--max-depth` | 28 | Maximum tree depth |
| `--stride` | 1 | Sample every N-th cycle when building training rows |

**RF features:** for each 30-cycle window, per sensor: **mean**, **std**, and **last-cycle value** → flat feature vector (length `3 × n_sensors`).

**Outputs:** `results/<FD00x>/rf/model.pkl`, `metrics.json`, plots in `rf/wykresy/`.

## Pipeline (shared)

1. Load space-separated files with column names (`unit_id`, `time_cycles`, settings, `sensor_1` … `sensor_21`).
2. Remove sensors with training variance &lt; `variance_threshold`.
3. Fit `MinMaxScaler` on training sensors only; transform test data with the same scaler.
4. Build sliding windows (zero-pad on the left if history &lt; 30 cycles).
5. Label RUL on training data: `min(max_cycle - current_cycle, max_rul)`.
6. Split engines (`unit_id`) into train / validation (no cycle leakage across splits).
7. **Test:** one prediction per test engine at its last cycle; compare to `RUL_FD00x.txt`.

## LSTM architecture

- 2-layer LSTM (hidden size 64, dropout 0.2 between layers)
- Head: Linear(64→32) → ReLU → Linear(32→1)
- Loss: MSE; optimizer: Adam
- Predictions clipped at 0 (no negative RUL)

## Plots (after training)

Each figure is saved as a **separate PNG** under `lstm/wykresy/` or `rf/wykresy/` (suitable for LaTeX `\includegraphics`).

| File | Description |
|------|-------------|
| `uczenie_mse_epoki.png` | *(LSTM only)* MSE vs epoch (batch / full train / validation) |
| `uczenie_dokladnosc_dyskryminacji_epoki.png` | *(LSTM only)* Discrimination accuracy vs epoch |
| `mse_warstwy.png` | *(LSTM only)* MSE per layer (sample vs full train) |
| `walidacja_rul_scatter.png` | True vs predicted RUL (validation windows) |
| `walidacja_rul_histogram_blad.png` | Prediction error histogram (validation) |
| `walidacja_macierz_pomylek.png` | Confusion matrix (RUL ≤ 30 vs &gt; 30) |
| `walidacja_dyskryminacja_slupki.png` | Correct / incorrect discrimination share |
| `test_*.png` | Same four plots for the test set (one point per engine) |
| `wagi_lstm_*.png`, `wagi_fc_*.png` | *(LSTM only)* Weight heatmaps |
| `wagi_hist_*.png` | *(LSTM only)* Weight histograms (bin width from `weight_hist_dx`) |
| `wagi_rozklad_boxplot.png` | *(LSTM only)* Weight distribution by layer |

**Binary discrimination (auxiliary):** class **1** = RUL ≤ `rul_critical_cycles` (default 30); class **0** = above threshold. Applied to both ground truth and predictions for reporting only; the model still predicts continuous RUL.

## Metrics

- **RMSE** — root mean squared error in cycles (main test metric: one value per test engine).
- **NASA score** — asymmetric PHM08 cost (lower is better; penalizes over-optimistic RUL more strongly).
- **Discrimination accuracy** — share of correct binary labels at the RUL ≤ 30 threshold.

## References

- Dataset description: `dataset/readme.txt`
- Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, 2008
