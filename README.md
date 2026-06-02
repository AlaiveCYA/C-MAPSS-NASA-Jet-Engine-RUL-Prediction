# CMAPSS — Remaining Useful Life

Predict how many operational cycles remain before jet engine failure using NASA CMAPSS sensor data.

Two **independent** models share `config.yaml` and preprocessing (drop constant sensors, MinMax scale):

| Script | Model | Output |
|--------|-------|--------|
| `train.py` | LSTM | `results/FD00x/` |
| `train_rf.py` | Random Forest | `results/FD00x/rf/` |

## Problem

Each engine is a time series (one row per **cycle**). Training engines run to failure; test engines stop early. Predict **RUL** at the **last cycle** of each test engine and compare to `RUL_FD00x.txt`.

| Subset | Difficulty |
|--------|------------|
| **FD001** | Easiest (start here) |
| FD002–FD004 | More conditions / fault types |

## Layout

```
dataset/          # train_*, test_*, RUL_*, readme.txt
config.yaml
train.py          # LSTM
train_rf.py       # las losowy (osobny pipeline)
src/
  data.py         # load, drop constant sensors, MinMax scale, windows
  rf_features.py  # cechy tabularne dla RF
  model.py        # LSTM
  metrics.py      # RMSE, NASA score
results/          # created when training
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train — LSTM

```bash
python train.py
python train.py --epochs 50 --stride 5   # faster on CPU
```

## Train — las losowy (Random Forest)

Osobny skrypt, ten sam `config.yaml` (w tym `variance_threshold: 0.01` → usuwanie stałych sensorów).

```bash
python train_rf.py
python train_rf.py --n-estimators 400 --stride 5
```

Wyniki: `results/FD00x/rf/model.pkl`, `metrics.json`, wykresy RUL w `rf/wykresy/`.

Cechy RF: ze okna 30 cykli liczone są **średnia, odch. std. i ostatni cykl** per sensor → wektor tabularny.

Edit `dataset_id` in `config.yaml` for FD002–FD004.

## Pipeline (LSTM)

1. Load data with standard column names (`unit_id`, `time_cycles`, sensors)
2. Drop sensors with train variance &lt; 0.01
3. `MinMaxScaler` on sensors (fit train, apply test)
4. Sliding windows of 30 cycles (zero-pad if shorter)
5. LSTM → RUL in cycles

## Wykresy (po treningu)

Każdy wykres **osobny plik PNG** w `results/FD00x/wykresy/` (pod LaTeX):

| Plik | Opis |
|------|------|
| `uczenie_mse_epoki.png` | MSE: batch / cały train / walidacja |
| `uczenie_dokladnosc_dyskryminacji_epoki.png` | Dokładność dyskryminacji RUL≤30 w epokach |
| `mse_warstwy.png` | MSE per warstwa (próbka vs cały train) |
| `walidacja_rul_scatter.png` | RUL prawda vs predykcja |
| `walidacja_rul_histogram_blad.png` | Histogram błędu |
| `walidacja_macierz_pomylek.png` | Macierz pomyłek |
| `walidacja_dyskryminacja_slupki.png` | Poprawne / błędne |
| `test_*.png` | To samo dla zbioru testowego (4 pliki) |
| `wagi_lstm_1.png`, `wagi_lstm_2.png` | Mapy ciepła wag LSTM |
| `wagi_fc_1.png`, `wagi_fc_2.png` | Wagi warstw liniowych |
| `wagi_hist_lstm_1.png`, `wagi_hist_lstm_2.png` | Histogram wag LSTM (Δ=0.05) |
| `wagi_hist_fc_1.png`, `wagi_hist_fc_2.png` | Histogram wag FC (Δ=0.05) |
| `wagi_rozklad_boxplot.png` | Boxplot wag |

Przykład LaTeX: `\includegraphics[width=0.8\linewidth]{wykresy/walidacja_rul_scatter.png}`

Dyskryminacja: klasa **1** = `RUL ≤ 30` (`rul_critical_cycles` w `config.yaml`).

## References

- `dataset/readme.txt`
- Saxena et al., PHM08
# C-MAPSS-NASA-Jet-Engine-RUL-Prediction
