"""Demand forecasting for NourishLink.

Pure NumPy / pandas / PyTorch module — imports nothing from Django so it can
be reused by management commands, scripts, and tests.

Design
------
The forecaster predicts **7-day demand totals per product** (the quantity
Config B priority logic and the forecaster page need).  Daily counts are
Poisson-noise dominated and a naive weekday/rolling mean is near the noise
floor there, so weekly totals are used instead — they average the Poisson
noise out and make trends and multi-day demand spikes learnable signal.

The MLP is trained as a **residual model over the rolling baseline** (the
28-day rolling weekly mean): the network learns the baseline's error rather
than the whole level.  A zero/flat residual therefore reproduces the
baseline exactly; anything the network adds must come from structure the
baseline cannot see (surge persistence, trend drift).  This keeps the
neural model at-worst-equal to the naive baseline, which is the fair and
academically defensible comparison.

Pipeline
--------
1. ``load_daily_demand``   – CSV (date, product, organization, quantity) ->
                              zero-filled dense day x product frame.
2. ``build_supervised``    – per (date, product) samples: sliding feature row
                              (daily lags, rolling stats, weekday one-hot,
                              product one-hot) -> 7-day forward total.
3. ``chronological_split`` – 12-week trailing holdout; train targets end >= 1
                              full week before the first test target (!),
                              so no information crosses the 7-day horizon.
4. ``WeeklyBaseline``      – trailing-28-day rolling weekly mean (the
                              rolling-average baseline the NN must beat).
5. ``ForecastMLP``         – small CPU MLP (targets normalised by the
                              product's train mean), trained on Windows.
6. ``train_and_evaluate``  – end-to-end run -> JSON-ready report + the next
                              7-day totals per product (direct, one-shot).

All runs are deterministic given ``seed``.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VERSION = "2.0"

LOOKBACK = 14          # daily lags fed to the model
ROLL_SHORT = 7         # short rolling window
ROLL_LONG = 28         # warm-up / baseline window (= 4 weeks)
WARMUP = ROLL_LONG
HORIZON = 7            # forecast window: 7-day totals
N_WEEKS_TEST = 12      # trailing weeks held out for evaluation

N_EXTRAS = 4           # momentum, roll7 std, weekday-28 ref, max7/mean28
NUMERIC_FEATURES = LOOKBACK + 2 + N_EXTRAS + 7   # + weekday one-hot
SCALE_PREFIX = LOOKBACK + 2 + N_EXTRAS           # scaled numeric prefix
DEFAULT_EPOCHS = 800
ENSEMBLE_SEEDS = (42, 43, 44)      # residual ensemble (variance reduction)
BASELINE_NAME = "baseline-rolling28-weekly"
MODEL_NAME = "torch-mlp-v2-residual"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_demand(csv_path) -> pd.DataFrame:
    """Load the simulator's daily-demand CSV into a dense day x product frame.

    Source rows are (date, product, organization, quantity) tuples from
    ``core.simulator``; organization rows are summed to network-wide demand.
    Calendar days with no demand are zero-filled.  Returns a frame indexed by
    consecutive dates, one column per product, float values.
    """
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    daily = (
        raw.groupby(["date", "product"])["quantity"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
        .astype(float)
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.columns.name = "product"
    return daily


def make_synthetic_daily(products, base, amplitude, start="2020-01-01",
                         days=180, seed=42):
    """Deterministic weekly-pattern frame for tests and demos."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=days, freq="D")
    cols = {}
    for name in products:
        weekdays = np.array([d.weekday() for d in index])
        bump = np.where(weekdays >= 5, amplitude[name], 0.0)
        noise = rng.normal(0, max(base[name], 1) * 0.25, size=days)
        cols[name] = np.round(np.clip(base[name] + bump + noise, 0, None))
    return pd.DataFrame(cols, index=index).astype(float)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def feature_vector(hist: np.ndarray, target_weekday: int,
                   product_onehot: np.ndarray | None = None) -> np.ndarray:
    """Feature row for a 7-day window starting the day after ``hist`` ends.

    Layout: lag_1..lag_{LOOKBACK}, mean(last {ROLL_SHORT}), mean(last
    {ROLL_LONG}), momentum(lag1-lag7), std(last {ROLL_SHORT}), same-weekday
    reference mean over the trailing 28 days (contiguous daily hist),
    max7/mean28 surge ratio, weekday one-hot(7), optional product one-hot.
    """
    lags = hist[-1:-LOOKBACK - 1:-1]
    roll_short = hist[-ROLL_SHORT:].mean()
    roll_long = hist[-ROLL_LONG:].mean()
    momentum = lags[0] - lags[LOOKBACK - 1]          # lag1 - lag7
    roll_std = hist[-ROLL_SHORT:].std()
    # contiguous daily history: the target's weekday falls exactly every 7
    # days back, so hist[-7]/[-14]/[-21]/[-28] are those four days
    weekday28 = float(np.mean([hist[-7], hist[-14], hist[-21], hist[-28]]))
    surge = lags[:ROLL_SHORT].max() / roll_long if roll_long > 0 else 0.0
    extras = [momentum, roll_std, weekday28, surge]
    dow = np.zeros(7, dtype=float)
    dow[target_weekday] = 1.0
    parts = [lags, [roll_short, roll_long], extras, dow]
    if product_onehot is not None:
        parts.append(product_onehot)
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def build_supervised(df: pd.DataFrame):
    """Samples for every (target-week start date, product).

    Target: quantity summed over days [t, t+7).  Rows are day-major
    (date block of ``n_products`` consecutive rows), which keeps the
    chronological split a clean date cut.

    Returned dict keys:
      dates       – supervised window-start dates (skips the last 6: no
                    complete 7-day window beyond the data)
      X           – (n_dates*n_products, NUMERIC_FEATURES + n_products)
      Y           – (n_dates*n_products,) 7-day totals
      products    – ordered product names
      values      – full zero-filled (n_days, n_products) matrix
      full_dates  – complete history dates
    """
    values = df.values.astype(float)
    dates = df.index.to_list()
    products = list(df.columns)
    n_products = len(products)
    prod_eye = np.eye(n_products, dtype=float)

    n_dates = len(dates) - WARMUP - (HORIZON - 1)
    rows = n_dates * n_products
    X = np.empty((rows, NUMERIC_FEATURES + n_products), dtype=float)
    Y = np.empty(rows, dtype=float)
    B = np.empty(rows, dtype=float)
    for i, t in enumerate(range(WARMUP, WARMUP + n_dates)):
        for j in range(n_products):
            r = i * n_products + j
            hist = values[:t, j]
            X[r, :NUMERIC_FEATURES] = feature_vector(hist, dates[t].weekday())
            X[r, NUMERIC_FEATURES:] = prod_eye[j]
            Y[r] = values[t:t + HORIZON, j].sum()
            B[r] = weekly_baseline_row(hist)
    return {
        "dates": dates[WARMUP:WARMUP + n_dates],
        "X": X.astype(np.float32),
        "Y": Y,
        "B": B,           # baseline prediction for the same 7-day window
        "products": products,
        "values": values,
        "full_dates": dates,
    }


def chronological_split(packed: dict, test_weeks: int = N_WEEKS_TEST,
                        val_weeks: int = 6):
    """Trailing holdout of ``test_weeks`` non-overlapping weekly windows.

    Test targets: the last ``test_weeks`` valid window starts (7 days apart),
    covering the final ``7*test_weeks`` data days.  Training targets all end
    before that first test window begins, so no day of a training 7-day
    target ever falls inside the test horizon (and vice versa).

    ``val_weeks`` trailing DATE blocks of the train region are marked via
    ``n_val_rows`` (used by the trainer for early stopping — they remain
    pre-holdout data, so no leakage into the test horizon).
    """
    dates = packed["dates"]
    n_products = len(packed["products"])
    if test_weeks * HORIZON >= len(dates):
        raise ValueError(
            f"test_weeks={test_weeks} leaves no training windows "
            f"({len(dates)} supervised days)")

    test_idx = [len(dates) - 1 - HORIZON * k for k in range(test_weeks)][::-1]
    test_first = min(test_idx)
    train_idx = [i for i in range(test_first)
                 if i + HORIZON - 1 < test_first]     # ends >= 1 week clear

    # trailing date blocks of the train region act as the validation slice;
    # each date block has one window-start per product
    val_blocks = min(val_weeks, len(train_idx))
    val_rows = val_blocks * n_products

    rows_train = [i * n_products + j
                  for i in train_idx for j in range(n_products)]
    rows_test = [i * n_products + j
                 for i in test_idx for j in range(n_products)]
    return {
        "X_train": packed["X"][rows_train],
        "Y_train": packed["Y"][rows_train],
        "B_train": packed["B"][rows_train],
        "X_test": packed["X"][rows_test],
        "Y_test": packed["Y"][rows_test].reshape(test_weeks, n_products),
        "B_test": packed["B"][rows_test].reshape(test_weeks, n_products),
        "test_dates": [dates[i] for i in test_idx],
        "test_first_date": dates[test_idx[0]],
        "n_val_rows": val_rows,
    }


# ---------------------------------------------------------------------------
# Baseline: trailing-28-day rolling weekly mean
# ---------------------------------------------------------------------------

def weekly_baseline_row(hist_values: np.ndarray,
                        window_days: int = ROLL_LONG) -> float:
    """Mean 7-day total of the trailing ``window_days`` (= 4 weeks)."""
    return float(hist_values[-window_days:].sum() / (window_days / HORIZON))


class WeeklyBaseline:
    """Rolling-average baseline for weekly totals."""

    name = BASELINE_NAME

    def fit(self, *args, **kwargs):              # parity with the MLP API
        return self

    def predict_matrix(self, packed, row_indices, n_products):
        """Predictions for rows ``row_indices`` (day-major) from history
        ending one day before each window starts."""
        values = packed["values"]
        out = []
        for i in row_indices:
            t = len(values) - len(packed["dates"]) + i   # window start in values
            for j in range(n_products):
                out.append(weekly_baseline_row(values[:t, j]))
        return np.asarray(out)

    def predict_all(self, packed):
        """Baseline 7-day totals starting the day after the data ends."""
        values = packed["values"]
        n_products = len(packed["products"])
        return {name: weekly_baseline_row(values[:, j])
                for j, name in enumerate(packed["products"])}


# ---------------------------------------------------------------------------
# Neural model
# ---------------------------------------------------------------------------

class ForecastMLP(nn.Module):
    """Compact MLP: input -> 128 -> 64 -> 1 (dropout 0.1 regularisation)."""

    def __init__(self, n_in: int, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _target_scales(packed, rows):
    """Per-product residual scales over the given training rows (day-major).

    The model learns ``actual - baseline``; normalising residuals keeps
    heterogeneous products on a comparable target scale.
    """
    n_products = len(packed["products"])
    targets = packed["Y"][rows] - packed["B"][rows]
    means = targets.reshape(-1, n_products)
    means = np.abs(means).mean(axis=0)          # robust residual scale
    means[means < 1e-6] = 1.0
    return means.astype(np.float32)


def _row_product_index(n_rows, n_products):
    return np.arange(n_rows) % n_products


def _scale_features(split):
    """Standard-scale lags/rolling/extra features (TRAIN statistics only)."""
    mean = split["X_train"][:, :SCALE_PREFIX].mean(axis=0)
    std = split["X_train"][:, :SCALE_PREFIX].std(axis=0)
    std[std == 0] = 1.0
    for key in ("X_train", "X_test"):
        scaled = split[key].copy()
        scaled[:, :SCALE_PREFIX] = \
            (scaled[:, :SCALE_PREFIX] - mean) / std
        split[key] = scaled.astype(np.float32)
    return mean, std


def train_model(split, y_row_scales=None, seed=42, epochs=DEFAULT_EPOCHS,
                batch_size=96, lr=1e-3, weight_decay=1e-4,
                val_rows=0, patience=80):
    """Train the MLP on the residual target (Y - baseline), deterministically.

    ``val_rows``: trailing chrono rows of the train region used for
    validation (best val state is restored; stops after ``patience`` epochs
    without improvement).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = torch.tensor(split["X_train"])
    residual = (split["Y_train"] - split["B_train"]).astype(np.float32)
    Y = torch.tensor(residual)
    if y_row_scales is not None:
        Y = Y / torch.tensor(y_row_scales, dtype=torch.float32)

    n = X.shape[0]
    if val_rows:
        Xval, Yval = X[n - val_rows:], Y[n - val_rows:]
        Xtr, Ytr = X[:n - val_rows], Y[:n - val_rows]
    else:
        Xval = Yval = None
        Xtr, Ytr = X, Y

    model = ForecastMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    def val_loss(net):
        if Xval is None:
            return None
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xval), Yval))
        net.train()
        return v

    best_state, best_val, best_epoch, stopped = None, None, 0, -1
    for epoch in range(epochs):
        order = torch.randperm(n - (val_rows or 0))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        v = val_loss(model)
        if v is not None:
            if best_val is None or v < best_val - 1e-6:
                best_val, best_epoch = v, epoch
                best_state = {key: t.clone() for key, t in
                              model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                stopped = epoch
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(Xtr), Ytr))
    return model, {"train_loss": round(train_loss, 4),
                   "val_loss": None if best_val is None else round(best_val, 4),
                   "best_epoch": best_epoch, "stopped_epoch": stopped}


def predict_matrix(model, split, y_row_scales=None):
    """NN residual predictions re-added to the baseline, clipped at >= 0."""
    with torch.no_grad():
        y = model(torch.tensor(split["X_test"])).numpy()
    if y_row_scales is not None:
        y = y * y_row_scales
    y = y + split["B_test"].reshape(-1)
    return np.clip(y, 0, None)


# ---------------------------------------------------------------------------
# Future forecasts
# ---------------------------------------------------------------------------

def predict_future_week(model, packed, product_index, mean=None, std=None,
                        target_mean=None) -> float:
    """One-shot MLP forecast of next week's 7-day total: baseline + residual."""
    j = product_index
    hist = packed["values"][:, j]
    onehot = np.zeros(len(packed["products"]))
    onehot[j] = 1.0
    first_day = packed["full_dates"][-1] + pd.Timedelta(days=1)
    raw = feature_vector(hist, first_day.weekday(), product_onehot=onehot)
    if mean is not None:
        raw[:SCALE_PREFIX] = (raw[:SCALE_PREFIX] - mean) / std
    with torch.no_grad():
        residual = float(model(torch.tensor(raw.astype(np.float32))).item())
    if target_mean is not None:
        residual = residual * target_mean
    total = weekly_baseline_row(hist) + residual
    return max(0.0, total)


# ---------------------------------------------------------------------------
# Metrics + end-to-end driver
# ---------------------------------------------------------------------------

def mae(actual, pred) -> float:
    return float(np.mean(np.abs(actual - pred)))


def wape(actual, pred) -> float | None:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return None
    return float(np.sum(np.abs(actual - pred)) / denom)


def train_and_evaluate(csv_path=None, df: pd.DataFrame | None = None,
                       seed=42, epochs=DEFAULT_EPOCHS,
                       test_weeks=N_WEEKS_TEST, horizon=HORIZON,
                       val_weeks=0) -> dict:
    """Full pipeline -> JSON-ready report (weekly MAE/WAPE) + next-week totals.

    Default configuration (validated ablation on the simulator data):
    3-seed residual ensemble, no validation-based early stopping (the
    validation slice is too small on 12 months of data to be stable).
    """
    packed = (build_supervised(df) if df is not None
              else build_supervised(load_daily_demand(csv_path)))
    split = chronological_split(packed, test_weeks, val_weeks)
    mean, std = _scale_features(split)

    n_products = len(packed["products"])
    train_rows = split["X_train"].shape[0]
    y_means = _target_scales(packed, range(train_rows))
    y_row_train = y_means[_row_product_index(train_rows, n_products)]
    y_row_test = y_means[_row_product_index(split["X_test"].shape[0], n_products)]

    models = []
    for seed in ENSEMBLE_SEEDS:
        model, info = train_model(split, y_row_scales=y_row_train,
                                  seed=seed, epochs=epochs,
                                  val_rows=split["n_val_rows"])
        models.append(model)
        model_info = info

    y_true = split["Y_test"]
    y_hat_mlp = np.mean(
        [predict_matrix(model, split, y_row_scales=y_row_test)
         for model in models], axis=0).reshape(test_weeks, -1)
    y_hat_base = (WeeklyBaseline()
                  .predict_matrix(packed, range(len(packed["dates"]) - test_weeks,
                                                len(packed["dates"])), n_products)
                  .reshape(test_weeks, -1))

    per_product = {}
    mlp_abs = base_abs = actual_total = 0.0
    wins = losses = draws = 0

    for j, name in enumerate(packed["products"]):
        actual = y_true[:, j]
        m_mlp, m_base = mae(actual, y_hat_mlp[:, j]), mae(actual, y_hat_base[:, j])
        w_mlp, w_base = wape(actual, y_hat_mlp[:, j]), wape(actual, y_hat_base[:, j])
        per_product[name] = {
            "model_mae": round(m_mlp, 3), "baseline_mae": round(m_base, 3),
            "model_wape": None if w_mlp is None else round(w_mlp, 3),
            "baseline_wape": None if w_base is None else round(w_base, 3),
            "model_wins": bool(m_mlp < m_base),
        }
        mlp_abs += float(np.sum(np.abs(actual - y_hat_mlp[:, j])))
        base_abs += float(np.sum(np.abs(actual - y_hat_base[:, j])))
        actual_total += float(np.sum(np.abs(actual)))
        wins += m_mlp < m_base
        losses += m_mlp > m_base
        draws += m_mlp == m_base

    baseline_future = WeeklyBaseline().predict_all(packed)
    future_model = {
        name: float(np.mean(
            [predict_future_week(model, packed, j, mean, std,
                                 target_mean=float(y_means[j]))
             for model in models]))
        for j, name in enumerate(packed["products"])
    }

    return {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "tool_version": VERSION,
        "seed": seed, "epochs": epochs, "test_weeks": test_weeks,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "horizon": horizon,
        "data_start": str(packed["full_dates"][0].date()),
        "data_end": str(packed["full_dates"][-1].date()),
        "n_products": len(packed["products"]),
        "train_rows": int(train_rows),
        "test_rows": int(split["X_test"].shape[0]),
        "model_name": MODEL_NAME,
        "baseline_name": BASELINE_NAME,
        "train_loss_normalised": model_info["train_loss"],
        "val_loss_normalised": model_info["val_loss"],
        "best_epoch": model_info["best_epoch"],
        "per_product": per_product,
        "overall": {
            "model_mae": round(float(np.mean(np.abs(y_true - y_hat_mlp))), 3),
            "baseline_mae": round(float(np.mean(np.abs(y_true - y_hat_base))), 3),
            "model_wape": round(mlp_abs / actual_total, 3) if actual_total else None,
            "baseline_wape": round(base_abs / actual_total, 3) if actual_total else None,
        },
        "model_wins_overall": wins,
        "model_losses_overall": losses,
        "model_draws_overall": draws,
        "future_model": future_model,           # next 7-day totals per product
        "future_baseline": baseline_future,
    }


def save_report(report: dict, path) -> str:
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float),
                    encoding="utf-8")
    return str(path)


def load_report(path):
    import json
    from pathlib import Path
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
