"""
estimation/models.py — Cell-level / Hybrid regressors (XGBoost + PyTorch LSTM).

Off-the-shelf applied models. Hyper-parameters and the compute device are injected
(from ModelCfg / config.device) rather than imported as globals.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_xgb_reg(X_tr, y_tr, xgb_params: dict, seed: int) -> XGBRegressor:
    m = XGBRegressor(**xgb_params, random_state=seed)
    m.fit(X_tr, y_tr)
    return m


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, dense: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, dense), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def make_seq(X: np.ndarray, y: np.ndarray, sl: int):
    Xs, ys = [], []
    for i in range(sl, len(X)):
        Xs.append(X[i - sl:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _train_pytorch_lstm(model, Xtr, ytr, Xval, yval, device, epochs, batch_size, lr, patience):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # from_numpy shares the buffer (torch.tensor always copies); the arrays are
    # already float32 and C-contiguous, so this is a no-op numerically.
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.float32)).view(-1, 1)
    Xval_t = torch.from_numpy(np.ascontiguousarray(Xval, dtype=np.float32)).to(device)
    yval_t = torch.from_numpy(
        np.ascontiguousarray(yval, dtype=np.float32)).view(-1, 1).to(device)
    n = len(Xtr_t)
    best_val = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t)
            val_mae = (vp - yval_t).abs().mean().item()
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_lstm_model(X_all, y_all, split_idx, seq_len, seed, device, lstm_cfg: dict):
    """Train LSTM on the train portion, predict on the test portion.

    Test sequences are built for EVERY test index using the preceding seq_len steps
    (which may include the last train-history rows), so the LSTM is evaluated on the
    exact same test targets y_all[split_idx:] as the XGB models. Returns
    (preds, targets) aligned to indices [split_idx, len(X_all))."""
    torch.manual_seed(seed); np.random.seed(seed)
    if split_idx < seq_len:
        return np.array([]), np.array([])
    sc = StandardScaler()
    sc.fit(X_all[:split_idx])
    # float32 here, not inside torch.tensor(): identical values (the same
    # float64 -> float32 rounding, just applied earlier) at half the RAM for
    # the dense (N, seq_len, F) sequence copies built below.
    X_scaled = sc.transform(X_all).astype(np.float32, copy=False)
    Xtr_seq, ytr_seq = make_seq(X_scaled[:split_idx], y_all[:split_idx], seq_len)
    if len(Xtr_seq) < 10:
        return np.array([]), np.array([])
    Xte_seq = np.stack([X_scaled[i - seq_len:i] for i in range(split_idx, len(X_scaled))])
    yte_seq = np.asarray(y_all[split_idx:])
    n_tr = int(len(Xtr_seq) * 0.8)
    model = LSTMRegressor(n_features=Xtr_seq.shape[2], hidden=lstm_cfg["hidden"],
                          dense=lstm_cfg["dense"], dropout=lstm_cfg["dropout"])
    model = _train_pytorch_lstm(
        model, Xtr_seq[:n_tr], ytr_seq[:n_tr], Xtr_seq[n_tr:], ytr_seq[n_tr:],
        device=device, epochs=lstm_cfg["epochs"], batch_size=lstm_cfg["batch"],
        lr=lstm_cfg["lr"], patience=lstm_cfg["patience"],
    )
    model.eval()
    # Chunked forward pass: eval mode + an LSTM means every row is independent,
    # so this is numerically identical to one big batch, but the peak device
    # tensor is bounded instead of scaling with the whole test split.
    chunk = 8192
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xte_seq), chunk):
            xb = torch.from_numpy(
                np.ascontiguousarray(Xte_seq[i:i + chunk], dtype=np.float32)).to(device)
            outs.append(model(xb).cpu().numpy())
    pred = (np.concatenate(outs).flatten() if outs else np.array([]))
    return pred, yte_seq.flatten()
