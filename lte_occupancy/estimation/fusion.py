"""
estimation/fusion.py — Stage 2/3 of the tuned protocol: 2-stage hybrid.

Base branches are trained independently; a compact FUSION model learns *what to
trust when*, not the raw features again. The protocol is now SYMMETRIC by model
family:

    Cell-XGB  + PerUE-XGB   -> Hybrid-XGB    (shallow conditional corrector)
    Cell-LSTM + PerUE-LSTM  -> Hybrid-LSTM   (causal window of fusion vectors)

Survival is removed from both per-UE branches (XGB-only / LSTM-only classifiers).

Leakage control (critical, identical for both families):
  - Hybrid TRAINING features are out-of-fold (OOF): base models never saw the label
    of the slot whose prediction feeds the hybrid. Expanding-window folds over the
    train region only; cell feature thresholds and per-UE context are REBUILT per
    fold with the fold's own train boundary.
  - Hybrid VALIDATION/TEST features come from full-train base models predicting on
    the (unseen) val/test slices.
  - No calibration is cross-fitted here: the hybrid input uses OOF *raw* base
    predictions (the hybrid learns the raw->count mapping itself).

Fusion feature vector z_t (compact, ~15) = the two base predictions, their
disagreement, and cheap observation-side confidence signals (counter volatility,
release/paging activity). Mode B uses observable signals only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .per_ue import UETrack, build_perue_xgb, compute_per_ue, make_cell_context_fn
from ..features.per_ue import per_ue_features_extended
from ..features.cell import build_cell_features
from ..event_order import event_key_with_pid

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                         # torch-free environments (XGB-only stages)
    _TORCH = False


# ======================================================================================
# per-UE count calibration (raw probability sum -> occupancy scale)
# ======================================================================================
# The per-UE branch sums P(present) over TRACKED tracks only, so it systematically
# UNDER-estimates occupancy by the untracked mass. The legacy pipeline corrected this with
# an isotonic map, but fitted it on the base model's own IN-SAMPLE training predictions
# (iso.fit(raw[:split], y[:split]) where raw came from a model trained on [:split]) — the
# map therefore adapts to optimistically small in-sample errors.
#
# Here the map is fitted on OUT-OF-FOLD base predictions instead (predictions the base
# model never saw the labels for) and applied to the held-out slice, which is the same
# leakage discipline the hybrid already uses. Whether calibration is adopted at all is an
# empirical question, decided by an ablation on held-out TEST seeds — not assumed.
CALIB_KINDS = ("none", "isotonic", "linear")


def _apply_finite(fn, v):
    """Apply fn to finite entries only; NaNs (e.g. LSTM warm-up) pass through."""
    v = np.asarray(v, np.float64)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    if m.any():
        out[m] = fn(v[m])
    return out


def fit_perue_calibrator(oof_perue, y, lo, hi, kind: str = "isotonic"):
    """Fit a monotone map raw_perue_count -> occupancy from OOF predictions on [lo, hi).

    kind="isotonic" is flexible (matches the legacy correction's shape freedom);
    kind="linear" is a 2-parameter map whose in-sample/out-of-sample gap is negligible,
    which is useful as a low-variance control in the ablation. Returns a callable; falls
    back to identity when there is not enough usable data."""
    if kind == "none":
        return lambda v: np.asarray(v, np.float64)
    x = np.asarray(oof_perue, np.float64)[lo:hi]
    t = np.asarray(y, np.float64)[lo:hi]
    m = np.isfinite(x) & np.isfinite(t)
    if m.sum() < 10 or np.ptp(x[m]) < 1e-9:
        return lambda v: np.asarray(v, np.float64)
    x, t = x[m], t[m]
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(x, t)
        return lambda v: _apply_finite(iso.predict, v)
    if kind == "linear":
        a, b = np.polyfit(x, t, 1)
        return lambda v: _apply_finite(lambda z: a * z + b, v)
    raise ValueError(f"unknown calibration kind {kind!r} (expected one of {CALIB_KINDS})")


# ======================================================================================
# expanding-window OOF base predictions over the TRAIN region [0, train_end)
# ======================================================================================
def _fold_bounds(train_end: int, n_folds: int):
    """Block boundaries 0 = b0 < b1 < ... < b_{n} = train_end."""
    return [int(round(train_end * i / n_folds)) for i in range(n_folds + 1)]


def oof_cell_xgb(df, mode, dsm, y, train_end, params, seed, n_folds=4):
    """OOF cell predictions on [b1, train_end). Cell features (whose thresholds are
    data-dependent) are REBUILT per fold with the fold's own train boundary, so an early
    fold never sees later-block statistics. Slots [0,b1) stay NaN (initial seed)."""
    b = _fold_bounds(train_end, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for i in range(1, n_folds):
        tr_end, pr_end = b[i], b[i + 1]
        feat = build_cell_features(df, mode, dsm["bs"], train_end=tr_end,
                                   wave_period=dsm["wave_period"],
                                   rolling_windows=dsm["rolling"], feat=dsm["feat"])
        X = feat.values.astype(np.float32)
        m = XGBRegressor(**params, random_state=seed)
        m.fit(X[:tr_end], y[:tr_end])
        oof[tr_end:pr_end] = m.predict(X[tr_end:pr_end])
    return oof


def oof_perue_xgb(df, mode, dsm, y, train_end, clf_params, seed, n_folds=4):
    b = _fold_bounds(train_end, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    ev, mt = dsm["events"], dsm["meta"]
    sub, prune = dsm["subsample"], dsm["prune"]
    for i in range(1, n_folds):
        tr_end, pr_end = b[i], b[i + 1]
        fold_ctx = make_cell_context_fn(df, mode=mode, train_end=tr_end,
                                        wave_period=dsm["wave_period"], feat=dsm["feat"])
        clf = build_perue_xgb(ev, mt, tr_end, seed=seed, xgb_clf_params=clf_params,
                              subsample=sub, prune_elapsed=prune, cell_context_fn=fold_ctx)
        _s, xgb_pred = compute_per_ue(ev, mt, lambda *a, **k: 0.0, {}, clf, len(y),
                                      subsample=sub, prune_elapsed=prune, cell_context_fn=fold_ctx)
        oof[tr_end:pr_end] = xgb_pred[tr_end:pr_end]
    return oof


def oof_cell_lstm(df, mode, dsm, y, train_end, params, seed, device, n_folds=4):
    """OOF Cell-LSTM predictions on [b1, train_end), symmetric to oof_cell_xgb. Uses the
    COMPACT Cell-LSTM feature set (raw counters + diff + short rolling mean/std). Those
    transforms carry no fitted threshold, so — unlike the XGB cell features — they need no
    per-fold rebuild; only the fold LSTM is refit on [0, tr_end) to predict [tr_end,
    pr_end). Slots [0,b1) stay NaN."""
    b = _fold_bounds(train_end, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    X = dsm["X_cell_lstm"]                     # compact, boundary-free
    for i in range(1, n_folds):
        tr_end, pr_end = b[i], b[i + 1]
        full, _ = train_lstm_regressor(
            Z_train=X, y_train=y, train_idx=range(0, tr_end),
            Z_eval=X, y_eval=y, eval_idx=range(tr_end, pr_end),
            device=device, seed=seed, train_forbid_boundaries=(),
            eval_forbid_boundaries=(), **_lstm_reg_kwargs(params))
        oof[tr_end:pr_end] = full[tr_end:pr_end]
    return oof


# ======================================================================================
# compact fusion features (shared by both hybrid families)
# ======================================================================================
def build_fusion_features(cell_pred, perue_pred, df, mode: str, windows=(10, 60)) -> pd.DataFrame:
    """~15 compact features: the two predictions, their disagreement, and cheap
    observation-side confidence signals available to the observer."""
    cp = np.asarray(cell_pred, dtype=np.float64)
    pp = np.asarray(perue_pred, dtype=np.float64)
    f = pd.DataFrame(index=np.arange(len(cp)))
    f["cell_pred"] = cp
    f["perue_pred"] = pp
    f["pred_diff"] = pp - cp
    f["pred_absdiff"] = np.abs(pp - cp)
    f["pred_ratio"] = pp / (cp + 1.0)
    w0, w1 = windows

    def col(name, default=0.0):
        return df[name].astype(float) if name in df.columns else pd.Series(default, index=df.index)

    conn = col("n_connected_b") if mode == "B" else col("n_connected")
    f["conn_obs"] = conn.values
    f["conn_delta"] = conn.diff().fillna(0).values
    f["conn_std_w0"] = conn.rolling(w0).std().fillna(0).values
    f["conn_std_w1"] = conn.rolling(w1).std().fillna(0).values
    rel = col("n_release_inferred") if mode == "B" else col("n_release")
    f["release_obs"] = rel.values
    # Mode B (passive sniffer) CANNOT observe n_new_connections or n_paging_response —
    # those are truth/core-side counters. Use only observable signals: setup-complete as
    # a connection proxy, and no paging-response rate.
    if mode == "A":
        f["new_conn"] = col("n_new_connections").values
        f["paging_resp_rate"] = (col("n_paging_response").rolling(w1).sum()
                                 / (col("n_paging").rolling(w1).sum() + 1)).fillna(0).values
    else:
        f["new_conn"] = col("n_rrc_setup_complete").values     # observable proxy
        f["paging_resp_rate"] = np.zeros(len(cp), dtype=np.float64)  # not observable
    # temporal smoothing of the two base predictions (disagreement dynamics)
    # causal rolling (min_periods=1) — NO bfill, which would leak future predictions
    f["cell_pred_ma"] = pd.Series(cp).rolling(w0, min_periods=1).mean().values
    f["perue_pred_ma"] = pd.Series(pp).rolling(w0, min_periods=1).mean().values
    return f.astype(np.float32)


# ======================================================================================
# fusion families (shared arithmetic for BOTH Hybrid-XGB and Hybrid-LSTM)
# ======================================================================================
# Three fusion families the tuner can choose between:
#   direct   : hybrid predicts the count directly from the fusion vector z_t
#   residual : hybrid predicts a CORRECTION on top of the strong cell branch
#              y_hat = cell + f(z)
#   gated    : per-sample convex blend of the two base predictions
#              y_hat = w * cell + (1 - w) * perUE,   w in [0, 1]
# The gate w is LEARNED: for XGB a regressor predicts the blend weight; for the LSTM a
# sigmoid head emits it per timestep (see train_lstm_regressor).
FUSION_MODES = ("direct", "residual", "gated")


def gate_target(y, cell, perue, eps=1e-3):
    """The convex weight w* that makes  w*cell + (1-w)*perUE == y  exactly (where the two
    base predictions differ), clipped to [0,1]. Used as the regression target for the XGB
    gate. Where cell ~= perUE the blend is underdetermined, so w* falls back to 0.5."""
    y = np.asarray(y, np.float64); cell = np.asarray(cell, np.float64)
    perue = np.asarray(perue, np.float64)
    denom = cell - perue
    safe = np.abs(denom) > eps
    w = np.where(safe, (y - perue) / np.where(safe, denom, 1.0), 0.5)
    return np.clip(w, 0.0, 1.0)


def blend(cell, perue, w):
    """Convex combination w*cell + (1-w)*perUE (w already in [0,1])."""
    w = np.clip(np.asarray(w, np.float64), 0.0, 1.0)
    return w * np.asarray(cell, np.float64) + (1.0 - w) * np.asarray(perue, np.float64)


def fit_hybrid_xgb(Z_tr, y_tr, cell_tr, perue_tr, params, seed, fusion_mode: str):
    """Fit the Hybrid-XGB for the chosen fusion family. Returns a small dict carrying the
    fitted model and the mode so predict knows how to combine.
      direct   -> model predicts y
      residual -> model predicts y - cell
      gated    -> model predicts the convex blend weight w* (target from gate_target)"""
    if fusion_mode == "residual":
        target = y_tr - cell_tr
    elif fusion_mode == "gated":
        target = gate_target(y_tr, cell_tr, perue_tr)
    else:
        target = y_tr
    m = XGBRegressor(**params, random_state=seed)
    m.fit(Z_tr, target)
    return {"model": m, "fusion_mode": fusion_mode}


def predict_hybrid_xgb(fitted, Z, cell_pred, perue_pred, fusion_mode: str = None):
    """Combine base predictions per the fusion family. ``fitted`` may be the dict from
    fit_hybrid_xgb (preferred) or a bare model (then fusion_mode must be given)."""
    if isinstance(fitted, dict):
        model = fitted["model"]; fusion_mode = fitted["fusion_mode"]
    else:
        model = fitted
    out = model.predict(Z)
    if fusion_mode == "residual":
        return cell_pred + out
    if fusion_mode == "gated":
        return blend(cell_pred, perue_pred, out)
    return out


# ======================================================================================
# generic causal-window LSTM regressor  (Cell-LSTM base branch AND Hybrid-LSTM)  [torch]
# ======================================================================================
if _TORCH:
    class SeqLSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0.0,
                                batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            seq, _ = self.lstm(x)
            return self.head(seq[:, -1, :]).squeeze(-1)

    # backward-compatible alias (older references to HybridLSTM keep working)
    HybridLSTM = SeqLSTMRegressor


def _lstm_reg_kwargs(params: dict) -> dict:
    """Pull the regressor hyper-parameters out of a tuned/param dict (ignore extras)."""
    return dict(seq_len=int(params["seq_len"]), hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]), dropout=float(params["dropout"]),
                lr=float(params["lr"]), epochs=int(params["epochs"]))


def make_windows(Z: np.ndarray, y: np.ndarray, idx_range, seq_len: int,
                 forbid_boundaries=()):
    """Build (window, target) pairs ending at each t in idx_range. A window spans
    [t-seq_len+1, t]; windows crossing any boundary in forbid_boundaries (e.g. OOF fold
    edges) are skipped so no sequence mixes differently-fitted OOF segments."""
    fb = sorted(set(forbid_boundaries))
    Xs, ys, ts = [], [], []
    for t in idx_range:
        lo = t - seq_len + 1
        if lo < 0:
            continue
        if any(lo < bnd <= t for bnd in fb):
            continue
        win = Z[lo:t + 1]
        if not np.isfinite(win).all() or not np.isfinite(y[t]):
            continue                       # skip windows spanning the OOF NaN gap
        Xs.append(win); ys.append(y[t]); ts.append(t)
    if not Xs:
        return (np.empty((0, seq_len, Z.shape[1]), np.float32),
                np.empty((0,), np.float32), np.empty((0,), int))
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32), np.asarray(ts, int)


def _combine_np(raw, cell, perue, fusion_mode):
    """Numpy mirror of the LSTM head combination (used for tests / eval)."""
    if fusion_mode == "direct":
        return np.asarray(raw, np.float64)
    if fusion_mode == "residual":
        return np.asarray(cell, np.float64) + np.asarray(raw, np.float64)
    w = 1.0 / (1.0 + np.exp(-np.asarray(raw, np.float64)))          # sigmoid gate
    return blend(cell, perue, w)


def train_lstm_regressor(Z_train, y_train, train_idx, Z_eval, y_eval, eval_idx, seq_len,
                         hidden_size, num_layers, dropout, lr, epochs, device,
                         train_forbid_boundaries=(), eval_forbid_boundaries=(),
                         batch_size=256, seed=0, weight_decay=1e-5,
                         grad_clip=1.0, fusion_mode="direct",
                         base_cell_train=None, base_perue_train=None,
                         base_cell_eval=None, base_perue_eval=None):
    """Generic causal-window LSTM regressor used by BOTH the Cell-LSTM base branch (Z =
    compact cell features, ``fusion_mode='direct'``) and Hybrid-LSTM (Z = fusion vectors).

    fusion_mode controls how the network output is turned into a count:
      direct   -> y_hat = f(window)
      residual -> y_hat = base_cell[t] + f(window)
      gated    -> w = sigmoid(f(window)); y_hat = w*base_cell[t] + (1-w)*base_perUE[t]
    For residual/gated the base predictions (COUNT scale, unstandardized) must be passed;
    they are aligned to each window's END index t.

    Forbidden boundaries are SEPARATE for train and eval. OOF training windows must not
    mix predictions from different folds (pass the fold bounds as
    ``train_forbid_boundaries``), but the eval windows read ``Z_eval`` from a SINGLE
    full-train base model, so they take no fold boundaries — otherwise the first
    ``seq_len-1`` val/test predictions would be dropped (and, worse, the surviving eval
    window count would depend on seq_len, making trials incomparable).

    Memory: windows stay on the CPU and are moved to the GPU one BATCH at a time (train and
    eval), so the full window tensor is never resident on the device. Returns
    (eval_pred_full[len=len(y_eval)], eval_mae). SmoothL1; train-only standardization;
    weight decay; gradient clipping."""
    if not _TORCH:
        raise RuntimeError("torch not available")
    if fusion_mode != "direct":
        assert base_cell_train is not None and base_perue_train is not None, \
            "residual/gated fusion needs base_cell/base_perue for the train windows"
        assert base_cell_eval is not None and base_perue_eval is not None, \
            "residual/gated fusion needs base_cell/base_perue for the eval windows"
    torch.manual_seed(seed); np.random.seed(seed)
    # train-only standardization of the input vectors (mixed scales)
    fin = np.isfinite(Z_train).all(axis=1)
    tr_rows = [t for t in train_idx if t < len(Z_train) and fin[t]]
    mu = Z_train[tr_rows].mean(axis=0) if tr_rows else np.zeros(Z_train.shape[1])
    sd = np.maximum(Z_train[tr_rows].std(axis=0) if tr_rows else 1.0, 1e-6)
    Zt = (Z_train - mu) / sd
    Ze = (Z_eval - mu) / sd
    Xtr, ytr, ttr = make_windows(Zt, y_train, train_idx, seq_len, train_forbid_boundaries)
    Xev, yev, tev = make_windows(Ze, y_eval, eval_idx, seq_len, eval_forbid_boundaries)
    if len(Xtr) == 0 or len(Xev) == 0:
        return np.full(len(y_eval), np.nan), float("nan")
    dev = torch.device(device)
    model = SeqLSTMRegressor(Z_train.shape[1], hidden_size, num_layers, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.SmoothL1Loss()

    def _gather(arr, idx):
        return None if arr is None else torch.tensor(np.asarray(arr, np.float32)[idx])

    # base values aligned to window END indices (kept on CPU, sliced per batch)
    ctr = _gather(base_cell_train, ttr); ptr = _gather(base_perue_train, ttr)
    cev = np.asarray(base_cell_eval, np.float32)[tev] if base_cell_eval is not None else None
    pev = np.asarray(base_perue_eval, np.float32)[tev] if base_perue_eval is not None else None
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)         # CPU tensors
    N = len(Xtr_t)
    model.train()
    for _ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            b = perm[i:i + batch_size]
            xb = Xtr_t[b].to(dev); yb = ytr_t[b].to(dev)
            opt.zero_grad()
            raw = model(xb)
            if fusion_mode == "direct":
                yhat = raw
            elif fusion_mode == "residual":
                yhat = ctr[b].to(dev) + raw
            else:                                                # gated
                w = torch.sigmoid(raw)
                yhat = w * ctr[b].to(dev) + (1.0 - w) * ptr[b].to(dev)
            loss = crit(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
    # batched eval (CPU windows -> GPU per chunk)
    model.eval()
    preds = np.empty(len(Xev), dtype=np.float32)
    Xev_t = torch.tensor(Xev)
    with torch.no_grad():
        for i in range(0, len(Xev_t), 4096):
            raw = model(Xev_t[i:i + 4096].to(dev))
            if fusion_mode == "direct":
                out = raw.cpu().numpy()
            elif fusion_mode == "residual":
                out = cev[i:i + 4096] + raw.cpu().numpy()
            else:
                w = torch.sigmoid(raw).cpu().numpy()
                out = blend(cev[i:i + 4096], pev[i:i + 4096], w)
            preds[i:i + 4096] = out
    # eval_idx / tev are ABSOLUTE indices; y_eval is the FULL-length array. Map by t.
    full = np.full(len(y_eval), np.nan)
    full[tev] = preds
    mae = mean_absolute_error(yev, preds)
    return full, float(mae)


# ======================================================================================
# PerUE-LSTM: per-track causal sequence classifier, probabilities summed to a count
# ======================================================================================
# The per-UE feature vector at slot t depends on the track's ACCUMULATED state
# (n_connections, session_lengths, elapsed, ...), which evolves as events arrive. A
# sequence model therefore needs the state AS IT WAS at each past sample step, so we
# replay the event stream once (exactly the state machine of per_ue.build_perue_xgb /
# compute_per_ue) and record, per track, a snapshot of its feature vector at every
# ELIGIBLE sample step (connected OR within the prune horizon). Windows for the LSTM are
# then built from consecutive snapshots of the same track; pruning gaps break a run so a
# window never bridges an idle-out/reconnect. Counting is symmetric to PerUE-XGB: at each
# sample step, sum P(present) over the tracks eligible at that step.

class PerUESnapshots:
    """Per-track feature-snapshot time series over the sampling grid.

    feats[pid]  : float32 [n_pid, F]  feature vectors at each eligible step
    labels[pid] : int8    [n_pid]     present(t) label at that step
    steps[pid]  : int     [n_pid]     ordinal index into sample_times (may have GAPS
                                      when a track is pruned then reappears)
    sample_times: int     [K]         the slot index of each sampling step
    """
    def __init__(self, feats, labels, steps, sample_times, n_feat):
        self.feats = feats
        self.labels = labels
        self.steps = steps
        self.sample_times = np.asarray(sample_times, dtype=int)
        self.n_feat = int(n_feat)

    @staticmethod
    def _run_start(steps, j, seq_len):
        """First index of the maximal run of CONSECUTIVE steps ending at j, capped so the
        window is at most seq_len long. A prune gap (non-consecutive step) stops the walk,
        so a window never bridges an idle-out/reconnect."""
        lo = j
        while lo > 0 and steps[lo] - steps[lo - 1] == 1 and (j - lo + 1) < seq_len:
            lo -= 1
        return lo

    @staticmethod
    def _window_from(feats_pid, lo, j, seq_len):
        """Causal window feats_pid[lo:j+1], LEFT-PADDED to seq_len by repeating the run's
        first snapshot (so a freshly-appeared track still contributes a prediction)."""
        win = feats_pid[lo:j + 1]
        if len(win) < seq_len:
            pad = np.repeat(win[:1], seq_len - len(win), axis=0)
            win = np.concatenate([pad, win], axis=0)
        return win

    def positions(self, seq_len: int):
        """Flat per-position index arrays (no window materialization): for every recorded
        (track, step) endpoint, the track id, run-start ``lo``, endpoint ``j``, the
        sampling-step index ``step_k`` and the present-label. Windows are built lazily from
        these by the Dataset (or eagerly by materialize)."""
        pids, los, js, ks, labs = [], [], [], [], []
        for pid, sn in self.feats.items():
            st = self.steps[pid]; lab = self.labels[pid]
            for j in range(len(st)):
                lo = self._run_start(st, j, seq_len)
                pids.append(pid); los.append(lo); js.append(j)
                ks.append(int(st[j])); labs.append(int(lab[j]))
        return (np.asarray(pids, int), np.asarray(los, int), np.asarray(js, int),
                np.asarray(ks, int), np.asarray(labs, np.int8))

    def materialize(self, seq_len: int):
        """Eagerly build one causal window per (track, step) position. Kept for tests and
        small uses; the trainer/predictor use positions() + a lazy Dataset instead.
        Returns (windows[N,seq_len,F] float32, labels[N] int8, step_k[N] int)."""
        pids, los, js, ks, labs = self.positions(seq_len)
        if len(pids) == 0:
            return (np.empty((0, seq_len, self.n_feat), np.float32),
                    np.empty((0,), np.int8), np.empty((0,), int))
        Xs = np.stack([self._window_from(self.feats[int(pids[i])], int(los[i]),
                                         int(js[i]), seq_len) for i in range(len(pids))])
        return Xs.astype(np.float32), labs, ks


def build_perue_snapshots(ue_events, ue_meta, horizon, subsample, prune_elapsed,
                          cell_context_fn=None) -> PerUESnapshots:
    """Replay connect/disconnect events over range(0, horizon, subsample) and record a
    per-track feature snapshot + present-label at every eligible step. Mirrors the exact
    state machine (whitelist + canonical same-slot ordering) of per_ue.build_perue_xgb."""
    all_events = [(t, pid, et) for pid, evs in ue_events.items()
                  for t, et in evs if et in ("connect", "disconnect")]
    all_events.sort(key=event_key_with_pid)   # same-slot: disconnect before connect
    sample_times = list(range(0, int(horizon), subsample))
    tracks: dict[int, UETrack] = {}
    feats: dict[int, list] = {}
    labels: dict[int, list] = {}
    steps: dict[int, list] = {}
    ev_idx = 0
    for k, t in enumerate(sample_times):
        while ev_idx < len(all_events) and all_events[ev_idx][0] <= t:
            ev_t, pid, et = all_events[ev_idx]
            if pid not in tracks:
                tr = UETrack(pid=pid, first_seen=ev_t)
                meta = ue_meta.get(pid, {})
                tr.usage_intensity = meta.get("usage_intensity")
                tr.dwell_type = meta.get("dwell_type")
                tr.bg_class = meta.get("device_bg_class")
                tracks[pid] = tr
            tr = tracks[pid]
            if et == "connect":
                if not tr.is_connected:
                    tr.n_connections += 1; tr._connect_start = ev_t
                tr.is_connected = True; tr.last_seen = ev_t
            elif et == "disconnect":
                if tr.is_connected and tr._connect_start >= 0:
                    tr.total_active_time += ev_t - tr._connect_start
                    tr.session_lengths.append(ev_t - tr._connect_start)
                tr.is_connected = False; tr.last_seen = ev_t; tr._connect_start = -1
            ev_idx += 1
        ctx = cell_context_fn(t) if cell_context_fn is not None else None
        for pid, tr in tracks.items():
            elapsed = t - tr.last_seen
            if elapsed > prune_elapsed and not tr.is_connected:
                continue                              # ineligible this step (gap)
            meta = ue_meta[pid]
            label = 1 if meta["enter_time"] <= t < meta["exit_time"] else 0
            feats.setdefault(pid, []).append(per_ue_features_extended(tr, t, ctx))
            labels.setdefault(pid, []).append(label)
            steps.setdefault(pid, []).append(k)
    n_feat = 0
    feats_np, labels_np, steps_np = {}, {}, {}
    for pid in feats:
        arr = np.asarray(feats[pid], dtype=np.float32)
        n_feat = arr.shape[1]
        feats_np[pid] = arr
        labels_np[pid] = np.asarray(labels[pid], dtype=np.int8)
        steps_np[pid] = np.asarray(steps[pid], dtype=int)
    return PerUESnapshots(feats_np, labels_np, steps_np, sample_times, n_feat)


def _train_feature_stats(snaps: PerUESnapshots, boundary_slot: int):
    """Train-only per-feature mean/std over the SNAPSHOT rows whose sampling slot is before
    the boundary (memory-light: over unique snapshots, not the windowed/padded frames)."""
    rows = []
    for pid, sn in snaps.feats.items():
        m = snaps.sample_times[snaps.steps[pid]] < boundary_slot
        if m.any():
            rows.append(sn[m])
    if not rows:
        F = snaps.n_feat
        return np.zeros(F, np.float32), np.ones(F, np.float32)
    allr = np.concatenate(rows, axis=0)
    return (allr.mean(axis=0).astype(np.float32),
            np.maximum(allr.std(axis=0), 1e-6).astype(np.float32))


if _TORCH:
    class PerUELSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0.0,
                                batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            seq, _ = self.lstm(x)
            return self.head(seq[:, -1, :]).squeeze(-1)   # logit

    class PerUESequenceDataset(torch.utils.data.Dataset):
        """Lazily materializes one standardized causal window per position on __getitem__.
        Only the (small) per-track snapshot arrays + integer index vectors are held; the
        dense [N, seq_len, F] window tensor is NEVER built, and the DataLoader moves data to
        the GPU one batch at a time. Returns (window[seq_len,F] float32, step_k int64,
        label float32)."""
        def __init__(self, snaps, pids, los, js, ks, labs, seq_len, mu, sd):
            self.feats = snaps.feats
            self.pids = pids; self.los = los; self.js = js
            self.ks = ks; self.labs = labs; self.seq_len = int(seq_len)
            self.mu = np.asarray(mu, np.float32); self.sd = np.asarray(sd, np.float32)

        def __len__(self):
            return len(self.pids)

        def __getitem__(self, i):
            win = PerUESnapshots._window_from(self.feats[int(self.pids[i])],
                                              int(self.los[i]), int(self.js[i]), self.seq_len)
            win = ((win - self.mu) / self.sd).astype(np.float32)
            return win, np.int64(self.ks[i]), np.float32(self.labs[i])


def _steps_to_full(per_step_count: np.ndarray, sample_times: np.ndarray,
                   total_time: int) -> np.ndarray:
    """Expand per-sampling-step counts to a full-length array, holding each value until
    the next sampled step (matching compute_per_ue's last-value hold)."""
    full = np.zeros(total_time, dtype=np.float32)
    K = len(sample_times)
    for k in range(K):
        t0 = int(sample_times[k])
        t1 = int(sample_times[k + 1]) if k + 1 < K else total_time
        full[t0:t1] = per_step_count[k]
    return full


def _perue_lstm_counts(snaps: PerUESnapshots, seq_len: int, train_boundary_slot: int,
                       params: dict, seed: int, device: str, total_time: int,
                       train_batch: int = 1024, eval_batch: int = 4096) -> np.ndarray:
    """Train a per-UE LSTM classifier on windows whose LABEL time < train_boundary_slot,
    then sum P(present) over eligible tracks at every sampling step -> full-length count.

    Windows are produced lazily by a Dataset/DataLoader and moved to the GPU one batch at a
    time (no full window tensor is ever resident). The loss is UNWEIGHTED BCE: with a
    pos_weight the sigmoid outputs are shifted away from the class posterior, so summing
    them would bias the head-count; unweighted keeps the summed probabilities calibrated as
    an expected occupancy. (Class imbalance, if it ever hurts, is better handled by a
    WeightedRandomSampler than by reweighting the loss.) Feature-side lookback may reach
    before the boundary (track state only, no labels)."""
    if not _TORCH:
        raise RuntimeError("torch not available")
    torch.manual_seed(seed); np.random.seed(seed)
    pids, los, js, ks, labs = snaps.positions(seq_len)
    K = len(snaps.sample_times)
    if len(pids) == 0:
        return np.zeros(total_time, dtype=np.float32)
    tpos = snaps.sample_times[ks]                           # label slot of each position
    tr = tpos < train_boundary_slot
    ytr = labs[tr].astype(np.float32)
    if tr.sum() == 0 or ytr.sum() == 0 or ytr.sum() == len(ytr):
        # degenerate training slice (no positives/negatives) -> fall back to zeros
        return np.zeros(total_time, dtype=np.float32)
    mu, sd = _train_feature_stats(snaps, train_boundary_slot)
    dev = torch.device(device)
    train_ds = PerUESequenceDataset(snaps, pids[tr], los[tr], js[tr], ks[tr], labs[tr],
                                    seq_len, mu, sd)
    all_ds = PerUESequenceDataset(snaps, pids, los, js, ks, labs, seq_len, mu, sd)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=train_batch, shuffle=True)
    all_ld = torch.utils.data.DataLoader(all_ds, batch_size=eval_batch, shuffle=False)
    model = PerUELSTMClassifier(snaps.n_feat, int(params["hidden_size"]),
                                int(params["num_layers"]), float(params["dropout"])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()                    # unweighted (calibrated sum)
    model.train()
    for _ep in range(int(params["epochs"])):
        for xb, _kb, yb in train_ld:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    model.eval()
    per_step = np.zeros(K, dtype=np.float32)
    with torch.no_grad():
        for xb, kb, _yb in all_ld:
            probs = torch.sigmoid(model(xb.to(dev))).cpu().numpy()
            np.add.at(per_step, kb.numpy(), probs)          # sum P(present) per step
    return _steps_to_full(per_step, snaps.sample_times, total_time)


def perue_lstm_predict(ue_events, ue_meta, cell_context_fn, total_time, train_end,
                       params, seed, device, subsample, prune_elapsed) -> np.ndarray:
    """Full-length PerUE-LSTM count: build snapshots (context at train_end), train on
    label-times < train_end, predict summed probabilities over [0, total_time)."""
    snaps = build_perue_snapshots(ue_events, ue_meta, total_time, subsample,
                                  prune_elapsed, cell_context_fn)
    return _perue_lstm_counts(snaps, int(params["seq_len"]), train_end, params, seed,
                              device, total_time)


def oof_perue_lstm(df, mode, dsm, y, train_end, params, seed, device, n_folds=4):
    """OOF PerUE-LSTM counts on [b1, train_end), symmetric to oof_perue_xgb. The per-UE
    cell context (surge threshold train-only) is rebuilt per fold, so snapshots are
    rebuilt per fold; the fold classifier trains on label-times < tr_end and the block
    [tr_end, pr_end) is kept."""
    b = _fold_bounds(train_end, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    ev, mt = dsm["events"], dsm["meta"]
    sub, prune = dsm["subsample"], dsm["prune"]
    for i in range(1, n_folds):
        tr_end, pr_end = b[i], b[i + 1]
        fold_ctx = make_cell_context_fn(df, mode=mode, train_end=tr_end,
                                        wave_period=dsm["wave_period"], feat=dsm["feat"])
        snaps = build_perue_snapshots(ev, mt, len(y), sub, prune, fold_ctx)
        full = _perue_lstm_counts(snaps, int(params["seq_len"]), tr_end, params, seed,
                                  device, len(y))
        oof[tr_end:pr_end] = full[tr_end:pr_end]
    return oof


# ======================================================================================
# search spaces (fusion is small -> shallow trees / modest LSTM)
# ======================================================================================
def sample_hybrid_xgb(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 5),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 3, 15),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        n_jobs=int(__import__("os").environ.get("LTE_XGB_JOBS", 4)),
        tree_method="hist", verbosity=0,
    )


def sample_hybrid_lstm(trial) -> dict:
    return dict(
        seq_len=trial.suggest_categorical("seq_len", [30, 60, 120, 300]),
        hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
        num_layers=trial.suggest_int("num_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        lr=trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        epochs=trial.suggest_int("epochs", 15, 60, step=15),
    )