"""
features/cell.py — Cell-level feature engineering (observable signals only).

Standard, interpretable time-series features built ON TOP of the BS-observable
counters. Two correctness properties are enforced here:

  * NO ORACLE FEATURES (fix 10.2). The former ``time_since_burst_end`` /
    ``is_burst`` read the simulator's ground-truth burst SCHEDULE. They are
    replaced by an OBSERVABLE surge signal: a causal z-score of the observed inflow
    vs its recent trailing baseline (``observable_surge``), plus a "time since
    observed surge".
  * NO TEST LEAKAGE (fix 10.3). Every threshold (release-spike, surge) is computed
    on the TRAIN slice only (``train_end``) and applied to the whole trace.

All feature-engineering knobs (EWMA alphas, lag offsets, window sizes, surge/spike
percentiles) come from FeatureCfg, so they appear in the config + registry rather
than being buried as module constants.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config.schema import FeatureCfg


def _new_and_release_signals(df: pd.DataFrame, mode: str):
    if mode == "A":
        new_s = df["n_new_connections"].astype(float)
        rel_s = df["n_release"].astype(float)
    else:
        new_s = df["n_rrc_setup_complete"].astype(float)
        rel_s = df["n_release_inferred"].astype(float)
    return new_s.reset_index(drop=True), rel_s.reset_index(drop=True)


def observable_surge(df: pd.DataFrame, mode: str, train_end: int,
                     window: int, percentile: float):
    """Observable demand-surge signal (replaces the oracle burst schedule).

    Returns (surge_z, surge_flag), float arrays of length len(df). surge_z is a
    causal z-score of observed inflow vs a strictly-past trailing baseline; the
    surge threshold is the given percentile of surge_z over the TRAIN slice only.
    """
    new_s, _ = _new_and_release_signals(df, mode)
    roll_mean = new_s.rolling(window, min_periods=1).mean().shift(1)
    roll_std = new_s.rolling(window, min_periods=1).std().shift(1)
    surge_z = ((new_s - roll_mean) / (roll_std + 1.0)).fillna(0.0).values.astype(np.float32)
    train_z = surge_z[:train_end]
    thr = float(np.percentile(train_z, percentile)) if train_z.size > 0 else 0.0
    surge_flag = (surge_z > thr).astype(np.float32)
    return surge_z, surge_flag


def _add_ewma(out: pd.DataFrame, cols: list, ewma_alphas) -> pd.DataFrame:
    for c in cols:
        s = out[c]
        for alpha in ewma_alphas:
            out[f"{c}_ewma_{alpha:.1f}"] = s.ewm(alpha=alpha, adjust=False).mean()
    return out


def _add_lagged_interactions(out: pd.DataFrame, src: pd.DataFrame, mode: str, lag_offsets) -> pd.DataFrame:
    if mode == "A":
        new_s = src["n_new_connections"].astype(float)
        rel_s = src["n_release"].astype(float)
        nc_s = src["n_connected"].astype(float)
    else:
        new_s = src["n_rrc_setup_complete"].astype(float)
        rel_s = src["n_release_inferred"].astype(float)
        nc_s = (src["n_connected_b"] if "n_connected_b" in src.columns
                else src["n_connected"]).astype(float)
    for k in lag_offsets:
        for j in lag_offsets:
            out[f"lag_new{k}_rel{j}"] = new_s.shift(k).fillna(0) * rel_s.shift(j).fillna(0)
        out[f"lag_new{k}_nc"] = new_s.shift(k).fillna(0) * nc_s
    if mode == "A" and "n_paging_response" in src.columns:
        pr_s = src["n_paging_response"].astype(float)
        for k in lag_offsets:
            out[f"lag_pr{k}_nc"] = pr_s.shift(k).fillna(0) * nc_s
    return out


def _add_delta(out: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in out.columns:
            out[f"d_{c}"] = out[c].diff().fillna(0)
            out[f"d2_{c}"] = out[c].diff(2).fillna(0)
    return out


def _add_time_since(out: pd.DataFrame, src: pd.DataFrame, mode: str,
                    total_time: int, train_end: int, feat: FeatureCfg) -> pd.DataFrame:
    # OBSERVABLE surge (replaces oracle time_since_burst_end / is_burst)
    surge_z, surge_flag = observable_surge(src, mode, train_end, feat.surge_window, feat.surge_percentile)
    out["surge_z"] = surge_z
    out["surge_flag"] = surge_flag
    tss = np.zeros(total_time, dtype=np.float32)
    last = -1
    for t in range(total_time):
        if surge_flag[t] > 0:
            last = t
        tss[t] = (t - last) if last >= 0 else t
    out["time_since_observed_surge"] = tss

    if mode == "A" and "n_paging_response" in src.columns:
        pr = src["n_paging_response"].values
        tsp = np.zeros(total_time, dtype=np.float32)
        last = -1
        for t in range(total_time):
            if pr[t] > 0:
                last = t
            tsp[t] = (t - last) if last >= 0 else t
        out["time_since_paging_resp"] = tsp

    # release-spike: threshold from the TRAIN slice only (fix 10.3)
    rel = src["n_release"].values if mode == "A" else src["n_release_inferred"].values
    train_rel = rel[:train_end]
    thresh = np.percentile(train_rel, feat.release_spike_percentile) if train_rel.size > 0 else 0
    tsr = np.zeros(total_time, dtype=np.float32)
    last = -1
    for t in range(total_time):
        if rel[t] > thresh:
            last = t
        tsr[t] = (t - last) if last >= 0 else t
    out["time_since_release_spike"] = tsr
    return out


def build_cell_features(df: pd.DataFrame, mode: str, bs_features: Sequence[str],
                        train_end: int, wave_period: int,
                        rolling_windows: Sequence[int], feat: FeatureCfg) -> pd.DataFrame:
    """Cell-level feature builder. mode is "A" or "B".

    train_end : split index (exclusive); all thresholds use df[:train_end] only.
    """
    df_view = df.copy()
    if mode == "B" and "n_connected_b" in df.columns:
        df_view["n_connected"] = df["n_connected_b"]
    out = df_view[list(bs_features)].copy()

    if mode == "A":
        new_conn_signal = df["n_new_connections"].values.astype(float)
        rel_signal = df["n_release"].values.astype(float)
    else:
        new_conn_signal = df["n_rrc_setup_complete"].values.astype(float)
        rel_signal = df["n_release_inferred"].values.astype(float)
    new_s = pd.Series(new_conn_signal)
    rel_s = pd.Series(rel_signal)

    out["delta_flow"] = new_conn_signal - rel_signal
    out["delta_flow_cum_300"] = pd.Series(out["delta_flow"]).rolling(
        feat.flow_cumulative_window, min_periods=1).sum().values
    for w in feat.turnover_windows:
        out[f"turnover_ratio_{w}"] = (
            rel_s.rolling(w, min_periods=1).sum().values /
            (new_s.rolling(w, min_periods=1).sum().values + 1.0)
        )
    t_arr = df["t"].values if "t" in df.columns else np.arange(len(df))
    out["t_sin"] = np.sin(2 * np.pi * t_arr / wave_period)
    out["t_cos"] = np.cos(2 * np.pi * t_arr / wave_period)
    if mode == "A":
        pg = pd.Series(df["n_paging"].values.astype(float))
        resp = pd.Series(df["n_paging_response"].values.astype(float))
        for w in feat.paging_windows:
            out[f"paging_resp_rate_{w}"] = (
                resp.rolling(w, min_periods=1).sum().values /
                (pg.rolling(w, min_periods=1).sum().values + 1.0)
            )
        if "n_meas_report" in df.columns:
            out["meas_per_conn"] = (
                df["n_meas_report"].values.astype(float) /
                np.maximum(df["n_connected"].values, 1)
            )
    base_cols = list(out.columns)

    out = _add_ewma(out, base_cols, feat.ewma_alphas)
    out = _add_lagged_interactions(out, df, mode, feat.lag_offsets)
    out = _add_delta(out, base_cols)
    out = _add_time_since(out, df, mode, total_time=len(df), train_end=train_end, feat=feat)

    for c in list(out.columns):
        for w in rolling_windows:
            out[f"{c}_ma_{w}"] = out[c].rolling(w).mean()
            out[f"{c}_std_{w}"] = out[c].rolling(w).std()

    return out.fillna(0)


def build_legacy_survival_hybrid_features(cell_df: pd.DataFrame, surv_est: np.ndarray,
                                          xgb_est: np.ndarray,
                                          rolling_windows: Sequence[int]) -> pd.DataFrame:
    """LEGACY (pre-tuned-protocol) hybrid features = cell features + Per-UE estimates,
    including a Survival estimate. The tuned six-model protocol drops Survival and fuses
    via estimation.fusion instead, so this is kept ONLY for the old experiments/training.py
    runner. Do not use it in the tuned pipeline (it would reintroduce Survival)."""
    out = cell_df.copy()
    out["pu_surv"] = surv_est
    out["pu_xgb"] = xgb_est
    out["pu_mean"] = (surv_est + xgb_est) / 2
    out["pu_diff"] = surv_est - xgb_est
    for w in rolling_windows:
        out[f"pu_surv_ma_{w}"] = out["pu_surv"].rolling(w).mean()
        out[f"pu_xgb_ma_{w}"] = out["pu_xgb"].rolling(w).mean()
    return out.fillna(0)


# Default short windows for the compact Cell-LSTM feature set (kept in one place so the
# full-length predictor and the per-fold OOF builder always agree).
CELL_LSTM_SHORT_WINDOWS = (5,)


def build_cell_lstm_features(df: pd.DataFrame, mode: str, bs_columns: Sequence[str],
                             short_windows: Sequence[int] = CELL_LSTM_SHORT_WINDOWS
                             ) -> pd.DataFrame:
    """Compact, causal feature set for the Cell-LSTM branch.

    The XGB branch uses a large pre-baked bank of rolling/lag/EWMA cross-terms (~1500
    columns for Mode A). Feeding that same bank into an LSTM with a long ``seq_len``
    produces enormous windows and defeats the point of letting the recurrent model learn
    the temporal structure itself. This builder instead keeps only:

        raw observable counters  +  1st difference  +  short rolling mean  +  short rolling std

    i.e. ~4x the number of raw observable counters (Mode A ~= 80, Mode B ~= 32 with a
    single short window) — an order of magnitude smaller than the XGB feature bank.

    OBSERVATION FIDELITY (critical): Mode B's ``bs_columns`` still lists ``n_connected``,
    but a passive sniffer only sees the DRX-degraded ``n_connected_b``. Exactly as
    build_cell_features does, we remap ``n_connected -> n_connected_b`` for Mode B so the
    Cell-LSTM never reads the clean truth counter (no truth-side leakage).

    Every transform is causal (``diff``, trailing ``rolling``) and carries NO fitted
    threshold, so there is no train/test leakage and no ``train_end`` argument is needed.
    NaNs from the initial diff / short window are zero-filled so every slot yields a
    usable window.
    """
    df_view = df.copy()
    if mode == "B" and "n_connected_b" in df_view.columns:
        df_view["n_connected"] = df_view["n_connected_b"]     # observable, DRX-degraded
    cols: dict[str, np.ndarray] = {}
    for c in bs_columns:
        s = df_view[c].astype(float)
        cols[c] = s.values
        cols[f"d_{c}"] = s.diff().fillna(0.0).values
        for w in short_windows:
            cols[f"{c}_ma{w}"] = s.rolling(w, min_periods=1).mean().values
            cols[f"{c}_sd{w}"] = s.rolling(w, min_periods=1).std().fillna(0.0).values
    # single construction (dict -> DataFrame) avoids the fragmentation of repeated inserts
    return pd.DataFrame(cols, index=df.index).astype(np.float32)
