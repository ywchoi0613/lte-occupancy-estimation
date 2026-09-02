"""Superseded single-seed pipeline. NO PUBLISHED RESULT COMES FROM THIS FILE.

Every number in the paper is produced by ``training_tuned.run_one_seed_tuned``
(``runner.py --model-set tuned``, the default). This module predates the tuned
protocol and is retained only so the earlier survival-based configuration remains
inspectable: it defines the Survival estimator, the 1505-dimensional Cell-LSTM and
the survival-hybrid features, none of which appear in the paper. It is not
maintained and its outputs are not comparable to the reported results.

    truth simulator -> observation (Mode A / Mode B) -> Per-UE (survival + XGB)
    -> cell features -> per-mode models

Six estimators per mode in THIS superseded configuration (PerUE_Surv, PerUE_XGB,
Cell_XGB, Cell_LSTM, Hyb_XGB, Hyb_LSTM). The paper's six replace PerUE_Surv with
Hybrid_XGB/Hybrid_LSTM as tuned branches; see training_tuned.py. Two naive
baselines are also computed here:
  naive_A = clean n_connected      (what a cooperative eNB sees)
  naive_B = DRX-degraded n_connected_b  (what the sniffer sees)

Mode A uses the clean per-UE event stream; Mode B uses a sniffer-DEGRADED per-UE
view (release lag + false negatives, plus S-TMSI fragmentation if enabled), so its
Per-UE / Hybrid estimates are a genuine lower bound rather than sharing Mode A's
exact session boundaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from ..config.schema import ExperimentConfig
from ..simulation.engine import Sim
from ..observation.mode_a import ModeAObserver
from ..observation.mode_b import (
    ModeBObserver, mode_b_xgb_params, install_stmsi_realloc_patch, split_events_by_stmsi,
)
from ..features.cell import build_cell_features, build_legacy_survival_hybrid_features
from ..estimation.survival import fit_clustered_empirical_survival
from ..estimation.per_ue import build_perue_xgb, compute_per_ue, make_cell_context_fn
from ..estimation.models import train_xgb_reg, train_lstm_model
from ..estimation.calibration import isotonic_correction


def run_one_seed(seed: int, timer: int, modes, cfg: ExperimentConfig, device,
                 verbose: bool = True):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    sim_cfg = cfg.simulation
    total_time = sim_cfg.time.total_slots
    split_idx = int(total_time * sim_cfg.time.train_ratio)
    seq_len = sim_cfg.time.seq_len
    wave_period = sim_cfg.arrival.diurnal_period_slots
    rolling_windows = sim_cfg.time.rolling_windows
    stmsi = cfg.observation.stmsi
    skip_lstm = os.environ.get("LTE_SKIP_LSTM", "0") == "1"   # XGB-only fast path
    # XGB hyperparameter set for the A-vs-B fairness diagnostic (2x2):
    #   default  -> current asymmetric baseline (A: xgb_reg, B: xgb_reg+overrides)
    #   common_a -> BOTH modes use A's params (xgb_reg)
    #   common_b -> BOTH modes use B's params (xgb_reg + mode_b_xgb_overrides)
    # Diagnostic only: the final config must be chosen on validation, not test.
    xgb_param_set = os.environ.get("LTE_XGB_PARAM_SET", "default")

    def _xgb_params_for(mode: str) -> dict:
        if xgb_param_set == "common_a":
            return dict(cfg.model.xgb_reg)
        if xgb_param_set == "common_b":
            return mode_b_xgb_params(cfg.model.xgb_reg, cfg.model.mode_b_xgb_overrides, seed)
        if mode == "B":   # default asymmetric baseline
            return mode_b_xgb_params(cfg.model.xgb_reg, cfg.model.mode_b_xgb_overrides, seed)
        return dict(cfg.model.xgb_reg)

    def _fit_xgb(X, y, mode: str) -> XGBRegressor:
        p = _xgb_params_for(mode)
        if "random_state" not in p:
            p = dict(p, random_state=seed)
        m = XGBRegressor(**p); m.fit(X, y); return m

    if verbose:
        print(f"\n[Seed {seed}, timer={timer}s, modes={list(modes)}]  Device: {device}"
              + (f"  XGB_PARAM_SET={xgb_param_set}" if xgb_param_set != "default" else ""))

    # ---- Step 1: TRUTH simulator (+ optional S-TMSI identity noise) ----
    t0 = time.time()
    if stmsi.enabled:
        install_stmsi_realloc_patch(Sim, stmsi.realloc_mean_slots)
        if verbose:
            print("  [S-TMSI realloc ENABLED — Mode B realistic identity noise]")
    sim = Sim(seed=seed, cfg=sim_cfg, rrc_timer=timer)
    if stmsi.enabled:
        sim._stmsi_realloc_mean = stmsi.realloc_mean_slots
    df = sim.run()

    # ---- Observation layer ----
    obs_a = ModeAObserver(cfg.observation)
    obs_b = ModeBObserver(cfg.observation, seed=seed)
    truth_df = df.copy()                  # truth-only snapshot BEFORE observation columns
    df = obs_b.annotate(df)               # adds n_connected_b, n_release_inferred (in place)

    if verbose:
        print(f"  Sim done ({time.time()-t0:.0f}s)  mean_present={df['n_present'].mean():.0f}  "
              f"mean_conn={df['n_connected'].mean():.0f}")

    y_all = df["n_present"].values
    y_te = y_all[split_idx:]
    naive_a = mean_absolute_error(y_te, obs_a.naive_connected(df)[split_idx:])
    naive_b = mean_absolute_error(y_te, obs_b.naive_connected(df)[split_idx:])

    results = {
        "seed": seed, "timer": timer,
        "mean_present": float(df["n_present"].mean()),
        "mean_connected": float(df["n_connected"].mean()),
        "naive_A": float(naive_a),
        "naive_B": float(naive_b),
    }
    preds = {
        "t": df["t"].values, "y": y_all,
        "n_connected": df["n_connected"].values,
        "n_connected_b": df["n_connected_b"].values,
        "n_idle": df["n_idle"].values, "split_idx": split_idx,
    }
    for col in ["n_idle_low", "n_conn_low", "n_idle_medium", "n_conn_medium",
                "n_idle_high", "n_conn_high"]:
        preds[col] = df[col].values

    # ---- Step 2: per-UE event sets per mode ----
    # Mode A (cooperative eNB) uses the clean truth event stream. Mode B (passive
    # sniffer) uses a DEGRADED per-UE view (release lag + false negatives), and — if
    # S-TMSI reallocation is on — fragmented tracks on top. Because Mode B's per-UE
    # events are now degraded even with S-TMSI off, PerUE_Surv_A and PerUE_Surv_B are
    # no longer identical (Mode B's survival is fit on noisier session boundaries).
    base_events, base_meta = sim.ue_events, sim.ue_meta
    b_events, b_meta = obs_b.observe_ue_events(base_events, base_meta, total_time=total_time)
    if stmsi.enabled:
        b_events, b_meta = split_events_by_stmsi(b_events, b_meta)
        if verbose:
            print(f"  [S-TMSI split (Mode B only): {len(base_events)} UEs -> {len(b_events)} segments]")

    # Survival is fit per event-set (cached).
    _surv_cache = {}

    def _survival_for(events, meta, key):
        if key not in _surv_cache:
            _surv_cache[key] = fit_clustered_empirical_survival(
                events, meta, split_idx, n_clusters=cfg.model.per_ue.n_clusters,
                max_elapsed=cfg.model.per_ue.survival_max_elapsed)
        return _surv_cache[key]

    # ---- Step 3: per-mode pipeline ----
    subsample = cfg.model.per_ue.subsample
    prune = cfg.model.per_ue.prune_elapsed
    manifests = {}
    for mode in modes:
        t2 = time.time()
        observer = obs_a if mode == "A" else obs_b
        bs_features = observer.feature_columns
        if mode == "B":
            m_events, m_meta, surv_key = b_events, b_meta, "B"
        else:
            m_events, m_meta, surv_key = base_events, base_meta, "A"
        get_survival, pid_clusters, surv_name = _survival_for(m_events, m_meta, surv_key)

        ctx_fn = make_cell_context_fn(df, mode=mode, train_end=split_idx,
                                      wave_period=wave_period, feat=cfg.features)
        perue_xgb = build_perue_xgb(
            m_events, m_meta, split_idx, seed=seed,
            xgb_clf_params=cfg.model.xgb_clf, subsample=subsample, prune_elapsed=prune,
            cell_context_fn=ctx_fn)
        surv_raw, xgb_raw = compute_per_ue(
            m_events, m_meta, get_survival, pid_clusters, perue_xgb,
            total_time, subsample=subsample, prune_elapsed=prune, cell_context_fn=ctx_fn)
        surv_corr = isotonic_correction(surv_raw, y_all, split_idx)
        xgb_corr = isotonic_correction(xgb_raw, y_all, split_idx)
        results[f"PerUE_Surv_{mode}"] = float(mean_absolute_error(y_te, surv_corr[split_idx:]))
        results[f"PerUE_XGB_{mode}"] = float(mean_absolute_error(y_te, xgb_corr[split_idx:]))
        preds[f"PerUE_Surv_{mode}"] = surv_corr
        preds[f"PerUE_XGB_{mode}"] = xgb_corr

        cell_feat = build_cell_features(df, mode, bs_features, train_end=split_idx,
                                        wave_period=wave_period, rolling_windows=rolling_windows,
                                        feat=cfg.features)
        raw_set = set(bs_features)
        manifests[mode] = {
            "raw_features": list(bs_features), "n_raw": len(bs_features),
            "n_total": int(cell_feat.shape[1]),
            "derived_features": [c for c in cell_feat.columns if c not in raw_set],
        }
        X_cell = cell_feat.values.astype(np.float32)

        # Cell_XGB — XGB params governed by LTE_XGB_PARAM_SET (default = current
        # asymmetric baseline: Mode B uses its own noisier-signal config).
        xgb_cell = _fit_xgb(X_cell[:split_idx], y_all[:split_idx], mode)
        cell_xgb_pred = xgb_cell.predict(X_cell)
        results[f"Cell_XGB_{mode}"] = float(mean_absolute_error(y_te, cell_xgb_pred[split_idx:]))
        preds[f"Cell_XGB_{mode}"] = cell_xgb_pred

        # Cell_LSTM
        if skip_lstm:
            lstm_p, lstm_y = np.array([]), np.array([])
        else:
            lstm_p, lstm_y = train_lstm_model(X_cell, y_all, split_idx, seq_len, seed, device, cfg.model.lstm)
        results[f"Cell_LSTM_{mode}"] = float(mean_absolute_error(lstm_y, lstm_p)) if len(lstm_p) else float("nan")
        cell_lstm_full = np.full(len(y_all), np.nan)
        cell_lstm_full[split_idx: split_idx + len(lstm_p)] = lstm_p
        preds[f"Cell_LSTM_{mode}"] = cell_lstm_full

        # Hybrid
        hyb_feat = build_legacy_survival_hybrid_features(cell_feat, surv_raw, xgb_raw, rolling_windows)
        X_hyb = hyb_feat.values.astype(np.float32)
        xgb_hyb = _fit_xgb(X_hyb[:split_idx], y_all[:split_idx], mode)
        hyb_xgb_pred = xgb_hyb.predict(X_hyb)
        results[f"Hyb_XGB_{mode}"] = float(mean_absolute_error(y_te, hyb_xgb_pred[split_idx:]))
        preds[f"Hyb_XGB_{mode}"] = hyb_xgb_pred

        if skip_lstm:
            lstm_hp, lstm_hy = np.array([]), np.array([])
        else:
            lstm_hp, lstm_hy = train_lstm_model(X_hyb, y_all, split_idx, seq_len, seed, device, cfg.model.lstm)
        results[f"Hyb_LSTM_{mode}"] = float(mean_absolute_error(lstm_hy, lstm_hp)) if len(lstm_hp) else float("nan")
        hyb_lstm_full = np.full(len(y_all), np.nan)
        hyb_lstm_full[split_idx: split_idx + len(lstm_hp)] = lstm_hp
        preds[f"Hyb_LSTM_{mode}"] = hyb_lstm_full

        if verbose:
            print(f"  Mode {mode} ({time.time()-t2:.0f}s, {len(bs_features)} bs feats): "
                  f"PerUE_Surv={results[f'PerUE_Surv_{mode}']:.2f}  "
                  f"PerUE_XGB={results[f'PerUE_XGB_{mode}']:.2f}  "
                  f"Cell_XGB={results[f'Cell_XGB_{mode}']:.2f}  "
                  f"Cell_LSTM={results[f'Cell_LSTM_{mode}']:.2f}  "
                  f"Hyb_XGB={results[f'Hyb_XGB_{mode}']:.2f}  "
                  f"Hyb_LSTM={results[f'Hyb_LSTM_{mode}']:.2f}")

    if verbose:
        print(f"  naive_A(n_conn) = {naive_a:.2f}   naive_B(n_conn_b) = {naive_b:.2f}")

    def _chash(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
    def _dfhash(d):
        return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()[:16]
    # Split PHYSICAL truth from OBSERVATION so that observation-only sweeps (DRX/FN/FP)
    # and S-TMSI can be proven not to change the physical cell trajectory:
    #   - physical_ue_events excludes s_tmsi_realloc (an observation event, not truth)
    #   - truth_dataframe is hashed BEFORE obs_b.annotate; mode_b_dataframe after.
    _PHYS = ("gt_enter", "connect", "disconnect", "gt_exit")
    physical_events = {pid: [(t, et) for t, et in evs if et in _PHYS]
                       for pid, evs in sim.ue_events.items()}
    stmsi_events = {pid: [(t, et) for t, et in evs if et == "s_tmsi_realloc"]
                    for pid, evs in sim.ue_events.items()}
    truth_hashes = {
        # physical truth (invariant to observation noise and S-TMSI)
        "truth_dataframe":    _dfhash(truth_df),
        "physical_ue_events": _chash(physical_events),
        "ue_meta":            _chash(sim.ue_meta),
        # observation-side (expected to change with noise / S-TMSI settings)
        "mode_b_dataframe":   _dfhash(df),
        "stmsi_events":       _chash(stmsi_events),
    }
    extras = {"trace": df, "manifests": manifests, "truth_hashes": truth_hashes}
    return results, preds, extras