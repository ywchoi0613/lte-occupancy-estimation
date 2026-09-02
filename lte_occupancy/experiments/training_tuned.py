"""
experiments/training_tuned.py — TUNED six-model robustness pipeline.

Evaluates the SAME six models the tuned headline protocol reports (no Survival):
    Cell_XGB   PerUE_XGB   Hybrid_XGB
    Cell_LSTM  PerUE_LSTM  Hybrid_LSTM
under a given scenario (scale / service-mix / S-TMSI / RRC-timer via cfg), reusing the
best hyperparameters selected by Optuna (default: an equal-budget study).

Robustness semantics (do NOT re-tune per scenario):
  * hyperparameters are FIXED to the tuned study's best params (Mode A uses A's params,
    Mode B uses B's) — each scenario measures how the CHOSEN model tolerates the change,
    not how a freshly re-optimized model would do.
  * per scenario/seed the models are RE-TRAINED on that scenario's first `train_ratio`
    (default 0.8) and evaluated on the last 20%.
  * the hybrid keeps the tuned protocol's 2-stage design: expanding-window OOF base
    predictions inside the first 80% train the hybrid, the base models are refit on the
    full 80%, and their full-train predictions over the last 20% feed the hybrid — i.e.
    the same logic as tune_runner.final_eval, only with fixed hyperparameters.

This module depends ONLY on the stable public `tuning` / `fusion` helpers (not on any
tune_runner private function), so it will not silently break if tune_runner is edited.
The small base-pred / hybrid-fit helpers below mirror tune_runner.final_eval; keep them in
sync if that logic changes.

Return contract matches the legacy runner: (results, preds, extras).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
try:
    import torch
    _TORCH = True
except Exception:                      # torch-less box: XGB family + param loading still work
    torch = None
    _TORCH = False
from sklearn.metrics import mean_absolute_error

from ..config.schema import ExperimentConfig
from . import tuning as TU
from ..estimation import fusion as FU

MODES = ("A", "B")
# (family_key, display_suffix, cell_branch, perue_branch, hybrid_branch)
FAMILIES = (("xgb", "XGB", "cell_xgb", "perue_xgb", "hybrid_xgb"),
            ("lstm", "LSTM", "cell_lstm", "perue_lstm", "hybrid_lstm"))
BRANCHES = ("cell_xgb", "perue_xgb", "cell_lstm", "perue_lstm", "hybrid_xgb", "hybrid_lstm")


# --------------------------------------------------------------------------------------
# best-param loading (fail-loud; comparison-aware)
# --------------------------------------------------------------------------------------
def load_best_params(params_dir, study_prefix: str, branch: str, mode: str,
                     comparison: str, required: bool = True):
    """Load best params for (branch, mode) from a tuned run directory.

    equal_budget studies write PER-MODE files (tag = mode); controlled studies write one
    SHARED file (tag = 'AB'). If the file is missing where the requested comparison expects
    it but present under the OTHER comparison's tag, raise a precise mismatch error (guards
    the "params are controlled but CLI said equal_budget" case). No default fallback."""
    pdir = Path(params_dir)
    if comparison == "equal_budget":
        want, other = mode, "".join(MODES)
    elif comparison == "controlled":
        want, other = "".join(MODES), mode
    else:
        raise ValueError(f"unknown comparison {comparison!r}")
    want_p = pdir / f"best_{study_prefix}_{branch}_{want}.json"
    if want_p.exists():
        return dict(json.load(open(want_p))["params"])
    other_p = pdir / f"best_{study_prefix}_{branch}_{other}.json"
    if other_p.exists():
        wrong = "controlled" if comparison == "equal_budget" else "equal_budget"
        raise FileNotFoundError(
            f"Comparison mismatch: --comparison {comparison} expects {want_p.name}, but "
            f"only {other_p.name} exists in {params_dir}. Those params look like the "
            f"'{wrong}' run — pass matching --comparison / --params-dir / --study-prefix.")
    if required:
        raise FileNotFoundError(
            f"Missing tuned params {want_p} (branch={branch}, mode={mode}, "
            f"comparison={comparison}). Run the tuning stage first; no fallback in tuned "
            f"robustness.")
    return None


def _resolve_param_path(params_dir, study_prefix, branch, mode, comparison) -> Path:
    tag = mode if comparison == "equal_budget" else "".join(MODES)
    return Path(params_dir) / f"best_{study_prefix}_{branch}_{tag}.json"


def _branch_prefix(branch: str, study_prefix: str, base_study_prefix: str | None) -> str:
    """Hybrids come from study_prefix; the four base branches may come from a different
    prefix (see tune_runner._prefix_for): re-tuning the hybrids on calibrated bases creates
    new hybrid studies while the base params stay exactly as tuned."""
    if branch in ("hybrid_xgb", "hybrid_lstm"):
        return study_prefix
    return base_study_prefix or study_prefix


def _load_and_check_params(params_dir, study_prefix, modes, comparison,
                           base_study_prefix=None, branches=None):
    """Load the FULL six-branch param set for every mode. Because all six are required,
    this enforces fail-loud on any missing base/hybrid JSON and on a partial LSTM family.
    Also returns the resolved best-JSON file paths (for file-level provenance)."""
    sel, paths = {}, {}
    for mode in modes:
        sel[mode] = {}
        for br in (branches if branches is not None else BRANCHES):
            pref = _branch_prefix(br, study_prefix, base_study_prefix)
            sel[mode][br] = load_best_params(params_dir, pref, br, mode, comparison,
                                             required=True)
            paths[f"{mode}/{br}"] = str(_resolve_param_path(params_dir, pref, br, mode,
                                                            comparison))
    return sel, paths


# The hybrid provenance guard used to be duplicated here and in tune_runner, and the two
# copies drifted: both skipped a study whose run_meta predates the perue_calib field, which
# is exactly the raw-tuned-hybrid-reported-as-calibrated case they were written to stop.
# TU.check_hybrid_provenance is now the single implementation; see tuning.py.


# --------------------------------------------------------------------------------------
# base predictions + hybrid (mirror of tune_runner.final_eval; fusion/tuning public API)
# --------------------------------------------------------------------------------------
def _base_preds(ds, mode, family, cell_params, perue_params, device, n_folds=4,
                perue_calib: str = "none", calib_scope: str = "full"):
    """Full-train + expanding-window OOF base predictions and the fusion frames, for the
    XGB base pair or the LSTM base pair. Mirrors tune_runner.base_preds_for.

    Calibration (fit on the OOF per-UE predictions inside the train region, applied to the
    held-out slice) mirrors tune_runner's ablation variants exactly:
        perue_calib="none"                    -> variant R_raw
        perue_calib=<kind>, scope="perue_only"-> variant S_perue_cal: the reported per-UE
            estimator is calibrated, the hybrid keeps RAW inputs (its tuned params stay valid)
        perue_calib=<kind>, scope="full"      -> variant C_full_cal: the hybrid is fit on
            calibrated bases too (its params should have been re-tuned with --perue-calib)
    The returned "perue_report" is what the standalone PerUE row reports; "full_perue" is
    what the hybrid consumes — they differ only under scope="perue_only"."""
    y, T, te = ds["y"], ds["T"], ds["train_end"]
    if family == "xgb":
        full_cell = TU.predict_cell_xgb(ds[mode], y, te, cell_params, ds["seed"])
        full_perue = TU.predict_perue_xgb(ds[mode], T, te, perue_params, ds["seed"])
        oof_cell = FU.oof_cell_xgb(ds["df"], mode, ds[mode], y, te, cell_params, ds["seed"], n_folds)
        oof_perue = FU.oof_perue_xgb(ds["df"], mode, ds[mode], y, te, perue_params, ds["seed"], n_folds)
    elif family == "lstm":
        full_cell = TU.predict_cell_lstm(ds[mode], y, te, cell_params, ds["seed"], device)
        full_perue = TU.predict_perue_lstm(ds[mode], T, te, perue_params, ds["seed"], device)
        oof_cell = FU.oof_cell_lstm(ds["df"], mode, ds[mode], y, te, cell_params, ds["seed"], device, n_folds)
        oof_perue = FU.oof_perue_lstm(ds["df"], mode, ds[mode], y, te, perue_params, ds["seed"], device, n_folds)
    else:
        raise ValueError(f"unknown family {family!r}")
    b1 = int(round(te * 1 / n_folds))
    perue_report = full_perue
    if perue_calib != "none":
        if calib_scope not in ("perue_only", "full"):
            raise ValueError(f"unknown calib_scope {calib_scope!r}")
        cal = FU.fit_perue_calibrator(oof_perue, y, b1, te, perue_calib)
        perue_report = cal(full_perue)
        if calib_scope == "full":
            full_perue, oof_perue = perue_report, cal(oof_perue)
    Z_oof = FU.build_fusion_features(oof_cell, oof_perue, ds["df"], mode).values
    Z_full = FU.build_fusion_features(full_cell, full_perue, ds["df"], mode).values
    return dict(y=y, T=T, train_end=te, b1=b1, full_cell=full_cell, full_perue=full_perue,
                perue_report=perue_report, oof_cell=oof_cell, oof_perue=oof_perue,
                Z_oof=Z_oof, Z_full=Z_full)


def _fit_predict_hybrid(bp, hyb_params, family, te, te_sl, device, seed):
    """Refit the tuned hybrid on the OOF region and predict the test slice. fusion_mode
    (direct/residual/gated) comes from the tuned params. Mirrors _hybrid_from_cache."""
    params = dict(hyb_params)
    fusion_mode = params.pop("fusion_mode", "direct")
    b1 = bp["b1"]
    if family == "xgb":
        params.pop("residual", None)
        mdl = FU.fit_hybrid_xgb(bp["Z_oof"][b1:te], bp["y"][b1:te], bp["oof_cell"][b1:te],
                                bp["oof_perue"][b1:te], params, seed, fusion_mode)
        return FU.predict_hybrid_xgb(mdl, bp["Z_full"][te_sl], bp["full_cell"][te_sl],
                                     bp["full_perue"][te_sl], fusion_mode)
    full, _ = FU.train_lstm_regressor(
        Z_train=bp["Z_oof"], y_train=bp["y"], train_idx=range(b1, te),
        Z_eval=bp["Z_full"], y_eval=bp["y"], eval_idx=range(te_sl.start, te_sl.stop),
        device=device, seed=seed, train_forbid_boundaries=FU._fold_bounds(te, 4),
        eval_forbid_boundaries=(), fusion_mode=fusion_mode,
        base_cell_train=bp["oof_cell"], base_perue_train=bp["oof_perue"],
        base_cell_eval=bp["full_cell"], base_perue_eval=bp["full_perue"],
        **FU._lstm_reg_kwargs(params))
    return full[te_sl]


def _tail_mae(y, full_pred, sl) -> float:
    fp = np.asarray(full_pred, np.float64)[sl]
    m = np.isfinite(fp)
    if not m.any():
        return float("nan")
    return float(mean_absolute_error(np.asarray(y)[sl][m], fp[m]))


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------
def _build_extras(ds, manifests, params_dir, study_prefix, comparison, train_ratio, sel,
                  param_files, perue_calib="none", calib_scope="full",
                  base_study_prefix=None):
    sim, truth_df, df = ds["sim"], ds["truth_df"], ds["df"]

    def _chash(o):
        return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _dfhash(d):
        return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()[:16]

    def _filesha(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for ch in iter(lambda: f.read(65536), b""):
                    h.update(ch)
            return h.hexdigest()
        except Exception:
            return "unavailable"

    _PHYS = ("gt_enter", "connect", "disconnect", "gt_exit")
    physical = {pid: [(t, et) for t, et in evs if et in _PHYS]
                for pid, evs in sim.ue_events.items()}
    stmsi_ev = {pid: [(t, et) for t, et in evs if et == "s_tmsi_realloc"]
                for pid, evs in sim.ue_events.items()}
    truth_hashes = {
        "truth_dataframe": _dfhash(truth_df), "physical_ue_events": _chash(physical),
        "ue_meta": _chash(sim.ue_meta), "mode_b_dataframe": _dfhash(df),
        "stmsi_events": _chash(stmsi_ev),
    }
    param_dict_hashes = {k: _chash(sel[m][b]) for m in sel for b in sel[m] for k in [f"{m}/{b}"]}
    param_file_sha256 = {k: _filesha(v) for k, v in param_files.items()}
    return {"trace": df, "manifests": manifests, "truth_hashes": truth_hashes,
            "model_set": "tuned", "params_dir": str(params_dir), "study_prefix": study_prefix,
            "base_study_prefix": base_study_prefix or study_prefix,
            "comparison": comparison, "train_ratio": train_ratio,
            "perue_calib": perue_calib,
            "calib_scope": (calib_scope if perue_calib != "none" else "none"),
            "final_variant": ("R_raw" if perue_calib == "none" else
                              "S_perue_cal" if calib_scope == "perue_only" else "C_full_cal"),
            "parameter_hashes": param_dict_hashes, "selected_params": sel,
            "param_files": param_files, "param_file_sha256": param_file_sha256}


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------
def run_one_seed_tuned(seed: int, timer: int, modes, cfg: ExperimentConfig, device,
                       params_dir, study_prefix: str, comparison: str = "equal_budget",
                       train_ratio: float = 0.8, perue_calib: str = "none",
                       calib_scope: str = "full", base_study_prefix: str | None = None,
                       families=None, verbose: bool = True):
    np.random.seed(seed); random.seed(seed)
    if _TORCH:
        torch.manual_seed(seed)
    modes = tuple(modes)
    # Fold the sweep timer into cfg (mirror runner's effective config) so the simulator
    # uses the scenario timer; build_dataset reads it from cfg.simulation.rrc.
    cfg = replace(cfg, simulation=replace(cfg.simulation,
                  rrc=replace(cfg.simulation.rrc, inactivity_timer_s=float(timer))))

    # ---- fail-loud: validate the FULL tuned param set + calibration provenance ----
    TU.check_hybrid_provenance(
        params_dir, study_prefix, perue_calib=perue_calib, calib_scope=calib_scope,
        base_study_prefix=(base_study_prefix or study_prefix), comparison=comparison,
        context="this robustness run")
    fams = tuple(families) if families else tuple(f[0] for f in FAMILIES)
    unknown = set(fams) - {f[0] for f in FAMILIES}
    if unknown:
        raise ValueError(f"unknown estimator family/families {sorted(unknown)}; "
                         f"valid: {[f[0] for f in FAMILIES]}")
    active = tuple(f for f in FAMILIES if f[0] in fams)
    # Only the ACTIVE families' branches are required. Demanding all six would make an
    # XGB-only robustness unit fail on a missing LSTM JSON it never reads.
    need = tuple(b for _f, _F, cb, pb, hb in active for b in (cb, pb, hb))
    sel, param_files = _load_and_check_params(params_dir, study_prefix, modes, comparison,
                                              base_study_prefix, branches=need)

    # ---- scenario dataset: temporal (train_ratio, 0, 1-train_ratio) ----
    ratios = (float(train_ratio), 0.0, 1.0 - float(train_ratio))
    ds = TU.build_dataset(seed=seed, modes=modes, horizon=None, ratios=ratios, cfg=cfg,
                          return_provenance=True)
    df, y, T, te = ds["df"], ds["y"], ds["T"], ds["train_end"]
    te_sl = slice(te, T); y_te = y[te_sl]
    obs_a, obs_b = ds["obs_a"], ds["obs_b"]

    # Guard: the first OOF fold trains on [0, te/4); a slot-based LSTM window must fit.
    max_slot_seq = max((int(sel[m][b].get("seq_len", 0))
                        for m in modes for b in ("cell_lstm", "hybrid_lstm")
                        if b in sel[m]), default=0)
    if round(te / 4) < max_slot_seq:
        raise ValueError(
            f"train region too short for 4-fold OOF: train_end={te} (fold≈{round(te/4)} "
            f"slots) < max LSTM seq_len={max_slot_seq}. Use a longer horizon or larger "
            f"--train-ratio.")

    if verbose:
        print(f"\n[Seed {seed}, timer={timer}s, modes={list(modes)}] tuned "
              f"({study_prefix}/{comparison}, perue_calib={perue_calib}/{calib_scope})  "
              f"mean_present={df['n_present'].mean():.0f}  train_end={te}/{T}")

    naive_a = mean_absolute_error(y_te, obs_a.naive_connected(df)[te_sl])
    naive_b = mean_absolute_error(y_te, obs_b.naive_connected(df)[te_sl])
    # NMAE denominator is the TEST-slice occupancy (the window the MAE is computed on),
    # not the whole-trace mean: MAE grows with cell size, so the scale table needs a
    # normalized figure to be comparable across profiles.
    mean_present_test = float(np.mean(y_te))
    results = {"seed": seed, "timer": timer,
               "mean_present": float(df["n_present"].mean()),
               "mean_present_test": mean_present_test,
               "mean_connected": float(df["n_connected"].mean()),
               # naive_A / naive_B are the raw connected-UE counts of each mode, recorded
               # as a sanity check on the simulator rather than as a comparison baseline.
               # The paper does not report them: n_connected measures a different quantity
               # from n_present, so its error is set by the idle fraction of the scenario,
               # not by any estimator's skill, and it would only inflate the headline gap.
               "naive_A": float(naive_a), "naive_B": float(naive_b)}

    def _put(name: str, mae: float):
        """Record a model's test MAE and its NMAE = MAE / mean(n_present) on the test slice."""
        results[name] = float(mae)
        results[f"{name}_nmae"] = (float(mae) / mean_present_test
                                   if np.isfinite(mae) and mean_present_test > 0 else float("nan"))

    for _n, _v in (("naive_A", naive_a), ("naive_B", naive_b)):
        results[f"{_n}_nmae"] = (float(_v) / mean_present_test if mean_present_test > 0
                                 else float("nan"))
    preds = {"t": df["t"].values, "y": y, "n_connected": df["n_connected"].values,
             "n_connected_b": df["n_connected_b"].values, "split_idx": te}
    # legacy-compatible extras: idle count + per-intensity occupancy (only if present)
    for col in ["n_idle", "n_idle_low", "n_conn_low", "n_idle_medium", "n_conn_medium",
                "n_idle_high", "n_conn_high"]:
        if col in df.columns:
            preds[col] = df[col].values
    manifests = {}

    for mode in modes:
        observer = obs_a if mode == "A" else obs_b
        if mode == "B":     # observation-fidelity guard by COLUMN NAME (order-independent)
            cols = ds[mode]["X_cell_lstm_columns"]
            idx = cols.index("n_connected")
            if not np.allclose(ds[mode]["X_cell_lstm"][:, idx], df["n_connected_b"].values):
                raise RuntimeError("Mode B Cell-LSTM is reading a non-observable counter "
                                   "(the n_connected column must equal DRX-degraded "
                                   "n_connected_b).")
        raw_set = set(observer.feature_columns)
        xgb_cols = ds[mode]["X_cell_columns"]
        manifests[mode] = {
            "raw_features": list(observer.feature_columns), "n_raw": len(observer.feature_columns),
            "n_total": len(xgb_cols),          # legacy field (XGB cell feature bank size)
            "derived_features": [c for c in xgb_cols if c not in raw_set],
            "n_cell_xgb": int(ds[mode]["X_cell"].shape[1]),
            "n_cell_lstm": int(ds[mode]["X_cell_lstm"].shape[1]),
        }
        for family, FAM, cb, pb, hb in active:
            bp = _base_preds(ds, mode, family, sel[mode][cb], sel[mode][pb], device,
                             perue_calib=perue_calib, calib_scope=calib_scope)
            _put(f"Cell_{FAM}_{mode}", _tail_mae(y, bp["full_cell"], te_sl))
            # the standalone PerUE row reports `perue_report` (calibrated when requested);
            # under scope="perue_only" the hybrid still consumes the RAW `full_perue`.
            _put(f"PerUE_{FAM}_{mode}", _tail_mae(y, bp["perue_report"], te_sl))
            preds[f"Cell_{FAM}_{mode}"] = bp["full_cell"]
            preds[f"PerUE_{FAM}_{mode}"] = bp["perue_report"]
            hp = np.asarray(_fit_predict_hybrid(bp, sel[mode][hb], family, te, te_sl,
                                                device, seed), np.float64)
            fm = np.isfinite(hp)
            if not fm.any():
                raise RuntimeError(f"Hybrid_{FAM}_{mode} produced no finite test "
                                   f"predictions (seed {seed}).")
            _put(f"Hybrid_{FAM}_{mode}", float(mean_absolute_error(y_te[fm], hp[fm])))
            hyb_full = np.full(T, np.nan); hyb_full[te_sl] = hp
            preds[f"Hybrid_{FAM}_{mode}"] = hyb_full
        if verbose:
            shown = " | ".join(
                " ".join(f"{stem}_{FAM}={results[f'{stem}_{FAM}_{mode}']:.2f}"
                         for stem in ("Cell", "PerUE", "Hybrid"))
                for _f, FAM, _cb, _pb, _hb in active)
            print(f"  Mode {mode}: {shown}")

    results["families"] = ",".join(fams)     # provenance: which branches this row holds
    extras = _build_extras(ds, manifests, params_dir, study_prefix, comparison,
                           float(train_ratio), sel, param_files, perue_calib, calib_scope,
                           base_study_prefix)
    return results, preds, extras