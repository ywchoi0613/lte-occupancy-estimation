"""
experiments/tuning.py — Stage 1 of the tuned modeling protocol.

Adds a 60/20/20 TEMPORAL split and an Optuna scaffold that selects hyperparameters
on the VALIDATION slice only (never test). Survival is dropped: both per-UE branches
are classifiers (PerUE-XGB and, for the LSTM family, PerUE-LSTM).

There are now FOUR base branches, in two symmetric families:
    cell_xgb   perue_xgb     -> feed Hybrid-XGB
    cell_lstm  perue_lstm    -> feed Hybrid-LSTM

This module does not import torch at the top level, so the XGB branches still run on
a torch-free CPU box; the LSTM branches delegate to estimation.fusion (which imports
torch lazily) and therefore require torch + a GPU to be present when invoked.

Split (chronological):
    train = [0, train_end)        60%   — base models fit here
    val   = [train_end, val_end)  20%   — Optuna objective / model selection
    test  = [val_end, T)          20%   — touched ONCE at final eval (tune_runner)

Selection modes:
    "val"          -> objective = mean VALIDATION MAE over dev seeds   (paper headline)
    "test_oracle"  -> objective = mean TEST MAE (an OPTIMISTIC upper bound; DIAGNOSTIC
                      only, never the reported number). Exposed so the "what if we tuned
                      on test" gap can be quantified honestly alongside the val result.

Two comparison protocols (see build/aggregate helpers in tune_runner):
    controlled       -> identical params for A and B (objective averaged over both modes)
    equal_budget     -> A and B tuned separately with the same trial budget & search space
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from ..config.defaults import build_config
from ..simulation.engine import Sim
from ..observation.mode_a import ModeAObserver
from ..observation.mode_b import ModeBObserver, split_events_by_stmsi, install_stmsi_realloc_patch
from ..features.cell import build_cell_features, build_cell_lstm_features
from ..estimation.per_ue import build_perue_xgb, compute_per_ue, make_cell_context_fn
from ..estimation import fusion as FU     # torch imported lazily inside fusion


# --------------------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------------------
def temporal_split(total_slots: int, ratios=(0.6, 0.2, 0.2)):
    """Return (train_end, val_end). test = [val_end, total_slots). Exclusive bounds."""
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    train_end = int(round(total_slots * ratios[0]))
    val_end = int(round(total_slots * (ratios[0] + ratios[1])))
    return train_end, val_end


def eval_slice(train_end: int, val_end: int, total: int, selection: str) -> slice:
    """Evaluation window for the objective: val slice, or test slice for test_oracle."""
    if selection == "val":
        return slice(train_end, val_end)
    if selection == "test_oracle":
        return slice(val_end, total)
    raise ValueError(f"unknown selection {selection!r}")


def _masked_mae(y_true, y_pred) -> float:
    """MAE over finite predictions only (LSTM leaves NaN where no causal window exists).
    Returns a large penalty if the slice has no valid prediction, so Optuna avoids it."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_pred)
    if not m.any():
        return 1e9
    return float(mean_absolute_error(np.asarray(y_true)[m], y_pred[m]))


# survival is removed: per-UE branches are classifiers, so compute_per_ue gets a null survival.
def _null_survival(*_a, **_k):
    return 0.0


# --------------------------------------------------------------------------------------
# per-(seed,mode) dataset — built ONCE, reused across all Optuna trials
# --------------------------------------------------------------------------------------
def build_dataset(seed: int, modes=("A", "B"), horizon: int | None = None,
                  ratios=(0.6, 0.2, 0.2), cfg=None, return_provenance: bool = False) -> dict:
    """Simulate once and materialize, per mode, everything a trial needs:
    X_cell, labels y, per-UE events/meta, cell-context fn, and split indices.
    Feature thresholds and context use train_end (60%) ONLY.

    return_provenance=True additionally attaches the Sim object, both observers, and the
    pre-annotation truth dataframe (used by the tuned robustness runner to emit the same
    truth-hash / manifest provenance as the legacy runner). It is off by default so the
    tuning hot loop holds no extra state."""
    import random
    from dataclasses import replace
    if cfg is None:
        cfg = build_config()
    T = int(horizon or cfg.simulation.time.total_slots)
    # Honor --horizon by actually shortening the simulation (previously the Sim ran the
    # full cfg length and we truncated the dataframe — wasteful and inconsistent).
    sim_cfg = cfg.simulation
    if T != sim_cfg.time.total_slots:
        sim_cfg = replace(sim_cfg, time=replace(sim_cfg.time, total_slots=T))
    train_end, val_end = temporal_split(T, ratios)

    np.random.seed(seed); random.seed(seed)
    stmsi = cfg.observation.stmsi
    # S-TMSI realloc must be installed on Sim BEFORE run() so the simulator actually emits
    # s_tmsi_realloc events; split_events_by_stmsi alone (below) only consumes them. The
    # patch is a no-op unless sim._stmsi_realloc_mean is set, and reallocation uses an
    # isolated RNG stream, so physical trajectories are identical with/without it.
    if stmsi.enabled:
        install_stmsi_realloc_patch(Sim, stmsi.realloc_mean_slots)
    sim = Sim(seed=seed, cfg=sim_cfg, rrc_timer=sim_cfg.rrc.inactivity_timer_s)
    if stmsi.enabled:
        sim._stmsi_realloc_mean = stmsi.realloc_mean_slots
    df = sim.run()
    if stmsi.enabled and T >= stmsi.realloc_mean_slots:
        n_realloc = sum(et == "s_tmsi_realloc"
                        for evs in sim.ue_events.values() for _, et in evs)
        assert n_realloc > 0, (
            f"S-TMSI enabled and horizon T={T} >= realloc_mean={stmsi.realloc_mean_slots} "
            f"but no s_tmsi_realloc events were generated (patch not applied?).")
    df = df.iloc[:T].reset_index(drop=True) if len(df) > T else df
    assert len(df) == T, f"simulation length mismatch: {len(df)} != {T}"
    obs_a = ModeAObserver(cfg.observation)
    obs_b = ModeBObserver(cfg.observation, seed=seed)
    truth_df = df.copy() if return_provenance else None    # BEFORE observation columns
    df = obs_b.annotate(df)
    y = df["n_present"].values.astype(np.float32)

    base_events, base_meta = sim.ue_events, sim.ue_meta
    b_events, b_meta = obs_b.observe_ue_events(base_events, base_meta, total_time=T)
    if cfg.observation.stmsi.enabled:
        b_events, b_meta = split_events_by_stmsi(b_events, b_meta)

    wave_period = cfg.simulation.arrival.diurnal_period_slots
    rolling = cfg.simulation.time.rolling_windows
    out = {"seed": seed, "T": T, "train_end": train_end, "val_end": val_end, "y": y,
           "df": df}
    if return_provenance:
        out["sim"] = sim; out["obs_a"] = obs_a; out["obs_b"] = obs_b; out["truth_df"] = truth_df
    for mode in modes:
        observer = obs_a if mode == "A" else obs_b
        bs = observer.feature_columns
        ev, mt = (base_events, base_meta) if mode == "A" else (b_events, b_meta)
        ctx = make_cell_context_fn(df, mode=mode, train_end=train_end,
                                   wave_period=wave_period, feat=cfg.features)
        cell_feat = build_cell_features(df, mode, bs, train_end=train_end,
                                        wave_period=wave_period, rolling_windows=rolling,
                                        feat=cfg.features)
        # Cell-LSTM gets a SEPARATE compact, causal, threshold-free feature set (raw
        # observable counters + diff + short rolling mean/std) — an order of magnitude
        # smaller than the XGB bank. Passing `mode` lets it remap Mode B's n_connected to
        # the observable n_connected_b (no truth leakage).
        cell_feat_lstm = build_cell_lstm_features(df, mode, bs)
        out[mode] = {
            "X_cell": cell_feat.values.astype(np.float32),           # XGB cell branch
            "X_cell_lstm": cell_feat_lstm.values.astype(np.float32),  # LSTM cell branch
            "X_cell_columns": list(cell_feat.columns),
            "X_cell_lstm_columns": list(cell_feat_lstm.columns),
            "events": ev, "meta": mt, "ctx": ctx,
            "subsample": cfg.model.per_ue.subsample,
            "prune": cfg.model.per_ue.prune_elapsed,
            # ingredients for STRICT fold-wise OOF preprocessing (rebuilt per fold):
            "bs": bs, "rolling": rolling, "wave_period": wave_period, "feat": cfg.features,
        }
    return out


# --------------------------------------------------------------------------------------
# base-branch predictors (full-length prediction arrays)
# --------------------------------------------------------------------------------------
def predict_cell_xgb(ds_mode: dict, y: np.ndarray, train_end: int, params: dict, seed: int):
    X = ds_mode["X_cell"]
    m = XGBRegressor(**params, random_state=seed)
    m.fit(X[:train_end], y[:train_end])
    return m.predict(X)


def predict_perue_xgb(ds_mode: dict, T: int, train_end: int, clf_params: dict, seed: int):
    clf = build_perue_xgb(ds_mode["events"], ds_mode["meta"], train_end, seed=seed,
                          xgb_clf_params=clf_params, subsample=ds_mode["subsample"],
                          prune_elapsed=ds_mode["prune"], cell_context_fn=ds_mode["ctx"])
    _surv, xgb_pred = compute_per_ue(
        ds_mode["events"], ds_mode["meta"], _null_survival, {}, clf, T,
        subsample=ds_mode["subsample"], prune_elapsed=ds_mode["prune"],
        cell_context_fn=ds_mode["ctx"])
    return xgb_pred


def predict_cell_lstm(ds_mode: dict, y: np.ndarray, train_end: int, params: dict,
                      seed: int, device: str):
    """Full-length Cell-LSTM prediction: train on [0, train_end), predict over [0, T).
    Uses the COMPACT LSTM feature set. NaN where no causal window exists (series start)."""
    X = ds_mode["X_cell_lstm"]
    T = len(y)
    full, _ = FU.train_lstm_regressor(
        Z_train=X, y_train=y, train_idx=range(0, train_end),
        Z_eval=X, y_eval=y, eval_idx=range(0, T),
        device=device, seed=seed, train_forbid_boundaries=(),
        eval_forbid_boundaries=(), **FU._lstm_reg_kwargs(params))
    return full


def predict_perue_lstm(ds_mode: dict, T: int, train_end: int, params: dict,
                       seed: int, device: str):
    """Full-length PerUE-LSTM count (per-track sequence classifier, probs summed)."""
    return FU.perue_lstm_predict(
        ds_mode["events"], ds_mode["meta"], ds_mode["ctx"], total_time=T,
        train_end=train_end, params=params, seed=seed, device=device,
        subsample=ds_mode["subsample"], prune_elapsed=ds_mode["prune"])


# --------------------------------------------------------------------------------------
# Optuna search spaces
# --------------------------------------------------------------------------------------
def sample_xgb_reg(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        n_jobs=int(__import__("os").environ.get("LTE_XGB_JOBS", 4)),
        tree_method="hist", verbosity=0,
    )


def sample_xgb_clf(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        n_jobs=int(__import__("os").environ.get("LTE_XGB_JOBS", 4)),
        tree_method="hist", verbosity=0,
    )


def sample_cell_lstm(trial) -> dict:
    # seq_len is in SLOTS (cell features are per-slot).
    return dict(
        seq_len=trial.suggest_categorical("seq_len", [10, 30, 60, 120]),
        hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
        num_layers=trial.suggest_int("num_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        lr=trial.suggest_float("lr", 5e-4, 1e-2, log=True),
        epochs=trial.suggest_int("epochs", 15, 60, step=15),
    )


def sample_perue_lstm(trial) -> dict:
    # seq_len is in SAMPLING STEPS (one step = per_ue.subsample slots).
    return dict(
        seq_len=trial.suggest_categorical("seq_len", [4, 8, 12, 20]),
        hidden_size=trial.suggest_categorical("hidden_size", [32, 64]),
        num_layers=trial.suggest_int("num_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        lr=trial.suggest_float("lr", 5e-4, 1e-2, log=True),
        epochs=trial.suggest_int("epochs", 15, 45, step=15),
    )


# --------------------------------------------------------------------------------------
# objectives (return mean MAE over dev-seed datasets, on the selection slice).
# All base objectives take `device`; the XGB objectives ignore it.
# --------------------------------------------------------------------------------------
def objective_cell_xgb(trial, datasets, modes, selection, device=None):
    params = sample_xgb_reg(trial)
    maes = []
    for ds in datasets:
        for mode in modes:
            pred = predict_cell_xgb(ds[mode], ds["y"], ds["train_end"], params, ds["seed"])
            sl = eval_slice(ds["train_end"], ds["val_end"], ds["T"], selection)
            maes.append(_masked_mae(ds["y"][sl], pred[sl]))
    return float(np.mean(maes))


def objective_perue_xgb(trial, datasets, modes, selection, device=None):
    params = sample_xgb_clf(trial)
    maes = []
    for ds in datasets:
        for mode in modes:
            pred = predict_perue_xgb(ds[mode], ds["T"], ds["train_end"], params, ds["seed"])
            sl = eval_slice(ds["train_end"], ds["val_end"], ds["T"], selection)
            maes.append(_masked_mae(ds["y"][sl], pred[sl]))
    return float(np.mean(maes))


def objective_cell_lstm(trial, datasets, modes, selection, device):
    params = sample_cell_lstm(trial)
    maes = []
    for ds in datasets:
        for mode in modes:
            pred = predict_cell_lstm(ds[mode], ds["y"], ds["train_end"], params,
                                     ds["seed"], device)
            sl = eval_slice(ds["train_end"], ds["val_end"], ds["T"], selection)
            maes.append(_masked_mae(ds["y"][sl], pred[sl]))
    return float(np.mean(maes))


def objective_perue_lstm(trial, datasets, modes, selection, device):
    params = sample_perue_lstm(trial)
    maes = []
    for ds in datasets:
        for mode in modes:
            pred = predict_perue_lstm(ds[mode], ds["T"], ds["train_end"], params,
                                      ds["seed"], device)
            sl = eval_slice(ds["train_end"], ds["val_end"], ds["T"], selection)
            maes.append(_masked_mae(ds["y"][sl], pred[sl]))
    return float(np.mean(maes))


OBJECTIVES = {"cell_xgb": objective_cell_xgb, "perue_xgb": objective_perue_xgb,
              "cell_lstm": objective_cell_lstm, "perue_lstm": objective_perue_lstm}
SAMPLERS = {"cell_xgb": sample_xgb_reg, "perue_xgb": sample_xgb_clf,
            "cell_lstm": sample_cell_lstm, "perue_lstm": sample_perue_lstm}
PREDICTORS = {"cell_xgb": predict_cell_xgb, "perue_xgb": predict_perue_xgb,
              "cell_lstm": predict_cell_lstm, "perue_lstm": predict_perue_lstm}

# which base branches feed which hybrid family
BASE_FOR_HYBRID = {"xgb": ("cell_xgb", "perue_xgb"), "lstm": ("cell_lstm", "perue_lstm")}
HYBRID_BRANCHES = ("hybrid_xgb", "hybrid_lstm")

# calibration variant -> (what the per-UE branch reports, what the hybrid is FED)
#   R_raw       raw per-UE reported,        hybrid fed RAW bases
#   S_perue_cal calibrated per-UE reported, hybrid fed RAW bases
#   C_full_cal  calibrated per-UE reported, hybrid fed CALIBRATED bases
VARIANT_SCOPE = {"R_raw": "none", "S_perue_cal": "perue_only", "C_full_cal": "full"}


# ======================================================================================
# tuned-hybrid provenance guard
# ======================================================================================
# SINGLE SOURCE OF TRUTH for both consumers of tuned hybrid params:
#   * the headline path   (tune_runner.final_eval)
#   * the robustness path (training_tuned.run_one_seed_tuned)
# Both used to carry their own copy of this check, which is how they drifted apart.
#
# Every worker of one hybrid study must agree on the protocol it was tuned under, and that
# protocol must equal what the consuming run is about to do. A best_*.json is the winner of
# ONE Optuna study; if its workers explored under different protocols, the "winner" is the
# argmax over a mixture and means nothing.
_PROVENANCE_KEYS = ("perue_calib", "base_study_prefix", "comparison", "selection",
                    "dev_seeds")
_MISSING = "<not recorded>"


def expected_hybrid_calib(perue_calib: str, calib_scope: str) -> str:
    """The per-UE calibration the hybrid's TUNING must have used, given the variant.

    Variant C (perue_calib=<kind>, scope='full') feeds the hybrid CALIBRATED bases, so its
    hyperparameters — including the tuned ``fusion_mode``, which selects a count-scale
    convex blend w*cell + (1-w)*perUE — must have been selected on calibrated bases.
    Variants R and S leave the hybrid inputs RAW, so their hybrids must have been tuned on
    raw. BOTH directions are errors: raw-tuned hybrids under C, and calibrated-tuned
    hybrids under R/S, each report a model tuned for other inputs as if it were tuned.
    """
    return perue_calib if (perue_calib != "none" and calib_scope == "full") else "none"


def _norm(v):
    """Hashable, comparable form (dev_seeds is a list in the JSON)."""
    return tuple(v) if isinstance(v, list) else v


def _recorded(meta: dict, key: str, study_prefix: str):
    """What a tuning worker recorded for `key`.

    Two keys carry a DOCUMENTED default for studies tuned before the field existed:

      perue_calib       -> "none".  The field was added in the same change as the
                           --perue-calib flag, so a study without it cannot have been
                           tuned on calibrated bases. This default only ever TIGHTENS the
                           check (it can make variant C fail, never pass), so inferring it
                           is safe; the reverse — inferring "isotonic" — never happens.
      base_study_prefix -> study_prefix.  Mirrors the writer's own
                           `args.base_study_prefix or args.study_prefix`.

    Anything else missing is reported as absent and fails the comparison rather than being
    guessed at.
    """
    if key == "perue_calib":
        return meta.get(key, "none")
    if key == "base_study_prefix":
        return meta.get(key) or study_prefix
    return meta.get(key, _MISSING)


def _hybrid_artifacts(params_dir, study_prefix: str, branch: str):
    """(tuned param files, [(filename, meta dict)]) for one hybrid branch of one study.

    The branch is part of both globs, so prefix 's3c5' never picks up 's3c5_cal' artifacts.
    """
    d = Path(params_dir)
    best = sorted(d.glob(f"best_{study_prefix}_{branch}_*.json"))     # _A/_B or _AB
    metas = []
    for p in sorted(d.glob(f"run_meta_{study_prefix}_{branch}_w*.json")):
        try:
            metas.append((p.name, json.loads(p.read_text())))
        except Exception as e:
            raise SystemExit(f"unreadable tuning provenance {p}: {e}")
    return best, metas


def _show(vals) -> str:
    s = sorted(str(v) for v in vals)
    return s[0] if len(s) == 1 else "{" + ", ".join(s) + "}"


def _remedy(study_prefix: str, want: dict) -> str:
    kind = want["perue_calib"]
    if kind == "none":
        return ("Variants R_raw/S_perue_cal feed the hybrid RAW bases, so point\n"
                "--study-prefix at the RAW hybrid study (the one tuned without\n"
                "--perue-calib), not at a calibrated one.")
    base = want.get("base_study_prefix", study_prefix)
    new = f"{base}_cal"
    return (f"Re-tune ONLY the two hybrid branches on {kind}-calibrated bases under a NEW\n"
            f"study prefix (the four base branches are reused as-is, nothing is copied):\n"
            f'    BRANCHES="hybrid_xgb hybrid_lstm" PREFIX={new} BASE_PREFIX={base} \\\n'
            f"      PERUE_CALIB={kind} CALIB_KIND={kind} FINAL_VARIANT=C_full_cal \\\n"
            f"      SKIP_FINAL=1 bash run_tuning_parallel.sh\n"
            f"then re-run this command with --study-prefix {new} "
            f"--base-study-prefix {base}.")


def check_hybrid_provenance(params_dir, study_prefix: str, *, perue_calib: str,
                            calib_scope: str, base_study_prefix: str | None = None,
                            comparison: str | None = None, selection: str | None = None,
                            context: str = "this run", allow_mismatch: bool = False):
    """Refuse to consume tuned hybrid params that were not selected for THESE inputs.

    Three independent failure modes, checked per hybrid branch:

      1. PROVENANCE PRESENT     tuned params sitting next to no run_meta have unknown
                                provenance. Silently treating an old raw study as
                                calibrated is the exact mistake this guard exists to
                                prevent, so missing metadata is an error, not a shrug.
      2. PROVENANCE CONSISTENT  all workers of one study agree on _PROVENANCE_KEYS.
      3. PROVENANCE MATCHES     the keys the caller pins equal what the workers recorded.

    Branches with no tuned params at all are skipped — there is nothing to misuse, and the
    param loader raises a better-targeted error downstream.

    allow_mismatch downgrades to a warning; it exists for tests. Paper call sites do NOT
    wire it to --allow-default-fallback: a missing-param fallback is a smoke-test
    convenience, whereas reporting a model tuned for other inputs is a result error.
    """
    want = {"perue_calib": expected_hybrid_calib(perue_calib, calib_scope)}
    if base_study_prefix is not None:
        want["base_study_prefix"] = base_study_prefix
    if comparison is not None:
        want["comparison"] = comparison
    if selection is not None:
        want["selection"] = selection

    problems = []
    for branch in HYBRID_BRANCHES:
        best, metas = _hybrid_artifacts(params_dir, study_prefix, branch)
        if not best:
            continue
        if not metas:
            problems.append(
                f"{branch}: {len(best)} tuned parameter file(s) exist under study "
                f"'{study_prefix}' but no run_meta_{study_prefix}_{branch}_w*.json beside "
                f"them. Their provenance is unknown, so they can be assumed neither raw "
                f"nor calibrated. Re-tune the branch, or delete the orphaned params.")
            continue
        for key in _PROVENANCE_KEYS:                       # (2) workers agree
            vals = {_norm(_recorded(m, key, study_prefix)) for _n, m in metas}
            if len(vals) > 1:
                problems.append(
                    f"{branch}: its {len(metas)} workers disagree on {key} = {_show(vals)}. "
                    f"best_{study_prefix}_{branch}_*.json is the winner of ONE study, so a "
                    f"mixed protocol makes that winner meaningless.")
        for key, exp in want.items():                      # (3) matches this run
            got = {_norm(_recorded(m, key, study_prefix)) for _n, m in metas}
            if got != {_norm(exp)}:
                problems.append(
                    f"{branch}: tuned with {key}={_show(got)}, but {context} needs "
                    f"{key}={exp!r}.")

    if not problems:
        return
    msg = ("tuned hybrid provenance mismatch\n"
           f"  params_dir : {params_dir}\n"
           f"  study      : {study_prefix}\n"
           f"  consumer   : {context}\n"
           + "\n".join(f"  - {p}" for p in problems)
           + "\n\n" + _remedy(study_prefix, want))
    if allow_mismatch:
        print("WARNING: " + msg)
        return
    raise SystemExit(msg)