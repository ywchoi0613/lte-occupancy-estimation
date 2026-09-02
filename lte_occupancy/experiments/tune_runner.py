"""
experiments/tune_runner.py — Stage 1-4 orchestration for the tuned protocol.

Pipeline (all selection on VALIDATION; TEST touched once at the end):
  Stage 1  base branches  : tune Cell-XGB, PerUE-XGB, Cell-LSTM, PerUE-LSTM per mode
  Stage 2/3 hybrids       : fix best base params -> OOF base preds -> tune Hybrid-XGB
                            (on the XGB base pair) and Hybrid-LSTM (on the LSTM base pair)
  Stage 4  final eval     : refit best configs, evaluate on TEST with held-out TEST seeds,
                            report per-model MAE (up to six models) and the paired A-B interval

Symmetric hybrids (this is the finalized design):
    Cell-XGB  + PerUE-XGB   -> Hybrid-XGB
    Cell-LSTM + PerUE-LSTM  -> Hybrid-LSTM

Reproducibility: each study uses a seeded multivariate TPE sampler. With N async
workers sharing one journal the exact trial *sequence* is not bit-reproducible, so the
sampler seed, Optuna version, worker count, and torch/device info are written to
run_meta_*.json alongside the results.

Parallelism: Optuna studies use one shared journal storage, so N worker processes (one
per GPU) add trials to the same study concurrently. GPUs matter for the LSTM branches;
XGB branches run on CPU but still parallelize across workers.

Examples
  # base branches (XGB family), 60 trials, dev seeds, validation-selected
  python -m lte_occupancy.experiments.tune_runner --branch cell_xgb  --trials 60 \
      --dev-seeds 7 13 42 --storage journal:tune.journal --out tune_out
  # base branches (LSTM family)
  python -m lte_occupancy.experiments.tune_runner --branch cell_lstm --trials 60 ...
  # hybrids (load best base params of the matching family from tune_out)
  python -m lte_occupancy.experiments.tune_runner --branch hybrid_lstm --trials 60 ...
  # final test eval with held-out seeds + paired CI (reports every tuned model)
  python -m lte_occupancy.experiments.tune_runner --final --test-seeds 101 102 ... --out tune_out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from . import tuning as TU
from ..estimation import fusion as FU

MODES = ("A", "B")
HYBRID_BASE = {"hybrid_xgb": "xgb", "hybrid_lstm": "lstm"}


# --------------------------------------------------------------------------------------
# small io helpers
# --------------------------------------------------------------------------------------
def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically (tmp + os.replace) so concurrent workers never leave a
    half-written file for a long-running study."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _load_json(p):
    return json.load(open(p)) if Path(p).exists() else None


def _std(values) -> float:
    """SAMPLE standard deviation over seeds (ddof=1).

    Seeds are a sample of the scenario's randomness, not the population, so every reported
    spread in this project uses ddof=1. runner.py's summary.csv uses pandas .std() which is
    already ddof=1; tune_runner and make_paper_tables must match it, otherwise the same
    experiment prints two different '+/-' values."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


# --------------------------------------------------------------------------------------
# dataset cache (deterministic from seed; each worker builds independently)
# --------------------------------------------------------------------------------------
def build_datasets(seeds, horizon, modes, ratios=(0.6, 0.2, 0.2)):
    return [TU.build_dataset(seed=s, modes=modes, horizon=horizon, ratios=ratios) for s in seeds]


# --------------------------------------------------------------------------------------
# base-pred cache for hybrid (uses FIXED best base params of the requested family)
# --------------------------------------------------------------------------------------
def base_preds_for(ds, mode, base_kind, cell_params, perue_params, device, n_folds=4,
                   perue_calib: str = "none"):
    """Returns dict with full-train + OOF base preds and fusion frames for one dataset,
    for either the XGB base pair (Cell-XGB/PerUE-XGB) or the LSTM base pair
    (Cell-LSTM/PerUE-LSTM).

    perue_calib != "none" applies an OOF-fitted raw->count map to the per-UE branch BEFORE
    the fusion frames are built, i.e. the hybrid is tuned/fit on CALIBRATED bases (variant
    C). Leave it "none" (default) for the raw protocol; the calibration ablation decides."""
    y, T, te = ds["y"], ds["T"], ds["train_end"]
    if base_kind == "xgb":
        full_cell = TU.predict_cell_xgb(ds[mode], y, te, cell_params, ds["seed"])
        full_perue = TU.predict_perue_xgb(ds[mode], T, te, perue_params, ds["seed"])
        oof_cell = FU.oof_cell_xgb(ds["df"], mode, ds[mode], y, te, cell_params, ds["seed"], n_folds)
        oof_perue = FU.oof_perue_xgb(ds["df"], mode, ds[mode], y, te, perue_params, ds["seed"], n_folds)
    elif base_kind == "lstm":
        full_cell = TU.predict_cell_lstm(ds[mode], y, te, cell_params, ds["seed"], device)
        full_perue = TU.predict_perue_lstm(ds[mode], T, te, perue_params, ds["seed"], device)
        oof_cell = FU.oof_cell_lstm(ds["df"], mode, ds[mode], y, te, cell_params, ds["seed"], device, n_folds)
        oof_perue = FU.oof_perue_lstm(ds["df"], mode, ds[mode], y, te, perue_params, ds["seed"], device, n_folds)
    else:
        raise ValueError(f"unknown base_kind {base_kind!r}")
    b1 = int(round(te * 1 / n_folds))
    if perue_calib != "none":
        cal = FU.fit_perue_calibrator(oof_perue, y, b1, te, perue_calib)
        full_perue, oof_perue = cal(full_perue), cal(oof_perue)
    Z_oof = FU.build_fusion_features(oof_cell, oof_perue, ds["df"], mode).values
    Z_full = FU.build_fusion_features(full_cell, full_perue, ds["df"], mode).values
    # oof_perue / full_perue are kept (in addition to being folded into Z) so the GATED
    # fusion family can form its convex blend on the base predictions.
    return dict(y=y, T=T, train_end=te, val_end=ds["val_end"], b1=b1,
                full_cell=full_cell, full_perue=full_perue,
                oof_cell=oof_cell, oof_perue=oof_perue,
                Z_oof=Z_oof, Z_full=Z_full)


# --------------------------------------------------------------------------------------
# calibration variants (one definition used by the ablation AND by the final headline)
# --------------------------------------------------------------------------------------
#   R_raw       raw per-UE reported; hybrid fed raw bases            (current protocol)
#   S_perue_cal calibrated per-UE reported; hybrid still fed RAW bases
#               -> hybrid unchanged, so the tuned hybrid params stay valid
#   C_full_cal  calibrated per-UE reported AND fed to the hybrid
#               -> the hybrid sees inputs it was NOT tuned on; to claim C as the final
#                  protocol the two hybrid branches must be RE-TUNED on calibrated bases
#                  (tune with --perue-calib <kind> under a new --study-prefix, e.g. s3c5_cal)
VARIANTS = ("R_raw", "S_perue_cal", "C_full_cal")
# variant -> calibration scope. Defined in tuning.py so the headline path, the
# robustness path and the provenance guard all read ONE map.
VARIANT_SCOPE = TU.VARIANT_SCOPE


def _apply_calib_to_bp(bp, cal, df, mode):
    """Base-pred dict with the per-UE branch (full + OOF) mapped through `cal` and the
    fusion frames rebuilt from the calibrated per-UE predictions. The cell branch is
    untouched, so cell results are identical across calibration variants."""
    out = dict(bp)
    out["full_perue"] = cal(bp["full_perue"])
    out["oof_perue"] = cal(bp["oof_perue"])
    out["Z_oof"] = FU.build_fusion_features(bp["oof_cell"], out["oof_perue"], df, mode).values
    out["Z_full"] = FU.build_fusion_features(bp["full_cell"], out["full_perue"], df, mode).values
    return out


def variant_bp(bp, df, mode, y, te, variant: str, kind: str):
    """(bp_for_hybrid, perue_pred_to_report) for one calibration variant. The cell branch
    is never touched, so cell results are identical across variants."""
    if variant == "R_raw":
        return bp, bp["full_perue"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {VARIANTS})")
    cal = FU.fit_perue_calibrator(bp["oof_perue"], y, bp["b1"], te, kind)
    bp_c = _apply_calib_to_bp(bp, cal, df, mode)
    if variant == "S_perue_cal":
        return bp, bp_c["full_perue"]        # hybrid keeps RAW inputs
    return bp_c, bp_c["full_perue"]          # C_full_cal


# --------------------------------------------------------------------------------------
# Optuna objectives
# --------------------------------------------------------------------------------------
def obj_base(trial, datasets, modes, selection, branch, device):
    return TU.OBJECTIVES[branch](trial, datasets, modes, selection, device)


def obj_hybrid_xgb(trial, bp_cache, modes, selection):
    params = FU.sample_hybrid_xgb(trial)
    fusion_mode = trial.suggest_categorical("fusion_mode", list(FU.FUSION_MODES))
    maes = []
    for per_mode in bp_cache:
        for mode in modes:
            bp = per_mode[mode]
            te, ve, T, b1 = bp["train_end"], bp["val_end"], bp["T"], bp["b1"]
            Ztr, ytr = bp["Z_oof"][b1:te], bp["y"][b1:te]
            cell_tr, perue_tr = bp["oof_cell"][b1:te], bp["oof_perue"][b1:te]
            sl = slice(te, ve) if selection == "val" else slice(ve, T)
            mdl = FU.fit_hybrid_xgb(Ztr, ytr, cell_tr, perue_tr, params, 0, fusion_mode)
            pred = FU.predict_hybrid_xgb(mdl, bp["Z_full"][sl], bp["full_cell"][sl],
                                         bp["full_perue"][sl], fusion_mode)
            maes.append(mean_absolute_error(bp["y"][sl], pred))
    return float(np.mean(maes))


def obj_hybrid_lstm(trial, bp_cache, modes, selection, device):
    hp = FU.sample_hybrid_lstm(trial)
    fusion_mode = trial.suggest_categorical("fusion_mode", list(FU.FUSION_MODES))
    maes = []
    for per_mode in bp_cache:
        for mode in modes:
            bp = per_mode[mode]
            te, ve, T, b1 = bp["train_end"], bp["val_end"], bp["T"], bp["b1"]
            sl0, sl1 = (te, ve) if selection == "val" else (ve, T)
            # train windows over OOF region [b1,te); eval windows over the selection slice.
            # y_eval is FULL-length; eval_idx are absolute indices into it. residual/gated
            # blend on the raw base preds (OOF for train ends, full for eval ends).
            _full, mae = FU.train_lstm_regressor(
                Z_train=bp["Z_oof"], y_train=bp["y"], train_idx=range(b1, te),
                Z_eval=bp["Z_full"], y_eval=bp["y"], eval_idx=range(sl0, sl1),
                device=device, train_forbid_boundaries=FU._fold_bounds(te, 4),
                eval_forbid_boundaries=(), fusion_mode=fusion_mode,
                base_cell_train=bp["oof_cell"], base_perue_train=bp["oof_perue"],
                base_cell_eval=bp["full_cell"], base_perue_eval=bp["full_perue"],
                **FU._lstm_reg_kwargs(hp))
            maes.append(mae)
    return float(np.mean(maes))


# --------------------------------------------------------------------------------------
# study driver
# --------------------------------------------------------------------------------------
def _make_storage(spec):
    """`journal:/path` -> Optuna JournalStorage (robust for concurrent file access).
    Any other string (e.g. sqlite:///x.db) is passed through unchanged."""
    if spec and spec.startswith("journal:"):
        import optuna
        path = spec[len("journal:"):]
        try:                                   # optuna >= 4
            from optuna.storages.journal import JournalFileBackend
            return optuna.storages.JournalStorage(JournalFileBackend(path))
        except ImportError:                    # older layout only
            from optuna.storages import JournalFileStorage
            return optuna.storages.JournalStorage(JournalFileStorage(path))
    return spec


def _make_sampler(seed: int):
    """Seeded multivariate TPE. NOTE: with several async workers on a shared study the
    global trial *order* still varies run-to-run; the seed makes each worker's own
    proposals deterministic and is recorded in the run metadata for provenance."""
    import optuna
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)


def _effective_seed(args) -> int:
    """Per-worker sampler seed = base + worker index. Distinct workers therefore explore
    with DIFFERENT TPE proposals (no duplicated initial trials), while A and B see the same
    per-worker seed set so the comparison stays fair."""
    return int(args.sampler_seed) + int(getattr(args, "worker_index", 0))


def _write_run_meta(out: Path, args, device, kind: str):
    import optuna
    torch_ver, cuda_ok = None, None
    try:
        import torch
        torch_ver, cuda_ok = torch.__version__, bool(torch.cuda.is_available())
    except Exception:
        pass
    wi = int(getattr(args, "worker_index", 0))
    meta = {
        "kind": kind, "branch": getattr(args, "branch", None), "final": bool(args.final),
        "comparison": args.comparison, "selection": args.selection,
        "modes": list(args.modes), "dev_seeds": list(args.dev_seeds),
        "test_seeds": list(args.test_seeds), "horizon": args.horizon,
        "trials": args.trials, "study_prefix": args.study_prefix,
        "sampler": "TPESampler(multivariate=True)",
        "sampler_seed_base": int(args.sampler_seed), "worker_index": wi,
        "sampler_seed_effective": _effective_seed(args),
        "perue_calib": args.perue_calib, "final_variant": args.final_variant,
        "base_study_prefix": args.base_study_prefix or args.study_prefix,
        "calib_kind": args.calib_kind, "calib_seeds": list(args.calib_seeds or []),
        "workers": args.workers, "optuna_version": optuna.__version__,
        "torch_version": torch_ver, "cuda_available": cuda_ok, "device": device,
    }
    # per-worker filename so concurrent workers don't overwrite each other's provenance
    _atomic_write_json(out / f"run_meta_{args.study_prefix}_{kind}_w{wi}.json", meta)


def run_study(args, device):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = _make_storage(args.storage)
    modes = tuple(args.modes)
    datasets = build_datasets(args.dev_seeds, args.horizon, modes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _write_run_meta(out, args, device, kind=args.branch)

    def _new_study(name):
        return optuna.create_study(direction="minimize", study_name=name, storage=storage,
                                   sampler=_make_sampler(_effective_seed(args)),
                                   load_if_exists=True)

    if args.branch in ("cell_xgb", "perue_xgb", "cell_lstm", "perue_lstm"):
        # equal_budget -> one study per mode; controlled -> one study over both modes
        mode_groups = ([(m,) for m in modes] if args.comparison == "equal_budget"
                       else [modes])
        for mg in mode_groups:
            name = f"{args.study_prefix}_{args.branch}_{''.join(mg)}"
            study = _new_study(name)
            study.optimize(lambda t: obj_base(t, datasets, mg, args.selection, args.branch, device),
                           n_trials=args.trials)
            _save_best(out, name, study)

    elif args.branch in ("hybrid_xgb", "hybrid_lstm"):
        base_kind = HYBRID_BASE[args.branch]
        # load fixed best base params (matching family), build base-pred cache once
        bp_cache = _hybrid_cache(out, datasets, modes, args, base_kind, device)
        mode_groups = ([(m,) for m in modes] if args.comparison == "equal_budget"
                       else [modes])
        for mg in mode_groups:
            sub = [{m: per[m] for m in mg} for per in bp_cache]
            name = f"{args.study_prefix}_{args.branch}_{''.join(mg)}"
            study = _new_study(name)
            if args.branch == "hybrid_xgb":
                study.optimize(lambda t: obj_hybrid_xgb(t, sub, mg, args.selection),
                               n_trials=args.trials)
            else:
                study.optimize(lambda t: obj_hybrid_lstm(t, sub, mg, args.selection, device),
                               n_trials=args.trials)
            _save_best(out, name, study)


def _hybrid_cache(out, datasets, modes, args, base_kind, device):
    cell_branch, perue_branch = TU.BASE_FOR_HYBRID[base_kind]
    cache = []
    for ds in datasets:
        per = {}
        for m in modes:
            cell_p = _best_params(out, args, cell_branch, m)
            perue_p = _best_params(out, args, perue_branch, m)
            per[m] = base_preds_for(ds, m, base_kind, cell_p, perue_p, device,
                                    perue_calib=args.perue_calib)
        cache.append(per)
    return cache


def _default_params_for(branch):
    """Config-default params in the tuned schema (SMOKE ONLY fallback)."""
    from ..config.defaults import build_config
    cfg = build_config()
    if branch == "cell_xgb":
        return dict(cfg.model.xgb_reg)
    if branch == "perue_xgb":
        return dict(cfg.model.xgb_clf)
    lc = cfg.model.lstm
    if branch == "cell_lstm":
        return dict(seq_len=cfg.simulation.time.seq_len, hidden_size=lc["hidden"],
                    num_layers=1, dropout=lc["dropout"], lr=lc["lr"], epochs=lc["epochs"])
    if branch == "perue_lstm":
        return dict(seq_len=8, hidden_size=lc["hidden"], num_layers=1,
                    dropout=lc["dropout"], lr=lc["lr"], epochs=lc["epochs"])
    raise ValueError(branch)


HYBRID_BRANCHES = TU.HYBRID_BRANCHES        # single source of truth (see tuning.py)


def _prefix_for(args, branch: str) -> str:
    """Which study prefix owns this branch's best params.

    Re-tuning the hybrids on CALIBRATED bases (variant C) needs new hybrid studies while
    the four base branches stay exactly as tuned. --base-study-prefix expresses that:

        --base-study-prefix s3c5 --study-prefix s3c5_cal --perue-calib isotonic

    reads best_s3c5_{cell,perue}_* and writes best_s3c5_cal_hybrid_*. Without it, a fresh
    --study-prefix would look for base params under the new name and find nothing."""
    if branch in HYBRID_BRANCHES:
        return args.study_prefix
    return getattr(args, "base_study_prefix", None) or args.study_prefix


def _param_path(out, args, branch, mode):
    """Resolved best-params file for (branch, mode), or None. equal_budget uses the
    per-mode study; controlled uses the shared study."""
    prefix = _prefix_for(args, branch)
    for tag in (mode, "".join(MODES)):
        p = out / f"best_{prefix}_{branch}_{tag}.json"
        if p.exists():
            return p
    return None


def _maybe_params(out, args, branch, mode):
    """Best params for (branch, mode) if a tuned JSON exists, else None (no raise)."""
    p = _param_path(out, args, branch, mode)
    if p is None:
        return None
    d = json.load(open(p))["params"]
    return {k: v for k, v in d.items() if k not in ("residual", "fusion_mode")}


def _best_params(out, args, branch, mode):
    """Like _maybe_params but REQUIRED: missing tuned params is a HARD error for real
    runs (silently falling back to config defaults would report untuned models) — pass
    --allow-default-fallback only for smoke/dev."""
    p = _maybe_params(out, args, branch, mode)
    if p is not None:
        return p
    if not getattr(args, "allow_default_fallback", False):
        raise FileNotFoundError(
            f"Missing tuned params: branch={branch} mode={mode} "
            f"prefix={_prefix_for(args, branch)} out={out}. Run the tuning stage first, or "
            f"pass --allow-default-fallback (smoke only).")
    return _default_params_for(branch)


def tuning_dev_seeds(out, study_prefix: str) -> set:
    """Dev seeds the study's hyperparameters were selected on, from its run_meta files."""
    dev = set()
    for p in sorted(Path(out).glob(f"run_meta_{study_prefix}_*.json")):
        try:
            dev |= {int(s) for s in (json.loads(p.read_text()).get("dev_seeds") or [])}
        except Exception:
            continue
    return dev


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# The hybrid provenance guard (formerly check_hybrid_calib_match / hybrid_tuning_calib) now
# lives in tuning.py as TU.check_hybrid_provenance, so the headline path (here) and the
# robustness path (training_tuned) cannot drift apart — they used to keep separate copies,
# and both copies shared the same hole: a study whose run_meta predates the perue_calib
# field read back as "unknown -> skip", silently admitting raw-tuned hybrids as variant C.


def _save_best(out, name, study):
    best = {"study": name, "n_trials": len(study.trials),
            "best_value_val_or_oracle": study.best_value, "params": study.best_params}
    _atomic_write_json(out / f"best_{name}.json", best)
    print(f"[{name}] best={study.best_value:.4f}  params={study.best_params}")


# --------------------------------------------------------------------------------------
# Stage 4: final eval on held-out TEST seeds + paired A-B interval (up to six models)
# --------------------------------------------------------------------------------------
def _hybrid_from_cache(bp, hy_json, family, te, te_sl, device, seed):
    """Refit the tuned hybrid on OOF and predict on the test slice. Returns the test
    prediction, or None if the tuned hybrid JSON is absent. fusion_mode (direct/residual/
    gated) is read from the tuned params; residual/gated blend on the raw base preds."""
    if hy_json is None:
        return None
    b1 = bp["b1"]
    params = dict(hy_json["params"])
    # fusion_mode; keep backward-compat with an older 'residual' bool if present
    if "fusion_mode" in params:
        fusion_mode = params.pop("fusion_mode")
    else:
        fusion_mode = "residual" if params.pop("residual", False) else "direct"
    if family == "xgb":
        params.pop("residual", None)
        mdl = FU.fit_hybrid_xgb(bp["Z_oof"][b1:te], bp["y"][b1:te], bp["oof_cell"][b1:te],
                                bp["oof_perue"][b1:te], params, seed, fusion_mode)
        return FU.predict_hybrid_xgb(mdl, bp["Z_full"][te_sl], bp["full_cell"][te_sl],
                                     bp["full_perue"][te_sl], fusion_mode)
    else:
        full, _ = FU.train_lstm_regressor(
            Z_train=bp["Z_oof"], y_train=bp["y"], train_idx=range(b1, te),
            Z_eval=bp["Z_full"], y_eval=bp["y"], eval_idx=range(te_sl.start, te_sl.stop),
            device=device, seed=seed, train_forbid_boundaries=FU._fold_bounds(te, 4),
            eval_forbid_boundaries=(), fusion_mode=fusion_mode,
            base_cell_train=bp["oof_cell"], base_perue_train=bp["oof_perue"],
            base_cell_eval=bp["full_cell"], base_perue_eval=bp["full_perue"],
            **FU._lstm_reg_kwargs(params))
        return full[te_sl]


def _lstm_family_status(out, args, modes):
    """(present, expected, missing) for the LSTM family across the requested modes.
    Components per mode: cell_lstm, perue_lstm, hybrid_lstm."""
    present, expected, missing = 0, 0, []
    for m in modes:
        checks = {
            f"cell_lstm[{m}]": _maybe_params(out, args, "cell_lstm", m) is not None,
            f"perue_lstm[{m}]": _maybe_params(out, args, "perue_lstm", m) is not None,
            f"hybrid_lstm[{m}]": (
                _load_json(out / f"best_{args.study_prefix}_hybrid_lstm_{m}.json") is not None
                or _load_json(out / f"best_{args.study_prefix}_hybrid_lstm_{''.join(MODES)}.json") is not None),
        }
        for name, ok in checks.items():
            expected += 1
            if ok:
                present += 1
            else:
                missing.append(name)
    return present, expected, missing


def _rows_from_shards(out: Path, stem: str, ablation: bool):
    """Rebuild the (variant, model, mode, seed) table from shard files. Shards store raw
    per-seed values precisely so the merge is a re-aggregation, not an average of averages
    (which would be wrong for unequal shard sizes)."""
    shards = sorted(Path(out).glob(f"{stem}.shard_*.json"))
    if not shards:
        raise SystemExit(f"No shards matching {out}/{stem}.shard_*.json. Run the sharded "
                         f"jobs first (--out-tag <TAG>).")
    rows, metas, seen = {}, [], {}
    for sp in shards:
        blob = json.loads(sp.read_text())
        metas.append(blob.get("meta", {}))
        for model, per_mode in blob.get("results", {}).items():
            for mode, entry in per_mode.items():
                if mode not in MODES:
                    continue                       # e.g. paired_A_minus_B in final summaries
                items = entry.items() if ablation else [("_", entry)]
                for variant, e in items:
                    for s, val in (e.get("per_seed") or {}).items():
                        key = (variant, model, mode, int(s))
                        if key in seen and abs(seen[key] - float(val)) > 1e-9:
                            raise SystemExit(
                                f"Shards disagree on {model}/{mode}/{variant} seed {s}: "
                                f"{seen[key]} vs {val}. Shards must cover DISJOINT seeds; "
                                f"delete the stale ones and re-run.")
                        seen[key] = float(val)
                        rows.setdefault(variant, {}).setdefault(
                            model, {a: {} for a in MODES})[mode][int(s)] = float(val)
    # every shard must describe the same protocol
    for k in ("study_prefix", "base_study_prefix", "comparison", "horizon",
              "calibration_kind" if ablation else "final_variant"):
        vals = {json.dumps(m.get(k), sort_keys=True) for m in metas if k in m}
        if len(vals) > 1:
            raise SystemExit(f"Shards disagree on {k}: {sorted(vals)}. They are not one "
                             f"experiment; re-run them with identical settings.")
    print(f"merged {len(shards)} shards: " + ", ".join(p.name for p in shards))
    return rows


def merge_shards(args, device):
    """Combine sharded --final / --calib-ablation runs into the canonical file."""
    out = Path(args.out)
    args.out_tag = None                    # write the canonical (unsharded) filename
    if args.calib_ablation:
        if not args.calib_seeds:
            args.calib_seeds = []
        rows = _rows_from_shards(out, f"final_test_calib_ablation_{args.calib_kind}",
                                 ablation=True)
        for v in VARIANTS:
            rows.setdefault(v, {})
        _report_ablation(rows, out, args)
    else:
        rows = _rows_from_shards(out, "final_test_summary", ablation=False)
        _report(rows.get("_", {}), out, args)


def final_eval(args, device):
    out = Path(args.out)
    modes = tuple(args.modes)
    _write_run_meta(out, args, device, kind="final")
    # Fail loud if the LSTM family was only PARTIALLY tuned: reporting a subset would be
    # misleading. Either none of it is present (XGB-only final) or all of it is.
    n_present, n_expected, missing = _lstm_family_status(out, args, modes)
    if 0 < n_present < n_expected and not getattr(args, "allow_default_fallback", False):
        raise FileNotFoundError(
            "Incomplete LSTM family tuning results: "
            f"{n_present}/{n_expected} components present, missing {missing}. "
            "Finish tuning cell_lstm + perue_lstm + hybrid_lstm for all modes, or pass "
            "--allow-default-fallback (smoke only).")
    # Seeds. The ablation is a MODEL-SELECTION step (it picks a variant), so it must not
    # run on the final test seeds — choosing isotonic because it wins on 101-110 and then
    # reporting 101-110 as the headline IS test-set selection. --calib-seeds enforces the
    # separation; overlap with --test-seeds is a hard error.
    if args.calib_ablation:
        if not args.calib_seeds:
            raise SystemExit(
                "--calib-ablation requires --calib-seeds: a selection pool that is disjoint "
                "from --test-seeds (e.g. --calib-seeds 51 52 53 54 55 while the headline "
                "keeps --test-seeds 101 ...). Selecting the variant on the final test seeds "
                "would be test-set selection.")
        overlap = sorted(set(args.calib_seeds) & set(args.test_seeds))
        if overlap:
            raise SystemExit(
                f"--calib-seeds overlaps --test-seeds on {overlap}. The calibration variant "
                f"must be chosen on seeds that never appear in the reported headline.")
        eval_seeds = list(args.calib_seeds)
        # Second-order but worth saying: on a tuning dev seed the BASE hyperparameters were
        # already selected against that trajectory, so its base predictions are optimistic
        # and the raw-vs-calibrated gap measured there is mildly distorted.
        dev_overlap = sorted(set(args.calib_seeds)
                             & tuning_dev_seeds(out, _prefix_for(args, "cell_xgb")))
        if dev_overlap:
            print("!" * 72)
            print(f"WARNING: --calib-seeds {dev_overlap} are also TUNING dev seeds. The base "
                  f"params were selected on them, so the calibration comparison is slightly "
                  f"optimistic. Prefer a pool disjoint from both --dev-seeds and --test-seeds.")
            print("!" * 72)
    else:
        eval_seeds = list(args.test_seeds)
        # The headline reuses tuned hybrid params: refuse unless they were tuned under
        # exactly this protocol. (The ablation above is diagnostic and states its own
        # lower-bound caveat in its meta, so it is exempt.) NOT downgradable by
        # --allow-default-fallback: that flag is a smoke-test convenience for missing
        # params, whereas reporting a model tuned for other inputs is a result error.
        TU.check_hybrid_provenance(
            out, args.study_prefix,
            perue_calib=("none" if args.final_variant == "R_raw" else args.calib_kind),
            calib_scope=TU.VARIANT_SCOPE[args.final_variant],
            base_study_prefix=(args.base_study_prefix or args.study_prefix),
            comparison=args.comparison, selection=args.selection,
            context=f"--final --final-variant {args.final_variant}")
    # For the REPORTED numbers, selection is done: refit on the full 80% (train+val) and
    # evaluate on the held-out 20% test slice. (val slice is empty here.)
    datasets = build_datasets(eval_seeds, args.horizon, modes, ratios=(0.8, 0.0, 0.2))
    # variant -> model -> {mode -> [per-seed test MAE]}. The ablation scores all three; a
    # normal --final scores exactly one (--final-variant, default R_raw = raw protocol).
    variants = list(VARIANTS) if args.calib_ablation else [args.final_variant]
    # variant -> model -> mode -> {seed: mae}. Keying by SEED (not append order) lets shards
    # from different processes be merged unambiguously, and makes the paired A-B difference
    # match on the seed itself rather than on list position.
    rows = {v: {} for v in variants}

    def add(variant, model, mode, seed, val):
        if val is None or not np.isfinite(val):
            return
        rows[variant].setdefault(model, {a: {} for a in MODES})[mode][int(seed)] = float(val)

    # families to include: XGB always; LSTM only if its base + hybrid were tuned
    families = [("xgb", "Cell_XGB", "PerUE_XGB", "Hybrid_XGB", "cell_xgb", "perue_xgb", "hybrid_xgb")]
    families.append(("lstm", "Cell_LSTM", "PerUE_LSTM", "Hybrid_LSTM",
                     "cell_lstm", "perue_lstm", "hybrid_lstm"))

    def _hyb_mae(bp_v, hy, kind, te, te_sl, device, seed, y, hyb_name, m):
        pred = _hybrid_from_cache(bp_v, hy, kind, te, te_sl, device, seed)
        if pred is None:
            return None
        pred = np.asarray(pred, dtype=np.float64)
        fm = np.isfinite(pred)
        if not fm.any():
            raise RuntimeError(
                f"{hyb_name} produced no finite test predictions (mode {m}) — "
                "check seq_len vs test-slice length / eval boundaries.")
        return mean_absolute_error(np.asarray(y[te_sl])[fm], pred[fm])

    for ds in datasets:
        y, T, te, ve = ds["y"], ds["T"], ds["train_end"], ds["val_end"]
        te_sl = slice(ve, T)
        seed = ds["seed"]
        for m in modes:
            for (kind, cell_name, perue_name, hyb_name,
                 cell_branch, perue_branch, hyb_branch) in families:
                cell_p = _maybe_params(out, args, cell_branch, m)
                perue_p = _maybe_params(out, args, perue_branch, m)
                if cell_p is None or perue_p is None:
                    if kind == "xgb":       # XGB base is mandatory for a real final run
                        _best_params(out, args, cell_branch, m)   # raises with guidance
                    continue                # LSTM family simply not tuned -> skip
                # expensive part (base training + OOF) is computed ONCE and shared by all
                # calibration variants; only the 1-D map and the hybrid refit differ.
                bp = base_preds_for(ds, m, kind, cell_p, perue_p, device)
                hy = (_load_json(out / f"best_{args.study_prefix}_{hyb_branch}_{m}.json")
                      or _load_json(out / f"best_{args.study_prefix}_{hyb_branch}_{''.join(MODES)}.json"))
                if hy is None and kind == "xgb" and not getattr(args, "allow_default_fallback", False):
                    raise FileNotFoundError(
                        f"Missing tuned {hyb_branch} params for mode {m} "
                        f"(prefix={args.study_prefix}, out={out}). Tune {hyb_branch} first, "
                        f"or pass --allow-default-fallback.")
                for v in variants:                     # cell branch is variant-invariant
                    add(v, cell_name, m, seed, _tail_mae(y, bp["full_cell"], te_sl))
                # R and S share the RAW hybrid, so it is fit at most once per (seed,mode)
                hyb_cache = {}
                for v in variants:
                    bp_v, perue_rep = variant_bp(bp, ds["df"], m, y, te, v, args.calib_kind)
                    add(v, perue_name, m, seed, _tail_mae(y, perue_rep, te_sl))
                    key = "raw" if bp_v is bp else "cal"
                    if key not in hyb_cache:
                        hyb_cache[key] = _hyb_mae(bp_v, hy, kind, te, te_sl, device, seed,
                                                  y, hyb_name, m)
                    add(v, hyb_name, m, seed, hyb_cache[key])

    if args.calib_ablation:
        _report_ablation(rows, out, args)
    else:
        _report(rows[args.final_variant], out, args)


def _tail_mae(y, full_pred, sl) -> float:
    fp = np.asarray(full_pred)[sl]
    m = np.isfinite(fp)
    if not m.any():
        return float("nan")
    return float(mean_absolute_error(np.asarray(y)[sl][m], fp[m]))


def _vals(rows, variant, model, mode, seeds=None):
    """MAEs for (variant, model, mode) as a seed-sorted array, optionally on given seeds."""
    d = rows.get(variant, {}).get(model, {}).get(mode, {})
    ks = sorted(d) if seeds is None else [s for s in seeds if s in d]
    return np.array([d[k] for k in ks], float), ks


def _shard_path(out: Path, stem: str, tag) -> Path:
    """Shard files sit next to the canonical file and are merged by --merge-shards."""
    return out / (f"{stem}.shard_{tag}.json" if tag else f"{stem}.json")


def _report_ablation(rows, out, args):
    """Print the per-UE calibration ablation: raw vs OOF-calibrated, per model/mode.
    Decision rule: adopt a calibrated variant only if it improves consistently on the
    CALIBRATION-SELECTION seeds (disjoint from the reported test seeds)."""
    order = ["Cell_XGB", "PerUE_XGB", "Hybrid_XGB", "Cell_LSTM", "PerUE_LSTM", "Hybrid_LSTM"]
    variants = list(VARIANTS)
    tag = getattr(args, "out_tag", None)
    if not tag:
        print(f"\n=== per-UE CALIBRATION ABLATION ({args.calib_kind}) — MAE on selection seeds ===")
        print("  R_raw       = raw per-UE, raw hybrid inputs (current protocol)")
        print("  S_perue_cal = calibrated per-UE reported; hybrid still fed RAW bases")
        print("  C_full_cal  = calibrated per-UE AND calibrated hybrid inputs")
        print(f"  {'model':12} {'mode':5} " + "  ".join(f"{v:>13}" for v in variants))
    summary = {}
    models = [m for m in order if any(m in rows.get(v, {}) for v in variants)]
    for model in models:
        for mode in MODES:
            cells, entry = [], {}
            base, base_seeds = _vals(rows, "R_raw", model, mode)
            for v in variants:
                a, ks = _vals(rows, v, model, mode)
                if len(a):
                    entry[v] = {"mean": float(a.mean()), "std": _std(a), "n_seeds": len(a),
                                "per_seed": {str(k): float(x) for k, x in zip(ks, a)}}
                    cells.append(f"{a.mean():7.3f}±{_std(a):5.3f}")
                else:
                    cells.append(f"{'-':>13}")
            if entry:
                for v in ("S_perue_cal", "C_full_cal"):
                    # paired on the SEED, so a partially-failed shard cannot misalign them
                    cur, ks = _vals(rows, v, model, mode, seeds=base_seeds)
                    ref, _ = _vals(rows, "R_raw", model, mode, seeds=ks)
                    if len(cur) and len(cur) == len(ref) and v in entry:
                        d = cur - ref
                        entry[v]["delta_vs_raw_mean"] = float(d.mean())
                        entry[v]["delta_vs_raw_wins"] = int((d < 0).sum())  # calib better
                summary.setdefault(model, {})[mode] = entry
                if not tag:
                    print(f"  {model:12} {mode:5} " + "  ".join(cells))
    meta = {"calibration_kind": args.calib_kind,
            "calib_selection_seeds": list(args.calib_seeds),
            "reported_test_seeds_NOT_used_here": list(args.test_seeds),
            "study_prefix": args.study_prefix,
            "base_study_prefix": getattr(args, "base_study_prefix", None) or args.study_prefix,
            "comparison": args.comparison, "horizon": args.horizon, "std_ddof": 1,
            "shard_tag": tag,
            "note": ("Selection-only run. Hybrid hyperparameters were tuned on RAW bases, so "
                     "C_full_cal reuses params it was not tuned for and is a LOWER BOUND: to "
                     "adopt C, re-tune hybrid_xgb/hybrid_lstm with --perue-calib <kind> under "
                     "a new --study-prefix (e.g. s3c5_cal) and re-run --final with "
                     "--final-variant C_full_cal. Adopting S needs no hybrid re-tune.")}
    path = _shard_path(out, f"final_test_calib_ablation_{args.calib_kind}", tag)
    _atomic_write_json(path, {"meta": meta, "results": summary})
    print(f"\nsaved -> {path}")
    if tag:
        print(f"shard '{tag}' done; merge all shards with: --final --calib-ablation "
              f"--merge-shards --calib-kind {args.calib_kind} --out {out}")
    else:
        print("NOTE: this run selects a variant; the headline must then be produced on the "
              f"untouched test seeds via: --final --final-variant <chosen> --test-seeds "
              f"{' '.join(str(s) for s in args.test_seeds)}")


def _report(rows, out, args):
    tag = getattr(args, "out_tag", None)
    if not tag:
        print("\n=== FINAL TEST (held-out seeds) — MAE mean +/- std (ddof=1) ===")
    order = ["Cell_XGB", "PerUE_XGB", "Hybrid_XGB", "Cell_LSTM", "PerUE_LSTM", "Hybrid_LSTM"]
    wrapped = {"_": rows}          # reuse _vals' (variant, model, mode) addressing
    summary = {}
    for model in [m for m in order if m in rows] + [m for m in rows if m not in order]:
        line, txt = {}, []
        for m in MODES:
            v, ks = _vals(wrapped, "_", model, m)
            if len(v):
                line[m] = {"mean": float(v.mean()), "std": _std(v), "n_seeds": len(v),
                           "per_seed": {str(k): float(x) for k, x in zip(ks, v)}}
                txt.append(f"{m}={v.mean():.3f}+/-{_std(v):.3f}")
        summary[model] = line
        # pair A and B on the SEED, so a missing seed on one side cannot silently offset
        _, ka = _vals(wrapped, "_", model, "A")
        _, kb = _vals(wrapped, "_", model, "B")
        shared = [s for s in ka if s in set(kb)]
        if len(shared) > 1:
            a, _ = _vals(wrapped, "_", model, "A", seeds=shared)
            b, _ = _vals(wrapped, "_", model, "B", seeds=shared)
            d = a - b
            # Student-t, NOT the 1.96 normal approximation: with n=10 seeds the
            # normal quantile is ~13% too small (t_{.975,9} = 2.262) and every
            # reported interval would be too narrow. No fallback if scipy is
            # absent — quietly computing a normal interval while still labelling
            # it "Student-t 95% CI" is exactly the failure this guards against.
            try:
                from scipy.stats import t as _student_t
            except ImportError as exc:                       # pragma: no cover
                raise RuntimeError(
                    "scipy is required for the paired Student-t interval written "
                    "to final_test_summary.json. Install scipy rather than "
                    "reverting to a normal approximation, which would change the "
                    "published statistic without changing its name.") from exc
            se = d.std(ddof=1) / np.sqrt(len(d))
            crit = float(_student_t.ppf(0.975, df=len(d) - 1))
            ci = (float(d.mean() - crit * se), float(d.mean() + crit * se))
            summary[model]["paired_A_minus_B"] = {"mean": float(d.mean()), "ci95": ci,
                                                  "n_seeds": len(d)}
            txt.append(f"  A-B={d.mean():+.3f} 95%CI[{ci[0]:+.2f},{ci[1]:+.2f}]")
        if not tag:
            print(f"  {model:12} " + "  ".join(txt))
    # Protocol fingerprint. make_paper_tables cross-checks this against every robustness
    # scenario, so a raw headline can never be tabulated next to calibrated robustness, nor
    # an controlled headline next to equal-budget robustness. param_file_sha256 pins the ACTUAL tuned files
    # (same keys/format as training_tuned's provenance), so a prefix that merely looks right
    # is still caught.
    variant = args.final_variant
    shas = {}
    for m in MODES:
        for br in ("cell_xgb", "perue_xgb", "cell_lstm", "perue_lstm") + HYBRID_BRANCHES:
            p = _param_path(out, args, br, m)
            if p is not None:
                shas[f"{m}/{br}"] = _file_sha256(p)
    meta = {"final_variant": variant,
            "perue_calib": ("none" if variant == "R_raw" else args.calib_kind),
            # explicit, so the frozen record does not depend on reading the
            # R_raw collapse out of perue_calib
            "calib_kind": ("none" if variant == "R_raw" else args.calib_kind),
            "calib_scope": VARIANT_SCOPE[variant],
            "study_prefix": args.study_prefix,
            "base_study_prefix": getattr(args, "base_study_prefix", None) or args.study_prefix,
            "comparison": args.comparison,
            "selection": getattr(args, "selection", None),
            "paired_ci": "Student-t 95% CI, df=n-1",
            "test_seeds": list(args.test_seeds), "horizon": args.horizon,
            "train_ratio": 0.8, "std_ddof": 1, "shard_tag": tag, "param_file_sha256": shas}
    path = _shard_path(out, "final_test_summary", tag)
    _atomic_write_json(path, {"meta": meta, "results": summary})
    print(f"\nsaved -> {path}  (variant={variant})")
    if tag:
        print(f"shard '{tag}' done; merge all shards with: --final --merge-shards "
              f"--final-variant {variant} --out {out}")


# --------------------------------------------------------------------------------------
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{int(os.environ.get('LTE_GPU_INDEX', 0))}"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["cell_xgb", "perue_xgb", "cell_lstm", "perue_lstm",
                                         "hybrid_xgb", "hybrid_lstm"])
    ap.add_argument("--final", action="store_true", help="run Stage-4 test evaluation")
    ap.add_argument("--final-variant", choices=list(VARIANTS), default="R_raw",
                    help="which per-UE calibration variant the HEADLINE uses. R_raw is the "
                         "raw protocol; pick S_perue_cal/C_full_cal only after the ablation "
                         "chose it on --calib-seeds. Recorded in final_test_summary.json so "
                         "the paper tables can enforce one protocol.")
    ap.add_argument("--calib-ablation", action="store_true",
                    help="with --final: score all per-UE calibration variants (R/S/C) on "
                         "--calib-seeds and write final_test_calib_ablation_{kind}.json. "
                         "This is a SELECTION run; it never touches --test-seeds.")
    ap.add_argument("--out-tag", default=None,
                    help="shard tag: write final_test_*.shard_<TAG>.json instead of the "
                         "canonical file. Seeds are independent, so --final/--calib-ablation "
                         "can be split across GPUs by giving each process a disjoint seed "
                         "subset and its own tag, then merged with --merge-shards.")
    ap.add_argument("--merge-shards", action="store_true",
                    help="merge *.shard_*.json in --out into the canonical file (re-aggregates "
                         "the stored per-seed values; runs no models)")
    ap.add_argument("--calib-seeds", nargs="+", type=int, default=None,
                    help="selection pool for --calib-ablation; must be disjoint from "
                         "--test-seeds (e.g. 51 52 53 54 55)")
    ap.add_argument("--calib-kind", choices=[k for k in FU.CALIB_KINDS if k != "none"],
                    default="isotonic", help="calibration map fitted on OOF base preds")
    ap.add_argument("--perue-calib", choices=list(FU.CALIB_KINDS), default="none",
                    help="TUNING stage: build hybrid base-pred caches from CALIBRATED per-UE "
                         "predictions. Needed only to re-tune hybrid_xgb/hybrid_lstm for "
                         "variant C; use a fresh --study-prefix (e.g. s3c5_cal).")
    ap.add_argument("--comparison", choices=["equal_budget", "controlled"], default="equal_budget")
    ap.add_argument("--selection", choices=["val", "test_oracle"], default="val")
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--dev-seeds", nargs="+", type=int, default=[7, 13, 42])
    ap.add_argument("--test-seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--storage", default=None, help="optuna storage, e.g. journal:tune.journal")
    ap.add_argument("--study-prefix", default="s1",
                    help="prefix this run's studies are written under (and the default for "
                         "base-param lookups)")
    ap.add_argument("--base-study-prefix", default=None,
                    help="read the four BASE branches' params from this prefix instead. Use "
                         "when re-tuning only the hybrids on calibrated bases: "
                         "--base-study-prefix s3c5 --study-prefix s3c5_cal --perue-calib isotonic")
    ap.add_argument("--sampler-seed", type=int, default=42, help="base TPE sampler seed")
    ap.add_argument("--worker-index", type=int, default=0,
                    help="per-worker offset added to the sampler seed (distinct proposals)")
    ap.add_argument("--workers", type=int, default=1, help="worker count (metadata only)")
    ap.add_argument("--allow-default-fallback", action="store_true",
                    help="use config-default params when a tuned JSON is missing (SMOKE ONLY)")
    ap.add_argument("--out", default="tune_out")
    args = ap.parse_args()
    device = get_device()
    print(f"device={device} selection={args.selection} comparison={args.comparison} "
          f"sampler_seed={args.sampler_seed}+{args.worker_index}={_effective_seed(args)} "
          f"perue_calib={args.perue_calib} final_variant={args.final_variant}")
    if args.final and args.merge_shards:
        merge_shards(args, device)
    elif args.final:
        final_eval(args, device)
    else:
        assert args.branch, "--branch required unless --final"
        run_study(args, device)


if __name__ == "__main__":
    main()