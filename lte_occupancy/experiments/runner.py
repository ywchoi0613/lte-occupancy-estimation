"""
experiments/runner.py — CLI entry point + reproducibility snapshots.

Usage:
    python -m lte_occupancy.experiments.runner                 # 3 seeds, modes A,B
    python -m lte_occupancy.experiments.runner --modes A --seeds 7
    python -m lte_occupancy.experiments.runner --timer 30 --out results_t30

Scale / scenario sweeps are driven by environment variables (see config/defaults):
    LTE_SCALE_PROFILE, LTE_SERVICE_MIX, LTE_STMSI_REALLOC, LTE_TOTAL_TIME, ...
GPU is selected with CUDA_VISIBLE_DEVICES=N (and LTE_GPU_INDEX if needed).

CLI arguments (--seeds/--modes/--timer) are folded into an EFFECTIVE config so the
saved snapshots reflect exactly what ran. Every run folder gets:
    resolved_config.json     the fully-resolved parameters (incl. CLI overrides)
    parameter_registry.csv   provenance for those exact parameters
    environment.json         git commit, library versions, empirical-file hash
If any seed fails, the run exits non-zero (so an incomplete sweep is never silently
turned into a short results table).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.defaults import build_config
from ..config.registry import write_registry_csv
from .training_legacy import run_one_seed
from .training_tuned import run_one_seed_tuned


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent.parent
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "xgboost", "sklearn", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not-installed"
    return out


def _sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


def _snapshot(out_dir: Path, cfg, args, device_str: str, use_cuda: bool):
    emp_path = cfg.simulation.background.empirical_idle_gap_file
    env_keys = [k for k in os.environ if k.startswith("LTE_")] + \
               ["CUDA_VISIBLE_DEVICES", "XGB_FORCE_CPU"]
    (out_dir / "environment.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "device": device_str,
        "cuda_available": use_cuda,
        "versions": _versions(),
        "cli_args": vars(args),
        "empirical_idle_gap_file": Path(emp_path).name,
        "empirical_idle_gap_sha256": _sha256(emp_path),
        "lte_env": {k: os.environ.get(k) for k in env_keys if k in os.environ},
    }, indent=2))
    # resolved_config = the EFFECTIVE config (CLI overrides already folded in)
    (out_dir / "resolved_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    # provenance table for these exact parameters
    write_registry_csv(cfg=cfg, path=str(out_dir / "parameter_registry.csv"),
                       model_set=getattr(args, "model_set", "legacy"))


def _save_tuned_provenance(out_dir: Path, extras: dict):
    """Persist the link between this robustness run and the tuning run: a provenance JSON
    (best-param dict hashes + file SHA-256) AND copies of the exact best_*.json files."""
    prov = {k: extras[k] for k in ("model_set", "params_dir", "study_prefix", "comparison",
                                   "train_ratio", "perue_calib", "calib_scope",
                                   "final_variant", "base_study_prefix", "parameter_hashes",
                                   "param_file_sha256", "selected_params")}
    (out_dir / "tuned_model_provenance.json").write_text(json.dumps(prov, indent=2))
    dst = out_dir / "selected_best_params"
    dst.mkdir(exist_ok=True)
    import shutil
    for src in extras.get("param_files", {}).values():
        try:
            shutil.copy2(src, dst / Path(src).name)
        except Exception as e:
            print(f"  (warn) could not copy best-param file {src}: {e}")


def _warn_if_tuning_seeds(args):
    """Robustness evaluated on the seeds that Optuna selected hyperparameters on is
    OPTIMISTIC: the scenario differs, but the parameters already saw those trajectories.
    The tuning run records its dev seeds in run_meta_*.json, so we can detect the overlap
    and say so loudly. Held-out seeds (e.g. the final test seeds) are preferred for the
    paper's robustness tables."""
    dev = set()
    prefixes = {args.study_prefix, args.base_study_prefix or args.study_prefix}
    metas = [q for pre in prefixes
             for q in sorted(Path(args.params_dir).glob(f"run_meta_{pre}_*.json"))]
    for p in metas:
        try:
            dev |= {int(s) for s in (json.loads(p.read_text()).get("dev_seeds") or [])}
        except Exception:
            continue
    overlap = sorted(set(int(s) for s in args.seeds) & dev)
    if overlap:
        print("!" * 72)
        print(f"WARNING: seeds {overlap} are the TUNING dev seeds of study "
              f"'{args.study_prefix}'.")
        print("  Hyperparameters were selected on these trajectories, so robustness numbers")
        print("  measured on them are optimistic. For paper tables prefer held-out seeds")
        print("  (e.g. the final test seeds 101+). Reporting anyway.")
        print("!" * 72)


def parse_args(defaults):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    p.add_argument("--modes", type=str, nargs="+", default=list(defaults.modes), choices=["A", "B"])
    p.add_argument("--timer", type=int, default=int(defaults.simulation.rrc.inactivity_timer_s))
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--model-set", choices=["legacy", "tuned"], default="tuned",
                   help="tuned = the six-estimator pipeline reported in the paper, with "
                        "hyperparameters loaded from --params-dir. legacy = a superseded "
                        "survival-based pipeline retained for provenance; it produced no "
                        "published result and is not maintained.")
    p.add_argument("--params-dir", type=str, default=None,
                   help="tuned only: directory holding the best_*.json parameter files")
    p.add_argument("--study-prefix", type=str, default="s3c5",
                   help="tuned only: study prefix of the best_*.json files (hybrids)")
    p.add_argument("--base-study-prefix", type=str, default=None,
                   help="tuned only: read the four BASE branches' params from this prefix "
                        "instead (use when only the hybrids were re-tuned on calibrated "
                        "bases: --base-study-prefix s3c5 --study-prefix s3c5_cal)")
    p.add_argument("--comparison", choices=["equal_budget", "controlled"], default="equal_budget",
                   help="tuned only: which tuned study the params came from")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="tuned only: temporal train fraction (test = the rest)")
    p.add_argument("--perue-calib", choices=["none", "isotonic", "linear"], default="none",
                   help="tuned only: per-UE raw->count map fitted on OOF preds. Keep 'none' "
                        "unless the calibration ablation (on its own selection seeds) "
                        "supports it.")
    p.add_argument("--calib-scope", choices=["perue_only", "full"], default="full",
                   help="tuned only, with --perue-calib: 'perue_only' = variant S (only the "
                        "reported PerUE estimator is calibrated; the hybrid keeps raw inputs "
                        "and its tuned params stay valid); 'full' = variant C (the hybrid is "
                        "fit on calibrated bases; its params should have been re-tuned).")
    p.add_argument("--families", type=str, nargs="+", default=["xgb", "lstm"],
                   choices=["xgb", "lstm"],
                   help="tuned only: estimator families to train. Default = both (all six "
                        "branches). '--families xgb' trains only Cell/PerUE/Hybrid-XGB, "
                        "which is what the robustness suite uses: the LSTM family showed "
                        "substantially higher cross-seed variability and "
                        "computational cost in the primary experiments.")
    p.add_argument("--overwrite", action="store_true",
                   help="replace a non-empty output directory (default: refuse)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _effective_config(cfg, args):
    """Fold CLI arguments into the config so snapshots reflect what actually ran."""
    eff_sim = replace(cfg.simulation,
                      rrc=replace(cfg.simulation.rrc, inactivity_timer_s=float(args.timer)))
    return replace(cfg, seeds=tuple(args.seeds), modes=tuple(args.modes), simulation=eff_sim)


def main():
    import warnings
    # Cell feature bank is built with many frame inserts (correctness-neutral). Optimizing
    # it would change the tuned feature vectors, so we suppress the noisy perf warning here
    # rather than touch build_cell_features.
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    base_cfg = build_config()
    args = parse_args(base_cfg)
    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be strictly between 0 and 1.")
    cfg = _effective_config(base_cfg, args)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; use --overwrite to replace it "
                         f"(prevents mixing artifacts from different configs).")
    if args.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..config.device import DEVICE, USE_CUDA
    device_str = str(DEVICE)

    if args.model_set == "tuned" and not args.params_dir:
        raise SystemExit("--model-set tuned requires --params-dir (e.g. --params-dir tune_out_s3c5).")
    if args.model_set == "tuned":
        _warn_if_tuning_seeds(args)

    print("=" * 64)
    print("LTE Cell Occupancy Estimation" + (f"  [{args.model_set.upper()} models]"))
    print("=" * 64)
    tuned_src = (f" | params={args.params_dir} ({args.study_prefix}/{args.comparison}, "
                 f"train_ratio={args.train_ratio}, "
                 f"perue_calib={args.perue_calib}/{args.calib_scope})"
                 if args.model_set == "tuned" else "")
    print(f"Device: {device_str} (CUDA={USE_CUDA}) | seeds={args.seeds} | "
          f"modes={args.modes} | families={args.families} | RRC timer={args.timer}s | "
          f"scale={cfg.scale_profile} | mix={cfg.simulation.traffic.service_mix_name} | "
          f"out={out_dir}{tuned_src}")
    print("=" * 64)

    _snapshot(out_dir, cfg, args, device_str, USE_CUDA)

    all_results = []
    failures = []
    manifests_saved = False
    t_total = time.time()
    for seed in args.seeds:
        try:
            if args.model_set == "tuned":
                res, preds, extras = run_one_seed_tuned(
                    seed=seed, timer=args.timer, modes=args.modes, cfg=cfg, device=DEVICE,
                    params_dir=args.params_dir, study_prefix=args.study_prefix,
                    comparison=args.comparison, train_ratio=args.train_ratio,
                    perue_calib=args.perue_calib, calib_scope=args.calib_scope,
                    base_study_prefix=args.base_study_prefix,
                    families=args.families, verbose=not args.quiet)
            else:
                res, preds, extras = run_one_seed(seed=seed, timer=args.timer, modes=args.modes,
                                                  cfg=cfg, device=DEVICE, verbose=not args.quiet)
            all_results.append(res)
            pd.DataFrame(all_results).to_csv(out_dir / "results.csv", index=False)
            to_save = {k: v for k, v in preds.items() if isinstance(v, np.ndarray)}
            to_save["split_idx"] = np.array(preds["split_idx"])
            np.savez(out_dir / f"preds_seed{seed}.npz", **to_save)
            # raw trace (all counters + Mode B noisy counters) for later inspection
            trace = extras["trace"]
            try:
                trace.to_parquet(out_dir / f"trace_seed{seed}.parquet", index=False)
            except Exception:
                trace.to_csv(out_dir / f"trace_seed{seed}.csv.gz", index=False)
            (out_dir / f"truth_hashes_seed{seed}.json").write_text(
                json.dumps(extras["truth_hashes"], indent=2))
            if not manifests_saved:      # feature names are seed-independent
                for m, man in extras["manifests"].items():
                    (out_dir / f"feature_manifest_{m}.json").write_text(json.dumps(man, indent=2))
                if args.model_set == "tuned":
                    _save_tuned_provenance(out_dir, extras)
                manifests_saved = True
        except Exception as exc:
            print(f"!!! Seed {seed} failed: {exc}")
            import traceback; traceback.print_exc()
            failures.append((seed, repr(exc)))

    df_res = pd.DataFrame(all_results)
    if not df_res.empty:
        df_res.to_csv(out_dir / "results.csv", index=False)
        numeric_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
        summary = pd.DataFrame({"mean": df_res[numeric_cols].mean(),
                                "std": df_res[numeric_cols].std()})
        summary.to_csv(out_dir / "summary.csv")
        print("\n" + "=" * 64)
        print(f"DONE in {(time.time()-t_total)/60:.1f} min")
        print(summary.round(3).to_string())
        print(f"Saved to: {out_dir}/  (results.csv, summary.csv, resolved_config.json, "
              f"parameter_registry.csv, environment.json)")

    # Never let an incomplete sweep pass silently.
    if failures:
        raise SystemExit(f"{len(failures)} seed(s) failed: {failures}")
    if len(all_results) != len(args.seeds):
        raise SystemExit(f"Incomplete: {len(all_results)}/{len(args.seeds)} seeds produced results.")


if __name__ == "__main__":
    main()