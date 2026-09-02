"""
config/registry.py — Parameter provenance registry.

Emits ``parameter_registry.csv``: for every simulator/observation setting, its
value, unit, and — crucially — WHERE it comes from. ``source_type`` is exactly one
of a fixed 4-value taxonomy:

  measured        : calibrated from real srsRAN epc.log measurements
  literature      : taken from a cited paper / 3GPP spec
  scenario        : a modelling / scenario assumption (nuance like "adapted from
                    3GPP" or "literature-informed" goes in the `source` column,
                    NOT in source_type)
  hyperparameter  : an ML training knob (performance-tuned, NOT physics)

Methodological point: simulator parameters must be justified by measurement,
literature, or an explicit scenario assumption — never silently tuned to test MAE.
Only ``hyperparameter`` rows are legitimately performance-tuned.

Structured (nested) parameters are FLATTENED to one row per leaf so the actual
numbers (weights, means, durations) appear in the table. The registry writer takes
an explicit ``cfg`` (the effective per-run config) rather than rebuilding a default.

Run:  python -m lte_occupancy.config.registry   ->  writes the default-config CSV
"""
from __future__ import annotations

import csv
from pathlib import Path

from .defaults import build_config

# Config hyperparameters that ONLY the legacy pipeline consumes. The tuned six-model
# pipeline reads its estimator hyperparameters from the Optuna best_*.json files instead
# (see tuned_model_provenance.json / selected_best_params/ in each tuned run directory), and
# the Survival estimator does not exist there at all. Emitting the config defaults for these
# in a TUNED run would claim a provenance the run never used, so write_registry_csv marks
# them explicitly rather than printing a value that had no effect.
_LEGACY_ONLY_HYPERPARAMS = {
    "model.xgb_reg":                 "Cell/Hybrid XGB settings",
    "model.xgb_clf":                 "per-UE XGB classifier settings",
    "model.lstm":                    "LSTM architecture/training settings",
    "model.mode_b_xgb_overrides":    "Mode-B XGB overrides",
    "model.per_ue.n_clusters":       "Survival behaviour clusters",
    "model.per_ue.survival_max_elapsed": "Survival table horizon",
}

# (dotted_name_under_ExperimentConfig, unit, source_type, source, rationale, sweep)
_META = [
    # ---- time / topology ----
    ("simulation.time.slot_duration_s", "s", "scenario", "design",
     "1 slot := 1 s; sets the seconds<->slots mapping", "-"),
    ("simulation.time.total_slots", "slots", "scenario", "design",
     "evaluation run length (432,000 slots = 5.0 days at 1 s/slot; warm-up excluded)", "LTE_TOTAL_TIME"),
    ("simulation.time.train_ratio", "-", "scenario", "design",
     "chronological train fraction; test is the final 20%", "-"),
    ("simulation.topology.n_cells", "count", "scenario", "design",
     "single-cell scope (no handover / inter-cell)", "-"),
    ("simulation.topology.initial_nonresident_ues", "UEs", "scenario", "scale profile",
     "seed population of finite-dwell UEs at t=0", "LTE_SCALE_PROFILE"),
    ("simulation.topology.resident_ues", "UEs", "scenario", "scale profile",
     "always-present UEs (never exit)", "LTE_SCALE_PROFILE"),
    # ---- arrival ----
    ("simulation.arrival.base_rate", "UEs/slot", "scenario", "scale profile",
     "base Poisson arrival intensity", "LTE_SCALE_PROFILE"),
    ("simulation.arrival.diurnal_period_slots", "slots", "scenario", "design",
     "one diurnal cycle = 86,400 slots = 24 h at 1 s/slot; t_sin/t_cos encode wall-clock time of day, not a simulator-internal period", "-"),
    ("simulation.arrival.global_event_prob", "prob/slot", "scenario", "design",
     "per-slot chance of a whole-cell demand surge", "-"),
    ("simulation.arrival.global_event_access_boost", "x", "scenario", "design",
     "access-prob multiplier during a whole-cell surge", "-"),
    # ---- mobility (flattened) ----
    ("simulation.mobility.dwell_classes", "slots/weight", "scenario", "design",
     "transient/regular/stationary dwell (exp) + class weights", "-"),
    ("simulation.mobility.min_dwell_slots", "slots", "scenario", "design",
     "floor on sampled dwell", "-"),
    # ---- traffic (flattened) ----
    ("simulation.traffic.usage_intensity", "prob/weight", "scenario", "design",
     "low/medium/high per-slot access probabilities + weights", "-"),
    ("simulation.traffic.service_types", "slots/weight", "scenario",
     "adapted from 3GPP TR 36.822",
     "voice/streaming/browsing durations, burst/idle structure, mix weights", "LTE_SERVICE_MIX"),
    ("simulation.traffic.min_service_duration_slots", "slots", "scenario", "design",
     "floor on a service session length", "-"),
    # ---- RRC ----
    ("simulation.rrc.inactivity_timer_s", "s", "literature", "Huang et al. 2013",
     "commercial LTE RRC inactivity ~10 s; srsRAN's ~30 s default is absorbed by "
     "the measured idle_gap, so this timer governs session length", "--timer / RRC sweep"),
    ("simulation.rrc.ra_retry_prob", "prob", "scenario", "design",
     "random-access preamble retry probability", "-"),
    ("simulation.rrc.reject_prob", "prob", "scenario", "design",
     "RRC setup reject probability", "-"),
    ("simulation.rrc.detach_on_exit_prob", "prob", "scenario", "design",
     "chance of explicit DETACH when a UE leaves", "-"),
    ("simulation.rrc.meas_report_prob_per_slot", "prob/slot", "scenario", "design",
     "per-slot measurement-report emission for a connected UE", "-"),
    ("simulation.rrc.nas_ul_on_attach", "msgs", "literature", "3GPP TS 24.301",
     "UL NAS messages on initial attach", "-"),
    ("simulation.rrc.nas_dl_on_attach", "msgs", "literature", "3GPP TS 24.301",
     "DL NAS messages on initial attach", "-"),
    # ---- paging ----
    ("simulation.paging.prob_base", "prob/slot", "scenario", "design",
     "base per-slot paging probability for an idle UE", "-"),
    ("simulation.paging.response_prob", "prob", "scenario", "design",
     "probability a paged UE responds", "-"),
    # ---- radio (generated, unused) ----
    ("simulation.radio.rsrp_at_center_dbm", "dBm", "literature", "typical macro-cell RSRP",
     "RSRP at cell centre; GENERATED per UE but UNUSED this edition", "-"),
    # ---- background reconnection: distributions MEASURED, class split SCENARIO ----
    ("simulation.background.empirical_idle_gap_file", "file", "measured", "srsRAN epc.log",
     "per-class release->reconnect idle-gap DISTRIBUTIONS, bootstrap-sampled, floored "
     "to slots; timer-invariant", "-"),
    ("simulation.background.device_bg_class", "weight/slots", "scenario", "design",
     "chatty/moderate/quiet_bg class WEIGHTS + burst means; 3-class split weakly "
     "supported (silhouette 0.16 for 3-class vs 0.35 for 2-class) — documented limit", "-"),
    # ---- observation: Mode A ----
    ("observation.mode_a_features", "counters", "scenario", "cooperative eNB",
     "20 observable S1AP/NAS/RRC counters (incl. n_paging_response)", "-"),
    # ---- observation: Mode B ----
    ("observation.mode_b_features", "counters", "literature", "LTESniffer (Hoang 2023)",
     "8 counters a passive over-the-air sniffer can recover", "-"),
    ("observation.drx_miss_prob", "prob", "scenario", "sniffer abstraction",
     "fraction of connected UEs invisible to the sniffer per slot (DRX)", "-"),
    ("observation.release_inference.mean_delay_s", "s", "scenario", "sniffer model",
     "mean release-detection lag (non-negative; clamped)", "-"),
    ("observation.release_inference.delay_std_s", "s", "scenario", "sniffer model",
     "release-detection lag jitter", "-"),
    ("observation.release_inference.false_negative_prob", "prob", "scenario", "sniffer model",
     "probability a true release is missed", "-"),
    ("observation.release_inference.false_positive_rate", "per-UE", "scenario", "sniffer model",
     "spurious-release rate per connected UE", "-"),
    ("observation.stmsi.realloc_mean_slots", "slots", "scenario",
     "Hong 2018 (informed); Shaik 2016",
     "mean S-TMSI reallocation interval (~3 h exp); fragments Mode B tracks only", "LTE_STMSI_REALLOC"),
    # ---- estimator hyperparameters (performance-tuned; NOT physics) ----
    ("model.per_ue.n_clusters", "count", "hyperparameter", "-",
     "UE behaviour clusters for the empirical survival curve", "-"),
    ("model.per_ue.prune_elapsed", "slots", "hyperparameter", "-",
     "per-UE feature horizon / track pruning", "-"),
    ("model.per_ue.subsample", "-", "hyperparameter", "-",
     "keep 1/N per-UE samples in time (compute budget)", "-"),
    ("model.per_ue.survival_max_elapsed", "slots", "hyperparameter", "-",
     "max elapsed retained in the survival table", "-"),
    ("model.xgb_reg", "-", "hyperparameter", "-",
     "XGBoost regressor settings (tuned for MAE, not physics)", "-"),
    ("model.mode_b_xgb_overrides", "-", "hyperparameter", "-",
     "Mode-B-specific XGB overrides (noisier signals)", "-"),
    ("model.lstm", "-", "hyperparameter", "-",
     "LSTM architecture/training settings (tuned for MAE)", "-"),
    # ---- previously-missing groups ----
    ("simulation.arrival.commute_bumps", "-", "scenario", "design",
     "two Gaussian commute-like peaks (center/width/height) on the arrival rate", "-"),
    ("simulation.arrival.day_scale_range", "-", "scenario", "design",
     "per-cycle Uniform arrival multiplier (mild aperiodicity)", "-"),
    ("simulation.arrival.day_scale_buffer_days", "cycles", "scenario", "design",
     "extra cycles appended to the per-cycle scale vector", "-"),
    ("simulation.arrival.burst_events", "-", "scenario", "scale profile",
     "fixed rectangular demand surges (start/dur/amplitude); amplitude scales with profile", "LTE_SCALE_PROFILE"),
    ("simulation.mobility.resident_exit_pad_slots", "slots", "scenario", "design",
     "residents' exit = total_slots + pad (never leave)", "-"),
    ("simulation.paging.intensity_factor", "x", "scenario", "design",
     "paging-rate multiplier by usage class (low/medium/high)", "-"),
    ("simulation.radio.dist_beta", "-", "scenario", "design",
     "Beta(a,b) normalized distance; GENERATED but unused", "-"),
    ("simulation.radio.rsrp_edge_drop_db", "dB", "scenario", "design",
     "RSRP drop centre->edge; GENERATED but unused", "-"),
    ("simulation.radio.shadowing_std_db", "dB", "scenario", "design",
     "log-normal shadowing std; GENERATED but unused", "-"),
    ("simulation.gc_interval_slots", "slots", "scenario", "design",
     "prune OUT UEs every N slots (housekeeping; no result impact)", "-"),
    ("observation.stmsi.enabled", "bool", "scenario", "LTE_STMSI_REALLOC",
     "whether S-TMSI reallocation identity-noise is active (Mode B only)", "LTE_STMSI_REALLOC"),
    ("model.xgb_clf", "-", "hyperparameter", "-",
     "Per-UE XGBoost classifier settings (tuned for MAE)", "-"),
    # ---- feature-engineering hyperparameters (affect model inputs) ----
    ("features.ewma_alphas", "-", "hyperparameter", "-",
     "EWMA smoothing factors for cell features", "-"),
    ("features.lag_offsets", "slots", "hyperparameter", "-",
     "lagged-interaction offsets", "-"),
    ("features.surge_window", "slots", "hyperparameter", "-",
     "trailing window for the observable-surge baseline", "sweepable"),
    ("features.surge_percentile", "-", "hyperparameter", "-",
     "train-only percentile defining an observed surge", "-"),
    ("features.release_spike_percentile", "-", "hyperparameter", "-",
     "train-only percentile defining a release spike", "-"),
    ("features.turnover_windows", "slots", "hyperparameter", "-",
     "rolling windows for turnover-ratio features", "-"),
    ("features.flow_cumulative_window", "slots", "hyperparameter", "-",
     "window for cumulative net-flow feature", "-"),
    ("features.paging_windows", "slots", "hyperparameter", "-",
     "rolling windows for paging-response-rate features (Mode A)", "-"),
]


def _get(cfg, dotted):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _flatten(name, val):
    """Yield (leaf_name, value_str) — recursing into dicts so real numbers show."""
    if isinstance(val, dict):
        for k, v in val.items():
            yield from _flatten(f"{name}.{k}", v)
    elif isinstance(val, (tuple, list)):
        yield name, "; ".join(str(x) for x in val)
    else:
        yield name, str(val)


def make_rows(cfg):
    rows = []
    for name, unit, stype, source, rationale, sweep in _META:
        try:
            val = _get(cfg, name)
        except AttributeError:
            val = "(missing)"
        # store a logical id for file paths, not a machine-specific absolute path
        if name.endswith("empirical_idle_gap_file"):
            val = f"lte_occupancy/{Path(str(val)).name}"
        for leaf_name, leaf_val in _flatten(name, val):
            rows.append({"parameter": leaf_name, "value": leaf_val, "unit": unit,
                         "source_type": stype, "source": source,
                         "rationale": rationale, "sweep": sweep})
    return rows


def write_registry_csv(cfg=None, path: str = None, model_set: str = "legacy"):
    """Emit the provenance registry for ONE run.

    model_set="tuned" rewrites the rows whose config value the tuned pipeline never reads:
    its estimator hyperparameters come from the Optuna best_*.json files, and Survival is
    not part of the six-model family at all. Those rows keep their parameter name (so the
    table stays comparable across pipelines) but report where the effective value actually
    lives, instead of a config default that had no effect on the run."""
    if cfg is None:
        cfg = build_config()
    if model_set not in ("legacy", "tuned"):
        raise ValueError(f"model_set must be 'legacy' or 'tuned', got {model_set!r}")
    rows = make_rows(cfg)
    if model_set == "tuned":
        for r in rows:
            name = r["parameter"]
            # flattened leaves keep the group as a prefix (e.g. "model.xgb_reg.max_depth")
            group = next((g for g in _LEGACY_ONLY_HYPERPARAMS
                          if name == g or name.startswith(g + ".")), None)
            if group is not None:
                r["value"] = "Optuna-selected (per mode/branch)"
                r["source"] = "best_*.json"
                r["rationale"] = (
                    f"{_LEGACY_ONLY_HYPERPARAMS[group]}: this CONFIG DEFAULT is not read by "
                    f"the tuned pipeline. The effective value IS used, but it is chosen by "
                    f"Optuna per (mode, branch) and recorded in tuned_model_provenance.json "
                    f"and selected_best_params/"
                    + (" (Survival is not part of the tuned six-model family)"
                       if "per_ue" in group else ""))
            elif name == "simulation.time.train_ratio":
                r["value"] = "set by CLI --train-ratio"
                r["source"] = "CLI"
                r["rationale"] = ("tuned runs take the split from --train-ratio (0.8 for the "
                                  "headline/robustness; the tuning stage uses 0.6/0.2/0.2), "
                                  "not from this config field")
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "parameter_registry.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "value", "unit", "source_type",
                                          "source", "rationale", "sweep"])
        w.writeheader()
        w.writerows(rows)
    return path, len(rows)


if __name__ == "__main__":
    # The repo-level CSV is generated by hand, so the model set must be stated explicitly:
    # silently defaulting to "legacy" here would reintroduce exactly the provenance error
    # this flag exists to prevent (the paper reports the TUNED pipeline).
    import argparse
    ap = argparse.ArgumentParser(description="write the parameter provenance registry")
    ap.add_argument("--model-set", choices=["legacy", "tuned"], required=True,
                    help="which pipeline the CSV describes; 'tuned' marks the estimator "
                         "hyperparameters as Optuna-selected instead of printing config "
                         "defaults the tuned pipeline never reads")
    ap.add_argument("--out", default=None, help="output path (default: package root)")
    a = ap.parse_args()
    p, n = write_registry_csv(path=a.out, model_set=a.model_set)
    print(f"wrote {n} rows ({a.model_set}) -> {p}")