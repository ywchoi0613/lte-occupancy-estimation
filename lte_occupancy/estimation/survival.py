"""
estimation/survival.py — Per-UE CLUSTERED EMPIRICAL SURVIVAL estimator.

Per-UE occupancy estimation needs P(UE still in cell | elapsed since last seen).
We estimate it with a clustered EMPIRICAL survival curve:

  1. Each UE is assigned to one of N clusters by its observed connection rate.
  2. Within each cluster, S_cluster(e) = fraction of OBSERVED exits with duration
     beyond e slots after their last observed activity.
  3. At inference, for a UE unseen for `elapsed` slots, look up S_cluster(elapsed).

NAMING (fix 10.6): this is NOT the strict Kaplan-Meier product-limit estimator.
Right-censored UEs (still present at train_end) inform the clustering but are NOT
folded into a product-limit risk-set correction; dropping them biases the curve
toward shorter dwell. We therefore call it a "clustered empirical survival"
estimator rather than Kaplan-Meier. If a strict product-limit estimate with proper
censoring is required, swap in ``lifelines.KaplanMeierFitter`` per cluster (this
changes the numbers and adds a dependency, so it is left as an explicit option).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_NAME = "clustered_empirical_survival"


def _gather_survival_rows(ue_events: dict, ue_meta: dict, train_end: int) -> pd.DataFrame:
    """duration = slots from a UE's LAST observed activity to its exit (event=1)
    or the last trainable slot (right-censored, event=0). conn_rate -> clustering.

    train_end is EXCLUSIVE (test begins at index train_end); the last trainable
    slot is train_end-1, so events at t >= train_end are ignored (fixes the former
    1-slot boundary leakage)."""
    censor_time = train_end - 1
    rows = []
    for pid, events in ue_events.items():
        meta = ue_meta.get(pid)
        if meta is None or meta.get("is_resident", False):
            continue
        first_seen = last_seen = exit_time = None
        n_conn = 0
        for t, et in events:
            if t >= train_end:
                break
            if et in ("connect", "disconnect"):
                if first_seen is None:
                    first_seen = t
                last_seen = t
            if et == "connect":
                n_conn += 1
            elif et == "gt_exit":
                exit_time = t
        if first_seen is None or last_seen is None:
            continue
        if exit_time is not None and last_seen <= exit_time < train_end:
            duration = max(1, exit_time - last_seen); event = 1
        else:
            duration = max(1, censor_time - last_seen); event = 0
        obs_span = max(1, last_seen - first_seen)
        rows.append({"pid": pid, "duration": duration, "event": event,
                     "conn_rate": n_conn / obs_span * 1000.0})
    return pd.DataFrame(rows)


def fit_clustered_empirical_survival(ue_events, ue_meta, train_end, n_clusters: int,
                                     max_elapsed: int):
    """Returns (get_survival(cluster_or_cov, elapsed), pid_clusters, model_name)."""
    if n_clusters != 3:
        # The 30th/80th-percentile split below hard-codes three activity clusters.
        raise ValueError(
            f"clustered_empirical_survival currently supports exactly 3 clusters, got {n_clusters}.")
    rows = _gather_survival_rows(ue_events, ue_meta, train_end)
    if len(rows) == 0:
        return (lambda c, e: 0.5), {}, MODEL_NAME

    rates = rows["conn_rate"].values
    q_low = np.percentile(rates, 30)
    q_high = np.percentile(rates, 80)
    pid_clusters = {}
    cluster_exits = {0: [], 1: [], 2: []}
    for _, r in rows.iterrows():
        cl = 0 if r["conn_rate"] <= q_low else (1 if r["conn_rate"] <= q_high else 2)
        pid_clusters[int(r["pid"])] = cl
        if r["event"] == 1:
            cluster_exits[cl].append(int(r["duration"]))

    surv_funcs = {}
    for cl in range(n_clusters):
        exits = cluster_exits[cl]
        if len(exits) < 10:
            all_exits = [e for exl in cluster_exits.values() for e in exl]
            exits = all_exits if all_exits else [100]
        ea = np.array(exits)
        max_e = min(int(np.percentile(ea, 99)), max_elapsed)
        surv_funcs[cl] = np.array([np.mean(ea > t) for t in range(max_e + 2)])

    def get_survival(covariates_or_cluster, elapsed):
        if isinstance(covariates_or_cluster, dict):
            cl = covariates_or_cluster.get("cluster", 1)
        else:
            cl = int(covariates_or_cluster) if covariates_or_cluster is not None else 1
        sv = surv_funcs.get(cl, surv_funcs.get(1, np.array([0.5])))
        e = int(elapsed)
        if e < 0:
            return 1.0
        if e >= len(sv):
            return float(sv[-1])
        return float(sv[e])

    return get_survival, pid_clusters, MODEL_NAME
