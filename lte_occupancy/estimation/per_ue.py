"""
estimation/per_ue.py — Per-UE occupancy estimator.

Wraps the clustered empirical survival model with event-stream processing to
produce slot-by-slot occupancy estimates.

  PerUE_Surv : sum of P(still in cell | cluster, elapsed) over tracked UEs
  PerUE_XGB  : XGB "is_present?" classifier sum (subsampled in time)

The optional cell-context uses the OBSERVABLE surge_flag (fix 10.2), never the
burst schedule.

PERFORMANCE NOTE (2026-07-28)
-----------------------------
``compute_per_ue`` used to be O(T x N_ever_seen) with a large constant, which
made long horizons quadratic-ish: ``tracks`` keeps every UE that has ever been
seen (needed: a returning UE must keep its history), yet the per-slot loop
walked that whole dict and threw away everyone past ``prune_elapsed``. At the
legacy 8k-slot tuning horizon almost nothing was stale, so the defect was
invisible; at 432k slots the dict holds ~3e4 tracks of which only ~9e2 can
contribute, i.e. ~97% of the scan is dead weight. Worse, every surviving UE
went through ``_track_covariates``, which rebuilds a dict and runs
``np.mean``/``np.std`` over ``session_lengths`` -- a list that itself grows with
the horizon -- on EVERY slot, only for ``get_survival`` to read the single
``cluster`` key and discard the rest.

Two changes, both exact (see verify_perue_fix.py):

1. ``get_survival(tr.cluster, elapsed)`` instead of
   ``get_survival(_track_covariates(tr, t), elapsed)``. ``get_survival``
   accepts a bare cluster id and reads only ``cluster`` from the dict form, so
   the returned value is identical -- the covariate dict was pure waste.

2. An active set (``live``) holding exactly the tracks that can contribute at
   the current slot, maintained by a lazy timer wheel. Membership is removed at
   ``last_seen + prune_elapsed + 1`` -- precisely the first slot at which the
   old predicate would have skipped the track -- and restored on the next
   event, so the set equals the old loop's non-skipped subset at every slot.
   ``live`` is kept sorted by creation ordinal, which is the dict-insertion
   order the old loop walked, so contributions are summed in the SAME order and
   the floating-point results are bitwise identical, not merely close.

``compute_per_ue_legacy`` below is the previous implementation kept verbatim so
the equivalence can be re-checked at any time. ``build_perue_xgb`` is left
untouched: it walks ``tracks`` only every ``subsample`` slots and measured well
under 2% of per-trial cost.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from xgboost import XGBClassifier

from ..features.per_ue import per_ue_features_extended
from ..features.cell import observable_surge
from ..event_order import event_key_with_pid


@dataclass
class UETrack:
    pid: int
    first_seen: int = 0
    last_seen: int = 0
    is_connected: bool = False
    n_connections: int = 0
    total_active_time: int = 0
    _connect_start: int = -1
    n_paging_sent: int = 0
    n_paging_resp: int = 0
    session_lengths: list = field(default_factory=list)
    cluster: int = -1
    usage_intensity: Optional[str] = None
    dwell_type: Optional[str] = None
    bg_class: Optional[str] = None
    # --- active-set bookkeeping (performance only; never read as a feature) ---
    _idx: int = -1          # creation ordinal == tracks-dict insertion order
    _live: bool = False     # currently a member of the active set
    _exp_at: int = -1       # slot of this track's one pending expiry check


def _track_covariates(tr: UETrack, t: int) -> dict:
    """Full covariate dict for the survival model.

    RETAINED FOR API COMPATIBILITY ONLY. ``get_survival`` reads just the
    ``cluster`` key, so the hot loop passes ``tr.cluster`` directly; building
    this dict per UE per slot (two numpy reductions over a list that grows with
    the horizon) was the dominant cost of the per-UE branch.
    """
    tk = max(1, t - tr.first_seen)
    return {
        "n_connections": tr.n_connections,
        "conn_rate": tr.n_connections / tk * 1000.0,
        "mean_session": float(np.mean(tr.session_lengths)) if tr.session_lengths else 0.0,
        "std_session": float(np.std(tr.session_lengths)) if len(tr.session_lengths) > 1 else 0.0,
        "active_ratio": tr.total_active_time / tk,
        "usage_intensity": tr.usage_intensity, "dwell_type": tr.dwell_type,
        "bg_class": tr.bg_class, "cluster": tr.cluster,
    }


def build_perue_xgb(ue_events: dict, ue_meta: dict, train_end: int, seed: int,
                    xgb_clf_params: dict, subsample: int, prune_elapsed: int,
                    cell_context_fn: Optional[Callable[[int], dict]] = None):
    # train_end is EXCLUSIVE: test begins at index train_end, so training uses
    # strictly t < train_end (fixes the former 1-slot boundary leakage).
    # Whitelist connect/disconnect ONLY. Excluding just gt_enter/gt_exit would let
    # s_tmsi_realloc events through (Mode A reads the raw ue_events, which carry those
    # events when S-TMSI is ON), spuriously creating/advancing UETracks and perturbing
    # Mode A — S-TMSI must affect Mode B tracks only.
    all_events = [(t, pid, et) for pid, evs in ue_events.items()
                  for t, et in evs if t < train_end and et in ("connect", "disconnect")]
    # Canonical order: at the same slot a (clamped) disconnect must precede the next
    # connect. Plain .sort() would order by the event STRING (connect < disconnect)
    # and reintroduce the 'delayed release closes a new session' bug.
    all_events.sort(key=event_key_with_pid)
    sample_times = list(range(0, int(train_end), subsample))
    tracks: dict[int, UETrack] = {}
    features, labels = [], []
    ev_idx = 0
    for t in sample_times:
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
                continue
            meta = ue_meta[pid]
            label = 1 if meta["enter_time"] <= t < meta["exit_time"] else 0
            features.append(per_ue_features_extended(tr, t, ctx))
            labels.append(label)
    if len(features) == 0:
        return None
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    pos_ratio = y.mean() if len(y) > 0 else 0.5
    spw = (1 - pos_ratio) / max(pos_ratio, 0.01)
    clf = XGBClassifier(**xgb_clf_params, scale_pos_weight=spw, random_state=seed)
    clf.fit(X, y)
    return clf


def compute_per_ue(ue_events, ue_meta, get_survival, pid_clusters, xgb_model,
                   total_time, subsample: int, prune_elapsed: int,
                   cell_context_fn: Optional[Callable[[int], dict]] = None):
    """Slot-by-slot PerUE_Surv / PerUE_XGB estimates.

    Output-identical to compute_per_ue_legacy (see module docstring); the only
    differences are that stale tracks are not re-scanned every slot and that the
    survival model is handed the cluster id it actually uses.
    """
    # Whitelist connect/disconnect ONLY (see build_perue_xgb): keeps S-TMSI's
    # s_tmsi_realloc events from perturbing Mode A per-UE tracking.
    all_events = [(t, pid, et) for pid, evs in ue_events.items()
                  for t, et in evs if et in ("connect", "disconnect")]
    all_events.sort(key=event_key_with_pid)   # same-slot: disconnect before connect

    tracks: dict[int, UETrack] = {}     # pid -> track (never evicted: keeps history)
    by_idx: list[UETrack] = []          # creation ordinal -> track
    live: list[int] = []                # ordinals that can contribute now, ascending
    expiry: dict[int, list[int]] = {}   # slot -> ordinals to re-check (lazy timer wheel)
    ev_idx = 0
    surv_est = np.zeros(total_time); xgb_est = np.zeros(total_time)
    last_xgb_val = 0.0

    def _schedule(tr: UETrack, now: int) -> None:
        """Register the one pending expiry check for `tr`. Stale entries are
        ignored at pop time via the `_exp_at` stamp, so re-scheduling is free."""
        e = tr.last_seen + prune_elapsed + 1
        if e < now:          # defensive: already past due -> settle it this slot
            e = now
        if tr._exp_at != e:
            expiry.setdefault(e, []).append(tr._idx)
            tr._exp_at = e

    for t in range(total_time):
        while ev_idx < len(all_events) and all_events[ev_idx][0] <= t:
            ev_t, pid, et = all_events[ev_idx]
            tr = tracks.get(pid)
            if tr is None:
                tr = UETrack(pid=pid, first_seen=ev_t)
                meta = ue_meta.get(pid, {})
                tr.usage_intensity = meta.get("usage_intensity")
                tr.dwell_type = meta.get("dwell_type")
                tr.bg_class = meta.get("device_bg_class")
                if pid in pid_clusters:
                    tr.cluster = pid_clusters[pid]
                tr._idx = len(by_idx)
                by_idx.append(tr)
                tracks[pid] = tr
                live.append(tr._idx)      # newest ordinal is the largest: stays sorted
                tr._live = True
            if et == "connect":
                if not tr.is_connected:
                    tr.n_connections += 1; tr._connect_start = ev_t
                tr.is_connected = True; tr.last_seen = ev_t
                if pid in pid_clusters:
                    tr.cluster = pid_clusters[pid]
            elif et == "disconnect":
                if tr.is_connected and tr._connect_start >= 0:
                    tr.total_active_time += ev_t - tr._connect_start
                    tr.session_lengths.append(ev_t - tr._connect_start)
                tr.is_connected = False; tr.last_seen = ev_t; tr._connect_start = -1
            if not tr._live:                       # returning UE re-enters the scan
                insort(live, tr._idx); tr._live = True
            if not tr.is_connected:                # connected tracks never expire
                _schedule(tr, t)
            ev_idx += 1

        for idx in expiry.pop(t, ()):
            tr = by_idx[idx]
            if tr._exp_at != t or tr.is_connected:
                continue                           # superseded stamp, or still connected
            if t - tr.last_seen > prune_elapsed:
                pos = bisect_left(live, idx)
                if pos < len(live) and live[pos] == idx:
                    del live[pos]
                tr._live = False
            else:
                _schedule(tr, t)                   # event moved last_seen forward

        s_total = 0.0
        for idx in live:
            tr = by_idx[idx]
            if tr.is_connected:
                s_total += 1.0
            else:
                s_total += get_survival(tr.cluster, t - tr.last_seen)
        surv_est[t] = s_total

        if xgb_model is not None and t % subsample == 0:
            ctx = cell_context_fn(t) if cell_context_fn is not None else None
            feat_batch = [per_ue_features_extended(by_idx[i], t, ctx) for i in live]
            if feat_batch:
                probs = xgb_model.predict_proba(np.array(feat_batch, dtype=np.float32))[:, 1]
                last_xgb_val = float(probs.sum())
            else:
                last_xgb_val = 0.0
        xgb_est[t] = last_xgb_val
    return surv_est, xgb_est


def compute_per_ue_legacy(ue_events, ue_meta, get_survival, pid_clusters, xgb_model,
                          total_time, subsample: int, prune_elapsed: int,
                          cell_context_fn: Optional[Callable[[int], dict]] = None):
    """Pre-2026-07-28 implementation, kept verbatim as the equivalence oracle."""
    all_events = [(t, pid, et) for pid, evs in ue_events.items()
                  for t, et in evs if et in ("connect", "disconnect")]
    all_events.sort(key=event_key_with_pid)
    tracks: dict[int, UETrack] = {}
    ev_idx = 0
    surv_est = np.zeros(total_time); xgb_est = np.zeros(total_time)
    last_xgb_val = 0.0
    for t in range(total_time):
        while ev_idx < len(all_events) and all_events[ev_idx][0] <= t:
            ev_t, pid, et = all_events[ev_idx]
            if pid not in tracks:
                tr = UETrack(pid=pid, first_seen=ev_t)
                meta = ue_meta.get(pid, {})
                tr.usage_intensity = meta.get("usage_intensity")
                tr.dwell_type = meta.get("dwell_type")
                tr.bg_class = meta.get("device_bg_class")
                if pid in pid_clusters:
                    tr.cluster = pid_clusters[pid]
                tracks[pid] = tr
            tr = tracks[pid]
            if et == "connect":
                if not tr.is_connected:
                    tr.n_connections += 1; tr._connect_start = ev_t
                tr.is_connected = True; tr.last_seen = ev_t
                if pid in pid_clusters:
                    tr.cluster = pid_clusters[pid]
            elif et == "disconnect":
                if tr.is_connected and tr._connect_start >= 0:
                    tr.total_active_time += ev_t - tr._connect_start
                    tr.session_lengths.append(ev_t - tr._connect_start)
                tr.is_connected = False; tr.last_seen = ev_t; tr._connect_start = -1
            ev_idx += 1
        s_total = 0.0
        for pid, tr in tracks.items():
            if tr.is_connected:
                s_total += 1.0
            else:
                elapsed = t - tr.last_seen
                if elapsed > prune_elapsed:
                    continue
                s_total += get_survival(_track_covariates(tr, t), elapsed)
        surv_est[t] = s_total
        if xgb_model is not None and t % subsample == 0:
            ctx = cell_context_fn(t) if cell_context_fn is not None else None
            feat_batch = []
            for pid, tr in tracks.items():
                elapsed = t - tr.last_seen
                if elapsed > prune_elapsed and not tr.is_connected:
                    continue
                feat_batch.append(per_ue_features_extended(tr, t, ctx))
            if feat_batch:
                probs = xgb_model.predict_proba(np.array(feat_batch, dtype=np.float32))[:, 1]
                last_xgb_val = float(probs.sum())
            else:
                last_xgb_val = 0.0
        xgb_est[t] = last_xgb_val
    return surv_est, xgb_est


def make_cell_context_fn(df, mode: str, train_end: int, wave_period: int, feat):
    """Cell context for the Per-UE XGB. Uses the OBSERVABLE surge_flag (fix 10.2)
    computed from observed inflow with a train-only threshold — never the burst
    schedule. n_connected_t is the mode's observable connected count."""
    nc = (df["n_connected_b"].values if (mode == "B" and "n_connected_b" in df.columns)
          else df["n_connected"].values)
    t_arr = df["t"].values if "t" in df.columns else np.arange(len(df))
    _, surge_flag = observable_surge(df, mode, train_end, feat.surge_window, feat.surge_percentile)

    def ctx(t):
        return {
            "n_connected_t": float(nc[t]) if t < len(nc) else 0.0,
            "surge_flag": float(surge_flag[t]) if t < len(surge_flag) else 0.0,
            "t_sin": float(np.sin(2 * np.pi * t_arr[t] / wave_period)) if t < len(t_arr) else 0.0,
            "t_cos": float(np.cos(2 * np.pi * t_arr[t] / wave_period)) if t < len(t_arr) else 0.0,
        }
    return ctx
