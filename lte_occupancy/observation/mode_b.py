"""
observation/mode_b.py — Mode B: passive over-the-air sniffer.

Mode B models a PASSIVE-OBSERVATION setting (not a strict "realistic lower bound":
it is a constrained passive-monitoring scenario). It exposes TWO separately-noisy
observation outputs, which have different reliabilities and therefore different
noise models:

  1. Aggregate slot counters (what the sniffer tallies per slot):
       * connected count is DRX-degraded (some active UEs are invisible) -> n_connected_b
       * release count is inferred from C-RNTI inactivity, subject to detection
         DELAY, false NEGATIVES, and false POSITIVES                     -> n_release_inferred
  2. Per-track session boundaries (whether a *specific* track has ended):
       * a setup (connect) is seen over the air as-is;
       * a release is DELAYED but EVENTUALLY resolved from sustained C-RNTI
         inactivity — so per-track boundaries carry detection lag only. Per-track
         false-negative (never-resolved) and false-positive (spurious) releases are
         NOT injected: a permanent per-track FN would need an extra inactivity-
         timeout parameter, so we keep the lag-only per-track model (see
         observe_ue_events). Aggregate FN/FP still live at the counter level above.

Because a delayed release is applied per session, a large delay could otherwise push
one session's release past the NEXT session's setup; observe_ue_events builds truth
episodes and clamps each observed release to the next connect, since a new setup
proves the previous session already ended (episode-aware canonicalization).

  * under periodic S-TMSI reallocation the sniffer cannot link a UE's activity
    across reallocation boundaries -> per-UE tracks fragment.

The Mode-B naive baseline uses n_connected_b (NOT the clean n_connected), because
the sniffer never sees the clean count — using it would leak a Mode-A-only signal
into the Mode-B comparison (correctness fix 10.7).

Suggested paper wording for the two-tier model:
  "Mode B models two passive-observation outputs separately. Aggregate release counts
   are subject to delay, false negatives, and false positives, whereas per-track
   session boundaries are delayed but eventually resolved from sustained C-RNTI
   inactivity. Per-track false-positive termination is not injected."

References (S-TMSI reallocation):
  Hong et al. (2018) NDSS "GUTI Reallocation Demystified";
  Shaik et al. (2016) NDSS "Practical Attacks Against Privacy and Availability in 4G/LTE".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.schema import ObservationCfg, ReleaseInferenceCfg
from ..event_order import event_key


def _release_inference(release_actual, n_connected, cfg: ReleaseInferenceCfg,
                       total_time: int, seed: int = 0):
    """Truth releases -> sniffer-observed n_release_inferred (lag + FN/FP noise)."""
    rng = np.random.default_rng(seed)
    inferred = np.zeros(total_time, dtype=np.int32)
    for t in range(total_time):
        for _ in range(int(release_actual[t])):
            if rng.random() < cfg.false_negative_prob:
                continue
            # detection lag is non-negative: a sniffer cannot observe a release
            # BEFORE it happens, so clamp the (rarely negative) Gaussian jitter.
            delay = max(0.0, rng.normal(cfg.mean_delay_s, cfg.delay_std_s))
            new_t = t + int(round(delay))
            if new_t < total_time:
                inferred[new_t] += 1
    for t in range(total_time):
        n_fp = rng.poisson(cfg.false_positive_rate * max(0, int(n_connected[t])))
        inferred[t] += n_fp
    return inferred


def _drx_degraded_connected(n_connected_truth, drx_miss_prob: float, seed: int = 0):
    """Sniffer's noisy connected count (DRX hides some active UEs)."""
    rng = np.random.default_rng(seed + 9999)
    n = np.array(n_connected_truth, dtype=np.int32)
    miss = rng.binomial(np.maximum(n, 0), drx_miss_prob)
    return n - miss


class ModeBObserver:
    mode = "B"

    def __init__(self, cfg: ObservationCfg, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.feature_columns = list(cfg.mode_b_features)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the sniffer-observable columns (n_connected_b, n_release_inferred)
        to the truth DataFrame in place, returning it."""
        df["n_release_inferred"] = _release_inference(
            df["n_release"].values, df["n_connected"].values,
            self.cfg.release_inference, total_time=len(df), seed=self.seed)
        df["n_connected_b"] = _drx_degraded_connected(
            df["n_connected"].values, self.cfg.drx_miss_prob, seed=self.seed)
        return df

    def naive_connected(self, df: pd.DataFrame):
        """Naive occupancy estimate for Mode B = the DRX-degraded connected count
        (the only connected signal the sniffer actually observes). Fix 10.7."""
        col = "n_connected_b" if "n_connected_b" in df.columns else "n_connected"
        return df[col].values

    def observe_ue_events(self, ue_events: dict, ue_meta: dict, total_time: int):
        """Sniffer's per-track event view — EPISODE-AWARE, lag-only (fixes the former
        'Mode B Per-UE uses clean truth' asymmetry, and the follow-on bug where a naive
        per-event lag/FN could leave tracks stuck connected or let a delayed old release
        close a new session).

        The truth event log alternates connect/disconnect per UE. We pair it into
        SESSION EPISODES (connect, disconnect?) and derive each session's observed
        boundary:

          connect        -> observed as-is (RRC setup is over-the-air)
          disconnect     -> observed = disconnect + non-negative detection lag, then
                            CLAMPED to the next session's connect: a new setup proves
                            the previous session already ended, so the observed release
                            can never cross it. No per-track false-negative drop and no
                            per-track false-positive injection (see module docstring):
                            per-track boundaries are delayed but eventually resolved.
          (open session)  -> a session with no truth disconnect (still connected at exit
                            / horizon) emits no release.
          horizon         -> if a completed session's delayed release lands at/after
                            total_time, the sniffer simply has not observed it yet: this
                            is legitimate right-censoring, so no release is emitted and
                            the track ends 'connected'. (Distinct from a bug: a completed
                            IN-horizon session always gets its release.)
          gt_exit         -> kept (supervised labels / censoring)
          s_tmsi_realloc  -> kept (consumed later by split_events_by_stmsi)

        Same-slot ordering (a clamped release and the next setup can coincide) is
        disconnect-before-connect via event_key. ue_meta is returned unchanged (labels
        use the true presence interval). Uses a dedicated RNG so cell-level noise (the
        aggregate n_release_inferred, which DOES carry FN/FP) is unaffected.
        """
        ri = self.cfg.release_inference
        rng = np.random.default_rng(self.seed + 4242)
        new_events: dict = {}
        for pid, evs in ue_events.items():
            sev = sorted(evs, key=event_key)
            passthrough = [(t, et) for (t, et) in sev
                           if et not in ("connect", "disconnect")]
            # --- pair connect/disconnect into episodes: (connect_t, disconnect_t|None) ---
            episodes: list = []
            open_c = None
            for (t, et) in sev:
                if et == "connect":
                    if open_c is not None:            # (shouldn't happen in truth)
                        episodes.append((open_c, None))
                    open_c = t
                elif et == "disconnect":
                    if open_c is not None:
                        episodes.append((open_c, t)); open_c = None
                    # stray disconnect w/o open connect -> ignore (shouldn't happen)
            if open_c is not None:
                episodes.append((open_c, None))       # trailing open session
            # --- derive observed boundaries with lag + clamp + horizon censoring ---
            out = list(passthrough)
            for idx, (c_t, d_t) in enumerate(episodes):
                out.append((c_t, "connect"))
                if d_t is None:
                    continue                          # open session: no release seen
                delay = max(0, int(round(rng.normal(ri.mean_delay_s, ri.delay_std_s))))
                obs_d = d_t + delay
                if idx + 1 < len(episodes):           # clamp to next session's setup
                    obs_d = min(obs_d, episodes[idx + 1][0])
                if obs_d < total_time:                # else: right-censored (stays connected)
                    out.append((obs_d, "disconnect"))
            out.sort(key=event_key)                   # canonical order (disconnect<connect)
            new_events[pid] = out
        return new_events, dict(ue_meta)


# ---- Mode-B XGB params (noisier signals -> its own XGB config) ----
def mode_b_xgb_params(base_xgb_reg: dict, overrides: dict, seed: int) -> dict:
    params = dict(base_xgb_reg)
    params.update(overrides)
    params["random_state"] = seed
    return params


# ---- S-TMSI reallocation (per-UE identity noise) ----
def install_stmsi_realloc_patch(SimCls, realloc_mean_slots: int):
    """Monkey-patch Sim.step to emit `s_tmsi_realloc` events at exponential
    intervals (mean realloc_mean_slots), using a DEDICATED RNG isolated from the
    physical-simulation streams so the cell trajectory (n_present, all cell-level
    counters) is IDENTICAL with vs without realloc — only the per-UE identity, and
    hence the Per-UE/Hybrid pipeline, changes. Idempotent."""
    from ..simulation.state import State

    if not hasattr(SimCls, "_orig_step"):
        SimCls._orig_step = SimCls.step
    orig_step = SimCls._orig_step

    def patched_step(self, t):
        out = orig_step(self, t)
        if getattr(self, "_stmsi_realloc_mean", None):
            if getattr(self, "_realloc_rng", None) is None:
                self._realloc_rng = np.random.default_rng(getattr(self, "seed", 0) + 777)
            rr = self._realloc_rng
            mean = self._stmsi_realloc_mean
            for p in self.ppl:
                if p.state == State.OUT:
                    continue
                if p._next_realloc is None:
                    p._next_realloc = t + int(rr.exponential(mean))
                    p._s_tmsi_epoch = 0
                    continue
                if t >= p._next_realloc:
                    p._s_tmsi_epoch += 1
                    p._next_realloc = t + int(rr.exponential(mean))
                    self._log_event(p.pid, t, "s_tmsi_realloc")
        return out

    SimCls.step = patched_step


def split_events_by_stmsi(ue_events: dict, ue_meta: dict):
    """Split each UE's event list at s_tmsi_realloc boundaries into separate
    anonymous tracks (the sniffer cannot link them across reallocations).

    Each fragment is re-sorted with event_key so the canonical same-slot ordering
    (disconnect before connect) survives fragmentation, and enter_time is taken from
    the fragment's earliest event under that ordering."""
    EPOCH_OFFSET = 1_000_000
    new_events: dict = {}
    new_meta: dict = {}

    def _emit(pid, epoch, events, meta_base):
        events = sorted(events, key=event_key)        # preserve canonical order
        new_pid = pid + epoch * EPOCH_OFFSET
        new_events[new_pid] = events
        new_meta[new_pid] = dict(meta_base)
        new_meta[new_pid]["enter_time"] = events[0][0]

    for pid, ev_list in ue_events.items():
        epoch = 0
        events_this_epoch = []
        meta_base = dict(ue_meta.get(pid, {}))
        for (t, et) in ev_list:
            if et == "s_tmsi_realloc":
                if events_this_epoch:
                    _emit(pid, epoch, events_this_epoch, meta_base)
                epoch += 1
                events_this_epoch = []
                continue
            events_this_epoch.append((t, et))
        if events_this_epoch:
            _emit(pid, epoch, events_this_epoch, meta_base)
    return new_events, new_meta
