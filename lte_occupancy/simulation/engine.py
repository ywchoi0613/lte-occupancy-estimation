"""
simulation/engine.py — Single-cell control-plane TRUTH simulator (Phase-1).

Real 24 h axis (slot = 1 s, day = 86,400 slots). Changes vs the legacy engine:

 M1  Warm-up: the engine simulates ``warmup_slots`` BEFORE t=0 (whole days, so
     day-phase alignment holds). Rows/events are recorded only for t >= 0;
     exported times are shifted so recorded t=0 is midnight of day 0 (Monday).
 M3  Personas: access_prob is per-second and modulated by the 24 h
     activity_factor (night-suppressed user activity).
 M5  RACH: explicit 4-message chain per TR 37.868 —
       n_rach_trigger >= n_rar >= n_rrc_request >= n_rrc_setup
       = n_rrc_setup_complete = n_rach_success = n_new_connections.
     Per-attempt detection 1-e^-i (power ramping), per-RAO preamble collision,
     collided-but-detected UEs still send Msg3 (eNB cannot discern; resolved at
     Msg4), give-up after preamble_trans_max. Dedicated rng stream.
 M10 Paging = MT causes attributed to the empirically-timed background
     reconnect process (fraction f_mt is push-triggered; retries counted), plus
     a low-rate MT-voice Poisson, plus p_engage escalation (user opens the app
     -> non-voice service session). The srsRAN idle-gap anchor keeps fixing
     TOTAL reconnection timing; with p_engage=0 the truth trajectory is
     invariant to f_mt (asserted in tests). Dedicated rng stream.

Release semantics: on inactivity-timer expiry the eNB
emits RRCConnectionRelease to the UE (n_release) AND an S1AP UE CONTEXT RELEASE
REQUEST toward the MME (n_release_request) — there is no UE-originated "RRC
release request" in LTE; the counter name denotes the S1AP eNB->MME message.

RNG streams: legacy ``random`` + global ``np.random`` + ``self._rng`` keep their
legacy roles (class/service sampling, durations/Poisson, day-scale/idle-gap).
NEW mechanisms draw ONLY from dedicated generators ``self._rng_rach`` /
``self._rng_page`` so future ablations stay trajectory-isolated.
"""
from __future__ import annotations

import json
import math
import random
from typing import Optional

import numpy as np
import pandas as pd

from ..config.schema import SimConfig
from .state import State, Person
from .arrival import ArrivalModel, DAY_SLOTS


def _load_idle_gap_pools(path: str):
    with open(path) as f:
        raw = json.load(f)["idle_gap"]
    pools = {k: np.asarray(v, dtype=float) for k, v in raw.items() if len(v) > 0}
    print("[EMPIRICAL_BG] loaded idle-gap pools from " + path + ": "
          + ", ".join(f"{k}:n={len(v)}" for k, v in pools.items()))
    return pools


class Sim:
    """Discrete-time slot-level TRUTH simulator of a single LTE cell."""

    def __init__(self, seed: int, cfg: SimConfig, rrc_timer: Optional[int] = None):
        random.seed(seed)
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        self._rng_rach = np.random.default_rng((seed << 1) ^ 0x9AC4)
        self._rng_page = np.random.default_rng((seed << 1) ^ 0x7A6E)
        self.seed = seed
        self.cfg = cfg
        rrc_timer_s = float(rrc_timer) if rrc_timer is not None else float(cfg.rrc.inactivity_timer_s)
        self.RRC = cfg.time.seconds_to_slots(rrc_timer_s)
        self.BG = cfg.background.device_bg_class
        self.AF = tuple(cfg.traffic.activity_factor) or tuple([1.0] * 24)
        self.warmup = int(cfg.time.warmup_slots)
        self.horizon = int(cfg.time.total_slots)
        self.total_u = self.warmup + self.horizon
        self._mt_voice_rate = {  # per-active-second incoming-call rate, by class
            cls: cfg.paging.mt_voice_per_day * cfg.paging.intensity_factor[cls]
                 / (sum(self.AF) * 3600.0)
            for cls in cfg.paging.intensity_factor
        }
        self.ppl: list[Person] = []
        self.npid = 0
        self.ue_events: dict[int, list[tuple[int, str]]] = {}
        self.ue_meta: dict[int, dict] = {}
        # diagnostics for the fidelity gate (validate_fidelity.py)
        self.cause_counts: dict[int, dict] = {}
        self.rach_stats = dict(preamble_tx=0, rar=0, msg3=0, success=0,
                               collided_tx=0, giveup=0)
        self.page_stats = dict(sent=0, answered=0, mt_data=0, mt_voice=0,
                               engage=0, mo_bg=0)
        self.page_k: dict[int, int] = {}          # paging retry-count histogram
        self.service_sessions: list[dict] = []    # per-session fidelity log (T2)
        # pids whose single boundary-slot disconnect log is suppressed
        # (1-second left-censored episodes; see _open_recording_window)
        self._censor_skip: set[int] = set()

        self._emp_pools = _load_idle_gap_pools(cfg.background.empirical_idle_gap_file)
        # Arrival model draws day-scale + burst schedule from self._rng FIRST.
        self.arrival = ArrivalModel(cfg.arrival, self.total_u, self.warmup, self._rng)

        for _ in range(cfg.topology.resident_ues):
            p = self._create_person(0, True); p.state = State.IDLE; self.ppl.append(p)
        for _ in range(cfg.topology.initial_nonresident_ues):
            p = self._create_person(0); p.state = State.IDLE; self.ppl.append(p)

    # ---------- bookkeeping ----------
    def _log_event(self, pid, u, et):
        if u >= self.warmup:
            self.ue_events.setdefault(pid, []).append((u - self.warmup, et))

    def _log_disconnect(self, pid, u):
        if pid in self._censor_skip:
            self._censor_skip.discard(pid)
            return
        self._log_event(pid, u, "disconnect")

    def _count_cause(self, pid, cause):
        d = self.cause_counts.setdefault(pid, {})
        d[cause] = d.get(cause, 0) + 1

    def _record_meta(self, p: Person):
        self.ue_meta[p.pid] = {
            "usage_intensity": p.usage_intensity, "dwell_type": p.dwell_type,
            "device_bg_class": p.device_bg_class, "is_resident": p.is_resident,
            "enter_time": p.enter_time - self.warmup,
            "exit_time": p.exit_time - self.warmup,
        }

    def _close_service_session(self, p: Person, u: int, censored: bool = False):
        """Log one finished (or exit-truncated) service session for the T2
        fidelity tables: wall time vs total active transfer time vs bursts.
        Pure logging — consumes no RNG, so trajectories are untouched."""
        if p.active_service is None:
            return
        self.service_sessions.append({
            "service": p.active_service,
            "wall_s": int(u - getattr(p, "_svc_start", u)),
            "active_s": int(getattr(p, "_svc_active", 0)),
            "bursts": int(getattr(p, "_svc_bursts", 1)),
            "censored": bool(censored),
        })

    def _sample_weighted(self, d: dict) -> str:
        r = random.random(); c = 0.0
        for name, cfgd in d.items():
            c += cfgd["weight"]
            if r < c:
                return name
        return list(d.keys())[-1]

    def _sample_service_excluding(self, exclude: tuple, rng) -> str:
        pool = {k: v for k, v in self.cfg.traffic.service_types.items()
                if k not in exclude and v["weight"] > 0}
        tot = sum(v["weight"] for v in pool.values())
        r = rng.random() * tot; c = 0.0
        for name, v in pool.items():
            c += v["weight"]
            if r < c:
                return name
        return list(pool.keys())[-1]

    def _sample_idle_gap(self, bg_class: str) -> int:
        pool = self._emp_pools.get(bg_class)
        if pool is None or not len(pool):
            raise RuntimeError(f"No empirical idle-gap pool for bg_class '{bg_class}'.")
        return max(1, int(self._rng.choice(pool)))

    def _create_person(self, et: int, res: bool = False) -> Person:
        c = self.cfg
        u_cls = self._sample_weighted(c.traffic.usage_intensity)
        dw = self._sample_weighted(c.mobility.dwell_classes)
        bg = self._sample_weighted(self.BG)
        dist = np.random.beta(*c.radio.dist_beta)
        rsrp = c.radio.rsrp_at_center_dbm - c.radio.rsrp_edge_drop_db * dist \
            + np.random.normal(0, c.radio.shadowing_std_db)
        mean_dwell = c.mobility.dwell_classes[dw]["mean_dwell"]
        ext = (self.total_u + c.mobility.resident_exit_pad_slots) if res \
            else et + max(c.mobility.min_dwell_slots, int(np.random.exponential(mean_dwell)))
        p = Person(
            pid=self.npid, usage_intensity=u_cls, dwell_type=dw, device_bg_class=bg,
            enter_time=et, exit_time=ext, rsrp=rsrp, distance=dist, is_resident=res,
            next_bg_reconnect=et + self._sample_idle_gap(bg),
        )
        self._record_meta(p)
        self.npid += 1
        return p

    # ---------- session / connection ----------
    def _start_service(self, p: Person, st: str):
        cfg = self.cfg.traffic.service_types[st]
        td = max(self.cfg.traffic.min_service_duration_slots,
                 int(np.random.normal(cfg["duration_mean"], cfg["duration_std"])))
        p.last_service = p.active_service
        p.active_service = st
        p.service_total_remaining = td
        p.in_service_idle = False
        if not cfg.get("multi_rrc", False):
            p.session_remaining = td
            p.service_burst_remaining = td
        else:
            bl_raw = max(1, int(np.random.normal(cfg["burst_len_mean"], cfg["burst_len_std"])))
            bl = min(bl_raw, p.service_total_remaining)
            p.session_remaining = bl
            p.service_burst_remaining = bl

    def _finalize_connect(self, p: Person, u: int, is_bg: bool, via_paging: bool,
                          forced_service: Optional[str], cause: str) -> dict:
        """Establish the RRC connection (post-RACH) and emit NAS-side counters."""
        c = self.cfg
        p.state = State.CONNECTED
        if forced_service is not None:                    # MT voice / MT engage
            self._start_service(p, forced_service)
            p._svc_start = u; p._svc_bursts = 1
            p._svc_active = p.service_total_remaining
            service_changed = (p.last_service != forced_service)
        elif p.active_service and p.service_burst_remaining > 0:  # multi-RRC resume
            p.session_remaining = p.service_burst_remaining
            p._svc_bursts = getattr(p, "_svc_bursts", 0) + 1
            service_changed = False
        elif is_bg:                                        # serviceless short burst
            bg = self.BG[p.device_bg_class]
            p.session_remaining = max(1, int(np.random.normal(bg["bg_burst_mean"],
                                                              bg["bg_burst_std"])))
            p.active_service = None
            service_changed = False
        else:                                              # user-initiated (MO)
            new_service = self._sample_weighted(c.traffic.service_types)
            self._start_service(p, new_service)
            p._svc_start = u; p._svc_bursts = 1
            p._svc_active = p.service_total_remaining
            service_changed = (p.last_service != new_service)
        p.in_inactivity_wait = False
        p.inactivity_remaining = 0
        self._log_event(p.pid, u, "connect")
        self._count_cause(p.pid, cause)

        out = dict(initial_ue_msg=1, ul_nas=0, dl_nas=0, bearer=1, reconfig=1,
                   attach=0, svc_req=0, paging_resp=1 if via_paging else 0)
        if not p.has_attached:
            out["attach"] = 1
            out["ul_nas"] += c.rrc.nas_ul_on_attach
            out["dl_nas"] += c.rrc.nas_dl_on_attach
            p.has_attached = True
        else:
            out["svc_req"] = 1
        if service_changed and p.last_service is not None:
            out["reconfig"] += 1
        return out

    # ---------- RACH (M5) ----------
    def _rach_procedure(self, n_candidates: int):
        """Joint 4-message RACH for this slot's access attempts.
        Returns (winner_flags, stats). TR 37.868-derived assumptions:
        per-attempt detection prob 1-e^-i (power ramping). Collision handling
        is a stated MODELING CHOICE (TS 36.321-style contention resolution),
        not the unique TR 37.868 behavior: >=2 UEs on the same (RAO, preamble)
        are indistinguishable at Msg1; with collision_msg3=True all of them
        transmit Msg3 and Msg4 keeps exactly one, otherwise the attempt is
        unresolved and every collided UE retries. Counter chain:
        preamble_tx >= rar (rar counts detected GROUPS) and
        msg3 >= setup == success; rar vs msg3 has NO fixed order under
        collisions. Losers/undetected retry up to preamble_trans_max, then
        give up.
        The whole exchange resolves within the 1 s slot (200 RAOs, 20 ms
        backoff scale)."""
        rc = self.cfg.rach
        rng = self._rng_rach
        st = dict(preamble_tx=0, rar=0, msg3=0, success=0, collided_tx=0, giveup=0)
        winners = [False] * n_candidates
        remaining = list(range(n_candidates))
        for attempt in range(1, rc.preamble_trans_max + 1):
            if not remaining:
                break
            st["preamble_tx"] += len(remaining)
            p_det = 1.0 - math.exp(-attempt)
            if len(remaining) == 1:                       # fast path: no collision
                if rng.random() < p_det:
                    i = remaining.pop()
                    st["rar"] += 1; st["msg3"] += 1; st["success"] += 1
                    winners[i] = True
                continue
            raos = rng.integers(0, rc.raos_per_slot, size=len(remaining))
            pres = rng.integers(0, rc.n_preambles, size=len(remaining))
            groups: dict[tuple, list] = {}
            for k, i in enumerate(remaining):
                groups.setdefault((int(raos[k]), int(pres[k])), []).append(i)
            nxt = []
            for key, members in groups.items():
                if rng.random() >= p_det:                 # preamble not detected
                    nxt.extend(members); continue
                st["rar"] += 1
                if len(members) == 1:
                    st["msg3"] += 1; st["success"] += 1
                    winners[members[0]] = True
                else:                                     # >=2 UEs collided
                    st["collided_tx"] += len(members)
                    if rc.collision_msg3:
                        st["msg3"] += len(members)
                        w = members[int(rng.integers(0, len(members)))]
                        st["success"] += 1; winners[w] = True
                        nxt.extend(m for m in members if m != w)
                    else:
                        nxt.extend(members)   # unresolved: nobody proceeds
            remaining = nxt
        st["giveup"] = len(remaining)
        return winners, st

    # ---------- main step ----------
    def step(self, u: int) -> dict:
        c = self.cfg
        n_release = 0; n_release_request = 0
        n_paging_sent = 0; n_paging_response = 0
        n_initial_ue_msg = 0; n_ul_nas = 0; n_dl_nas = 0
        n_bearer_activated = 0; n_rrc_reconfig = 0
        n_attach_request = 0; n_detach_request = 0; n_service_request = 0
        elig = {"low": 0, "medium": 0, "high": 0}   # user-MO eligible this slot
        hour = (u % DAY_SLOTS) // 3600
        afac = self.AF[hour]

        if u > 0 and u % c.gc_interval_slots == 0:
            self.ppl = [p for p in self.ppl if p.state != State.OUT]
        nn = np.random.poisson(self.arrival.intensity(u))
        for _ in range(nn):
            self.ppl.append(self._create_person(u))
        for p in self.ppl:
            if p.state == State.OUT and p.enter_time <= u < p.exit_time:
                p.state = State.IDLE
        # Exit
        for p in self.ppl:
            if p.state != State.OUT and u >= p.exit_time and not p.is_resident:
                if p.state == State.CONNECTED:
                    n_release += 1
                    self._log_disconnect(p.pid, u)
                if p.active_service is not None:
                    self._close_service_session(p, u, censored=True)
                if random.random() < c.rrc.detach_on_exit_prob:
                    n_detach_request += 1
                    n_ul_nas += 1
                self._log_event(p.pid, u, "gt_exit")
                p.state = State.OUT
                p.session_remaining = 0; p.inactivity_remaining = 0
                p.in_inactivity_wait = False; p.active_service = None
                p.in_service_idle = False

        # ---- collect this slot's access attempts (all go through RACH) ----
        # ctx: (person, is_bg, via_paging, forced_service, cause, origin)
        # INVARIANT: a UE is enqueued AT MOST ONCE per slot. Loop (a) clears
        # in_service_idle, so without a guard loop (b) would re-enqueue the
        # same UE via the (deliberately suspended, hence stale) bg clock —
        # double-connecting it and letting the second _finalize_connect
        # overwrite the ongoing service session. Resume takes precedence.
        queue: list[tuple] = []
        queued_ids: set[int] = set()

        # (a) multi-RRC service-idle expiry -> resume burst
        for p in self.ppl:
            if p.state == State.IDLE and p.in_service_idle:
                p.service_idle_remaining -= 1
                if p.service_idle_remaining <= 0:
                    if p.service_total_remaining > 0:
                        cfgs = c.traffic.service_types[p.active_service]
                        bl = min(max(1, int(np.random.normal(cfgs["burst_len_mean"],
                                                             cfgs["burst_len_std"]))),
                                 p.service_total_remaining)
                        p.service_burst_remaining = bl
                        p.in_service_idle = False
                        queue.append((p, False, False, None, "service_resume", "resume"))
                        queued_ids.add(id(p))
                    else:
                        # Defensive branch (cannot fire under current burst
                        # accounting): session exhausted exactly at idle
                        # expiry — restart the bg process as at final release.
                        self._close_service_session(p, u)
                        p.active_service = None; p.in_service_idle = False
                        p.next_bg_reconnect = u + self._sample_idle_gap(p.device_bg_class)

        # (b) IDLE triggers: bg-due (MO or MT-attributed), MT voice, user access
        ge = random.random() < c.arrival.global_event_prob
        pg = c.paging
        for p in self.ppl:
            if p.state != State.IDLE or p.in_service_idle or id(p) in queued_ids:
                continue
            ifac = pg.intensity_factor[p.usage_intensity]
            if u >= p.next_bg_reconnect:
                f_mt = min(0.9, pg.f_mt_base * ifac)
                if self._rng_page.random() < f_mt:        # MT / push-triggered wakeup
                    k = 1
                    while k < pg.page_retry_max and self._rng_page.random() >= pg.response_prob:
                        k += 1
                    n_paging_sent += k
                    self.page_stats["sent"] += k
                    self.page_k[k] = self.page_k.get(k, 0) + 1
                    for _ in range(k):
                        self._log_event(p.pid, u, "paging")
                    answered = (k < pg.page_retry_max) or \
                               (self._rng_page.random() < pg.response_prob)
                    if answered:
                        self.page_stats["answered"] += 1
                        if self._rng_page.random() < pg.p_engage:
                            svc = self._sample_service_excluding(("voice",), self._rng_page)
                            self.page_stats["engage"] += 1
                            queue.append((p, False, True, svc, "mt_engage", "bg"))
                        else:
                            self.page_stats["mt_data"] += 1
                            queue.append((p, True, True, None, "mt_data", "bg"))
                    else:
                        # all pages missed (~response^retries): fall back to MO
                        self.page_stats["mo_bg"] += 1
                        queue.append((p, True, False, None, "mo_bg", "bg"))
                else:
                    self.page_stats["mo_bg"] += 1
                    queue.append((p, True, False, None, "mo_bg", "bg"))
                continue
            # MT voice (incoming call) — low-rate, activity-modulated
            if self._rng_page.random() < self._mt_voice_rate[p.usage_intensity] * afac:
                k = 1
                while k < pg.page_retry_max and self._rng_page.random() >= pg.response_prob:
                    k += 1
                n_paging_sent += k
                self.page_stats["sent"] += k
                self.page_k[k] = self.page_k.get(k, 0) + 1
                for _ in range(k):
                    self._log_event(p.pid, u, "paging")
                answered = (k < pg.page_retry_max) or \
                           (self._rng_page.random() < pg.response_prob)
                if answered:
                    self.page_stats["answered"] += 1
                    self.page_stats["mt_voice"] += 1
                    queue.append((p, False, True, "voice", "mt_voice", "voice"))
                continue
            # user-initiated access (persona rate x diurnal activity)
            elig[p.usage_intensity] += 1
            pa = c.traffic.usage_intensity[p.usage_intensity]["access_prob"] * afac
            if ge:
                pa *= c.arrival.global_event_access_boost
            if random.random() < pa:
                queue.append((p, False, False, None, "user_mo", "user"))

        # ---- joint RACH for the whole slot ----
        if queue:
            winners, st = self._rach_procedure(len(queue))
            for k in st:
                self.rach_stats[k] += st[k]
        else:
            winners, st = [], dict(preamble_tx=0, rar=0, msg3=0, success=0,
                                   collided_tx=0, giveup=0)
        n_new_conn = 0
        for idx, (p, is_bg, via_pg, forced, cause, origin) in enumerate(queue):
            if winners[idx]:
                n_new_conn += 1
                r = self._finalize_connect(p, u, is_bg, via_pg, forced, cause)
                n_initial_ue_msg += r["initial_ue_msg"]; n_ul_nas += r["ul_nas"]
                n_dl_nas += r["dl_nas"]; n_bearer_activated += r["bearer"]
                n_rrc_reconfig += r["reconfig"]; n_attach_request += r["attach"]
                n_service_request += r["svc_req"]; n_paging_response += r["paging_resp"]
            else:
                # access failure after preamble_trans_max (rare at these loads)
                if origin == "bg":
                    p.next_bg_reconnect = u + self._sample_idle_gap(p.device_bg_class)
                elif origin == "resume":
                    p.in_service_idle = True; p.service_idle_remaining = 1
                # user / voice: trigger simply lost; later slots retrigger

        # ---- CONNECTED: meas reports + inactivity-timer release ----
        n_meas_report = 0
        for p in self.ppl:
            if p.state == State.CONNECTED:
                if random.random() < c.rrc.meas_report_prob_per_slot:
                    n_meas_report += 1
                if p.in_inactivity_wait:
                    p.inactivity_remaining -= 1
                    if p.inactivity_remaining <= 0:
                        p.state = State.IDLE
                        p.in_inactivity_wait = False
                        # RRCConnectionRelease (eNB->UE) + S1AP UE Context
                        # Release Request (eNB->MME) — see module docstring.
                        n_release += 1
                        n_release_request += 1
                        self._log_disconnect(p.pid, u)
                        if p.active_service and p.service_total_remaining > 0:
                            cfgs = c.traffic.service_types[p.active_service]
                            if cfgs.get("multi_rrc", False):
                                p.service_idle_remaining = max(
                                    1, int(np.random.exponential(cfgs["idle_len_mean"])))
                                p.in_service_idle = True
                            else:
                                self._close_service_session(p, u)
                                p.active_service = None
                        else:
                            self._close_service_session(p, u)
                            p.active_service = None
                        if not p.in_service_idle:
                            p.next_bg_reconnect = u + self._sample_idle_gap(p.device_bg_class)
                else:
                    p.session_remaining -= 1
                    if p.active_service:
                        p.service_total_remaining -= 1
                        p.service_burst_remaining -= 1
                    if p.session_remaining <= 0:
                        p.in_inactivity_wait = True
                        p.inactivity_remaining = self.RRC

        # ---- state breakdown ----
        n_connected = 0; n_idle = 0
        cls = {"low": [0, 0], "medium": [0, 0], "high": [0, 0]}
        for p in self.ppl:
            if p.state == State.CONNECTED:
                n_connected += 1; cls[p.usage_intensity][1] += 1
            elif p.state == State.IDLE:
                n_idle += 1; cls[p.usage_intensity][0] += 1
        n_present = n_idle + n_connected

        return {
            "t": u - self.warmup, "n_present": n_present, "n_connected": n_connected,
            "n_idle": n_idle, "n_new_connections": n_new_conn,
            "n_release": n_release, "n_release_request": n_release_request,
            "n_paging": n_paging_sent,
            "n_initial_ue_msg": n_initial_ue_msg, "n_ul_nas": n_ul_nas,
            "n_dl_nas": n_dl_nas, "n_bearer_activated": n_bearer_activated,
            "n_rrc_request": st["msg3"], "n_rrc_setup": st["success"],
            "n_rrc_setup_complete": st["success"], "n_rrc_reconfig": n_rrc_reconfig,
            "n_meas_report": n_meas_report,
            "n_rach_trigger": st["preamble_tx"], "n_rar": st["rar"],
            "n_rach_success": st["success"],
            "n_attach_request": n_attach_request, "n_detach_request": n_detach_request,
            "n_service_request": n_service_request,
            "n_paging_response": n_paging_response,
            "n_idle_low": cls["low"][0], "n_conn_low": cls["low"][1],
            "n_idle_medium": cls["medium"][0], "n_conn_medium": cls["medium"][1],
            "n_idle_high": cls["high"][0], "n_conn_high": cls["high"][1],
            "n_elig_low": elig["low"], "n_elig_medium": elig["medium"],
            "n_elig_high": elig["high"],
        }

    def _reset_diagnostics(self):
        """Zero warm-up accumulation so diagnostic counters (rach_stats,
        page_stats, cause_counts) describe the *recorded* window only and
        stay consistent with the returned DataFrame / fidelity metrics."""
        for k in self.rach_stats:
            self.rach_stats[k] = 0
        for k in self.page_stats:
            self.page_stats[k] = 0
        self.cause_counts.clear()
        self.page_k.clear()
        self.service_sessions.clear()

    def _open_recording_window(self):
        """Left-censoring at the warm-up boundary: UEs already CONNECTED when
        recording starts get a synthetic 'connect' at recorded t=0, so the
        per-UE event stream keeps strict connect/disconnect alternation for
        the downstream estimators. Counters (n_new_connections, cause_counts)
        are NOT touched — these are not new connections."""
        self._reset_diagnostics()
        self._censor_skip: set[int] = set()
        for p in self.ppl:
            if p.state == State.CONNECTED:
                # UEs that will provably release within the boundary slot
                # itself (inactivity expiry or cell exit at u==warmup) would
                # produce a same-t (connect, disconnect) pair, which the
                # shared event_key tie-break (disconnect < connect) inverts.
                # Drop that 1-second censored episode entirely instead:
                # skip the synthetic connect and suppress that one
                # disconnect log. Dynamics, counters and RNG are untouched.
                releases_now = (p.in_inactivity_wait
                                and p.inactivity_remaining <= 1)
                exits_now = (not p.is_resident
                             and p.exit_time <= self.warmup)
                if releases_now or exits_now:
                    self._censor_skip.add(p.pid)
                else:
                    self._log_event(p.pid, self.warmup, "connect")

    def run(self) -> pd.DataFrame:
        rows = []
        for u in range(self.total_u):
            if u == self.warmup:
                self._open_recording_window()
            row = self.step(u)
            if u >= self.warmup:
                rows.append(row)
        return pd.DataFrame(rows)
