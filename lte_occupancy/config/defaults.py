"""
config/defaults.py — Single source of truth for every research parameter.
PHASE-1 EDITION (real 24 h axis). This is a deliberate DGP re-baseline: the
legacy 30,000-slot compressed-cycle trajectory is NOT reproduced. Legacy RNG
compatibility shims (kept-but-unused draws, reject/ra_retry counter draws) are
therefore removed; the new baseline is pinned by tests/test_phase1.py instead.

Evidence classes used in comments below:
  [E] Empirical anchor  — value taken from a cited measurement (verified text).
  [S] Standards anchor  — 3GPP model/assumption.
  [D] Design assumption — direction motivated by literature, value ours.
  [C] Calibrated        — refined by experiments/validate_fidelity.py --calibrate.
  [V] Verify            — provisional; confirm against the primary source before
                          camera-ready (tracked in README_PHASE1).

Key sources (content verified for the bracketed figures):
  Xu et al., IMC 2015          [E] site peak/valley times, P/V ratios, wkday:wkend.
  Falaki et al., MobiSys 2010  [E] 10-200 phone-use sessions/day across users,
                                    median ~50; diurnal usage.
  Hintze et al., MoMM 2014     [E] mean 58 interactions/day (1,585 devices).
  Pielot et al., MobileHCI 2014[E] 63.5 notifications/day; viewed within minutes.
  3GPP TR 37.868               [S] RACH: 54 preambles, detect 1-e^-i, TransMax 10,
                                    collided Msg3 path.
  Rao et al., CoNEXT 2011      [E] streaming long ON-OFF cycles, OFF up to 80 s
                                    (mobile native apps).
  srsRAN epc.log (this work)   [E] per-class release->reconnect idle gaps.
"""
from __future__ import annotations

import os
from pathlib import Path

from .schema import (
    TimeCfg, TopologyCfg, ArrivalCfg, MobilityCfg, TrafficCfg, RRCCfg, RACHCfg,
    PagingCfg, RadioCfg, BackgroundCfg, ReleaseInferenceCfg, StmsiCfg,
    ObservationCfg, PerUeCfg, ModelCfg, FeatureCfg, SimConfig, ExperimentConfig,
)

_PKG_ROOT = Path(__file__).resolve().parent.parent   # .../lte_occupancy
DAY = 86_400

# =====================================================================
# Dwell classes — COMMON definitions (review §4: one realistic definition,
# sites vary only the MIXTURE). [D][V dwell-time literature pending]
#   transient ~5 min | regular ~40 min | stationary ~3 h ; floor 60 s.
# =====================================================================
_DWELL_MEANS = {"transient": 300, "regular": 2_400, "stationary": 10_800}
_MIN_DWELL = 60

# Site composition (T5b). Comprehensive == balanced == the T4/T5a anchor.
# Weights are [D] (transit->short-stay etc.); marked as such in T1.
_SITE_DWELL_MIX = {
    "comprehensive": {"transient": 0.20, "regular": 0.50, "stationary": 0.30},
    "resident":      {"transient": 0.10, "regular": 0.35, "stationary": 0.55},
    "office":        {"transient": 0.15, "regular": 0.60, "stationary": 0.25},
    "transport":     {"transient": 0.60, "regular": 0.30, "stationary": 0.10},
}
_SITE_USAGE_MIX = {   # persona weights per site [D]
    "comprehensive": {"low": 0.35, "medium": 0.45, "high": 0.20},
    "resident":      {"low": 0.35, "medium": 0.45, "high": 0.20},
    "office":        {"low": 0.25, "medium": 0.50, "high": 0.25},
    "transport":     {"low": 0.35, "medium": 0.45, "high": 0.20},
}
_SITE_SERVICE_MIX = {  # [D]
    "comprehensive": "balanced",
    "resident": "streaming_heavy",
    "office": "browsing_heavy",
    "transport": "extended",       # messaging-flavoured short interactions
}

# =====================================================================
# Cell scale — matched-occupancy targets (review §3/§7: Xu traffic-volume
# ratios are NOT used as occupancy ratios; scale set independently).
# arrival_base initialised via Little's law  n~ = lambda*E[D] + residents  [C]
# =====================================================================
SCALE_PROFILES = {
    #               target n_present   residents
    "tiny":   dict(n_target=45,   n_res=4),
    "small":  dict(n_target=85,   n_res=8),
    "medium": dict(n_target=160,  n_res=15),
    "large":  dict(n_target=320,  n_res=30),   # baseline / default
    "xlarge": dict(n_target=640,  n_res=60),
}

# =====================================================================
# Usage personas. A UE's class IS its user type: the class fixes how often that UE
# starts sessions, so the class mixture is what the usage-mix robustness axis varies.
#   low    ~15 sessions/day  "rarely checks the phone"   [E Falaki lower band]
#   medium ~55 sessions/day  "typical user"              [E Falaki median ~50;
#                                                            Hintze mean 58]
#   high  ~150 sessions/day  "heavy, frequently checks"  [E Falaki upper band]
# access_prob = sessions_per_day / active_seconds_per_day, where
# active_seconds = sum(activity_factor)*3600 (~16 h effective awake time). [C]
# =====================================================================
_PERSONA_SESSIONS_PER_DAY = {"low": 15.0, "medium": 55.0, "high": 150.0}
# 24 h activity multiplier (night suppressed; Falaki Fig.7 diurnal use [E->D]):
_ACTIVITY_FACTOR = (
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,   # 00-05
    0.30, 0.70,                            # 06,07 ramp-up
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 08-21
    0.70, 0.30,                            # 22,23 wind-down
)
_ACTIVE_SECONDS = sum(_ACTIVITY_FACTOR) * 3600.0     # ~ 16.3 h equivalents


def _service_types(mix_name: str) -> dict:
    """Service/session models, shaped from the measurement literature. The service a
    session draws sets its RRC signature, so the service mixture is what the
    service-mix robustness axis varies:
      voice     single-RRC call; duration ~N(120,45) s            [D][V holding-time]
      streaming multi-RRC long ON-OFF: ON~N(30,8), OFF~Exp(40) s  [E Rao 2011:
                mobile native long cycles, OFF up to 80 s]; total ~N(480,240) [D]
      browsing  page bursts ~N(8,3) + reading time Exp(30) s      [S 3GPP HTTP
                model, mean 30 s reading][V]; total ~N(300,150)   [D]
      messaging short bursts ~N(6,2) + Exp(45) idles; total ~N(180,90) [D]
    OFF idles exceed the 10 s inactivity timer by design -> real RRC
    release/re-setup cycling (the sniffer-visible signalling source)."""
    base = {
        "voice":     {"duration_mean": 120, "duration_std": 45, "multi_rrc": False},
        "streaming": {"duration_mean": 480, "duration_std": 240, "multi_rrc": True,
                      "burst_len_mean": 30, "burst_len_std": 8, "idle_len_mean": 40},
        "browsing":  {"duration_mean": 300, "duration_std": 150, "multi_rrc": True,
                      "burst_len_mean": 8, "burst_len_std": 3, "idle_len_mean": 30},
        "messaging": {"duration_mean": 180, "duration_std": 90, "multi_rrc": True,
                      "burst_len_mean": 6, "burst_len_std": 2, "idle_len_mean": 45},
    }
    mixes = {
        "balanced":        {"voice": 0.20, "streaming": 0.35, "browsing": 0.45, "messaging": 0.00},
        "voice_heavy":     {"voice": 0.50, "streaming": 0.25, "browsing": 0.25, "messaging": 0.00},
        "streaming_heavy": {"voice": 0.15, "streaming": 0.60, "browsing": 0.25, "messaging": 0.00},
        "browsing_heavy":  {"voice": 0.15, "streaming": 0.20, "browsing": 0.65, "messaging": 0.00},
        "extended":        {"voice": 0.15, "streaming": 0.25, "browsing": 0.35, "messaging": 0.25},
    }
    if mix_name not in mixes:
        raise ValueError(f"Unknown service mix: {mix_name!r}. Valid: {sorted(mixes)}")
    w = mixes[mix_name]
    return {name: {"weight": w[name], **base[name]} for name in base}


_BG_MIXES = {
    "balanced":     {"chatty": 0.36, "moderate": 0.29, "quiet_bg": 0.35},
    "chatty_heavy": {"chatty": 0.60, "moderate": 0.25, "quiet_bg": 0.15},
    "quiet_heavy":  {"chatty": 0.15, "moderate": 0.25, "quiet_bg": 0.60},
}

# Mode A observable counters (21 = legacy 20 + n_rar).
_MODE_A_FEATURES = (
    "n_connected", "n_new_connections", "n_release", "n_release_request",
    "n_paging", "n_initial_ue_msg", "n_ul_nas", "n_dl_nas", "n_bearer_activated",
    "n_rrc_request", "n_rrc_setup", "n_rrc_setup_complete", "n_rrc_reconfig",
    "n_meas_report", "n_rach_trigger", "n_rach_success",
    "n_attach_request", "n_detach_request", "n_service_request",
    "n_paging_response", "n_rar",
)
# Mode B (passive up+downlink sniffer, documented assumption): 8 counters —
# n_rach_trigger and n_rrc_request are now genuinely DISTINCT signals (M5).
_MODE_B_FEATURES = (
    "n_connected", "n_rrc_request", "n_rrc_setup", "n_rrc_setup_complete",
    "n_paging", "n_rach_trigger", "n_rach_success", "n_release_inferred",
)


def _xgb_device() -> str:
    try:
        from .device import XGB_DEVICE
        return XGB_DEVICE
    except Exception:
        return "cpu"


def build_config() -> ExperimentConfig:
    # ---------------- env knobs ----------------
    scale_profile = os.environ.get("LTE_SCALE_PROFILE", "large")
    if scale_profile not in SCALE_PROFILES:
        raise ValueError(f"Unknown scale profile: {scale_profile!r}")
    sp = SCALE_PROFILES[scale_profile]

    arrival_profile = os.environ.get("LTE_ARRIVAL_PROFILE", "comprehensive")
    site_comp = os.environ.get("LTE_SITE_COMPOSITION", "0") == "1"   # T5b on/off
    site = arrival_profile if site_comp else "comprehensive"

    days = int(os.environ.get("LTE_TOTAL_DAYS", 10))
    warmup_days = int(os.environ.get("LTE_WARMUP_DAYS", 1))
    total_slots = int(os.environ.get("LTE_TOTAL_TIME", days * DAY))
    warmup_slots = int(os.environ.get("LTE_WARMUP_SLOTS", warmup_days * DAY))
    weekend_on = os.environ.get("LTE_WEEKEND", "1" if site_comp else "0") == "1"

    mix_name = os.environ.get("LTE_SERVICE_MIX", _SITE_SERVICE_MIX[site] if site_comp else "balanced")
    stmsi_enabled = os.environ.get("LTE_STMSI_REALLOC", "0") == "1"
    emp_file = os.environ.get("LTE_EMPIRICAL_BG_FILE",
                              str(_PKG_ROOT / "empirical_idle_gap.json"))

    # sweep knobs (unchanged semantics)
    drx = float(os.environ.get("LTE_DRX_MISS", 0.05))
    rel_fn = float(os.environ.get("LTE_RELEASE_FN", 0.05))
    rel_fp = float(os.environ.get("LTE_RELEASE_FP", 0.02))
    rel_delay = float(os.environ.get("LTE_RELEASE_DELAY", 1.5))
    dwell_scale = float(os.environ.get("LTE_DWELL_SCALE", 1.0))
    bg_mix = os.environ.get("LTE_BG_MIX", "balanced")
    if bg_mix not in _BG_MIXES:
        raise ValueError(f"Unknown bg mix: {bg_mix!r}")
    xgb_jobs = int(os.environ.get("LTE_XGB_JOBS", 4))

    # calibration multipliers (validate_fidelity --calibrate writes these)
    arr_scale = float(os.environ.get("LTE_ARRIVAL_SCALE", 1.0))
    det_days = os.environ.get("LTE_DETERMINISTIC_DAYS", "0") == "1"
    # ^ calibration / T5a control runs: freeze day-to-day variability so the
    #   drift gate and single-day arrival calibration are well-posed.
    acc_scale = float(os.environ.get("LTE_ACCESS_SCALE", 1.0))

    # composition mix overrides for T11/T12 single-factor ablations
    dwell_mix_env = os.environ.get("LTE_DWELL_MIX", "")     # "t,r,s"
    usage_mix_env = os.environ.get("LTE_USAGE_MIX", "")     # "low,med,high"

    # ---------------- derived composition ----------------
    dwell_w = dict(_SITE_DWELL_MIX[site])
    if dwell_mix_env:
        t, r, s = (float(x) for x in dwell_mix_env.split(","))
        dwell_w = {"transient": t, "regular": r, "stationary": s}
    usage_w = dict(_SITE_USAGE_MIX[site])
    if usage_mix_env:
        lo, me, hi = (float(x) for x in usage_mix_env.split(","))
        usage_w = {"low": lo, "medium": me, "high": hi}

    e_dwell = sum(dwell_w[k] * _DWELL_MEANS[k] * dwell_scale for k in dwell_w)  # E[D] s
    n_res = int(os.environ.get("LTE_N_RESIDENTS", sp["n_res"]))
    # Little's-law initial arrival_base; pilot calibration refines via arr_scale [C]
    arrival_base = arr_scale * float(os.environ.get(
        "LTE_ARRIVAL_BASE", max(1e-6, (sp["n_target"] - n_res) / e_dwell)))
    n_init = int(os.environ.get("LTE_N_PEOPLE", round(arrival_base * e_dwell)))

    total_days_abs = (warmup_slots + total_slots + DAY - 1) // DAY
    # calendar: recorded day 0 == Monday; warm-up days precede it
    calendar = tuple((d + 7 - warmup_days % 7) % 7 for d in range(total_days_abs + 2))

    usage_intensity = {
        cls: {"weight": usage_w[cls],
              "access_prob": acc_scale * _PERSONA_SESSIONS_PER_DAY[cls] / _ACTIVE_SECONDS,
              "sessions_per_day_target": _PERSONA_SESSIONS_PER_DAY[cls]}
        for cls in ("low", "medium", "high")
    }

    xgb_dev = _xgb_device()
    _bgw = _BG_MIXES[bg_mix]

    sim = SimConfig(
        time=TimeCfg(
            slot_duration_s=1.0, total_slots=total_slots, train_ratio=0.8,
            seq_len=10, rolling_windows=(3, 5, 10, 60, 300),
            warmup_slots=warmup_slots,
        ),
        topology=TopologyCfg(n_cells=1, initial_nonresident_ues=n_init,
                             resident_ues=n_res),
        arrival=ArrivalCfg(
            base_rate=arrival_base,
            diurnal_period_slots=DAY,          # wave_period for t_sin/cos == 24 h
            commute_bumps=(),                  # legacy field, superseded by profile
            day_scale_range=(1.0, 1.0) if det_days else (0.85, 1.15),  # [D]
            day_scale_buffer_days=2,
            burst_events=(),                   # per-day random bursts instead
            global_event_prob=2e-5,            # ~1.7 surges/day [D, re-based for 1 s]
            global_event_access_boost=2.2,
            profile=arrival_profile,
            weekday_calendar=calendar,
            weekend_factor=1.0 if not weekend_on else 0.0,  # 0.0 => use profile table
            bursts_per_day=0 if det_days else 1,
            burst_amp_rel=1.4, burst_duration_s=(90.0, 20.0),
        ),
        mobility=MobilityCfg(
            dwell_classes={k: {"weight": dwell_w[k],
                               "mean_dwell": int(round(_DWELL_MEANS[k] * dwell_scale))}
                           for k in ("transient", "regular", "stationary")},
            min_dwell_slots=_MIN_DWELL, resident_exit_pad_slots=9_999,
        ),
        traffic=TrafficCfg(
            usage_intensity=usage_intensity,
            service_types=_service_types(mix_name),
            service_mix_name=mix_name,
            min_service_duration_slots=3,
            activity_factor=_ACTIVITY_FACTOR,
        ),
        rrc=RRCCfg(
            inactivity_timer_s=10,             # [E Huang 2013-class LTE timers][V]
            ra_retry_prob=0.0, reject_prob=0.0,  # deprecated: superseded by RACHCfg
            detach_on_exit_prob=0.025,
            meas_report_prob_per_slot=0.195,
            nas_ul_on_attach=5, nas_dl_on_attach=6,
        ),
        rach=RACHCfg(                          # [S TR 37.868]
            n_preambles=54, raos_per_slot=200,
            preamble_trans_max=10, collision_msg3=True,
        ),
        paging=PagingCfg(                      # M10 MT-cause model
            f_mt_base=0.05,    # ~63.5 notif/day [E Pielot] / measured reconnects [C]
            p_engage=0.25,     # notifications "viewed within minutes" [E Pielot -> D]
            mt_voice_per_day=3.0,              # incoming calls/day [D][V]
            response_prob=0.90, page_retry_max=3,
            intensity_factor={"low": 0.6, "medium": 1.0, "high": 1.5},
        ),
        radio=RadioCfg(dist_beta=(2.0, 5.0), rsrp_at_center_dbm=-60.0,
                       rsrp_edge_drop_db=40.0, shadowing_std_db=3.0),
        background=BackgroundCfg(
            empirical_idle_gap_file=emp_file,
            device_bg_class={
                "chatty":   {"weight": _bgw["chatty"], "bg_burst_mean": 10, "bg_burst_std": 4},
                "moderate": {"weight": _bgw["moderate"], "bg_burst_mean": 10, "bg_burst_std": 4},
                "quiet_bg": {"weight": _bgw["quiet_bg"], "bg_burst_mean": 10, "bg_burst_std": 4},
            },
        ),
    )

    observation = ObservationCfg(
        mode_a_features=_MODE_A_FEATURES,
        mode_b_features=_MODE_B_FEATURES,
        drx_miss_prob=drx,
        release_inference=ReleaseInferenceCfg(
            mean_delay_s=rel_delay, delay_std_s=0.5,
            false_negative_prob=rel_fn, false_positive_rate=rel_fp),
        stmsi=StmsiCfg(enabled=stmsi_enabled, realloc_mean_slots=10_800),
    )

    model = ModelCfg(
        xgb_reg=dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, n_jobs=xgb_jobs,
                     objective="reg:squarederror", tree_method="hist", device=xgb_dev),
        xgb_clf=dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=xgb_jobs,
                     eval_metric="logloss", tree_method="hist", device=xgb_dev),
        mode_b_xgb_overrides=dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                                  subsample=0.85, colsample_bytree=0.85, n_jobs=xgb_jobs),
        lstm=dict(hidden=64, dense=32, dropout=0.1, lr=1e-3,
                  batch=256, epochs=18, patience=4),
        per_ue=PerUeCfg(gc_interval=500, prune_elapsed=3_600, subsample=60,
                        n_clusters=3, survival_max_elapsed=7_200),
    )

    features = FeatureCfg(
        ewma_alphas=(0.1, 0.3), lag_offsets=(1, 2, 3),
        surge_window=600, surge_percentile=97.5, release_spike_percentile=97.5,
        turnover_windows=(60, 300), flow_cumulative_window=600,
        paging_windows=(60, 300),
    )

    return ExperimentConfig(
        name=f"phase1_{arrival_profile}{'_site' if site_comp else ''}_{scale_profile}",
        seeds=(101,), modes=("A", "B"), rrc_timer_sweep_s=(5, 10, 15, 30),
        scale_profile=scale_profile,
        simulation=sim, observation=observation, features=features, model=model,
    )
