"""
config/schema.py — Typed configuration schema (frozen dataclasses).

The whole pipeline is configured by a single ``ExperimentConfig`` object built in
``config/defaults.py``. Nothing in ``simulation/``, ``observation/`` or
``estimation/`` reads bare module-level constants; every parameter arrives through
one of these typed sub-configs. This is the single source of truth referenced by
``config/registry.py`` (which emits ``parameter_registry.csv``).

Unit convention: field names carry the unit (``*_s`` = seconds, ``*_slots`` =
slots, ``*_prob`` = probability). ``TimeCfg.seconds_to_slots`` is the ONE place
seconds are converted to slots, so the ``1 slot = slot_duration_s`` mapping is
explicit and centralised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# Time / topology
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TimeCfg:
    slot_duration_s: float          # s   — physical duration of one slot
    total_slots: int                # slots — run length
    train_ratio: float              # chronological train fraction
    seq_len: int                    # slots — LSTM input window
    rolling_windows: Tuple[int, ...]  # slots — rolling-feature windows
    warmup_slots: int = 0           # slots — burn-in simulated BEFORE t=0, not recorded

    def seconds_to_slots(self, seconds: float) -> int:
        """Canonical seconds->slots conversion (rounds) for integer-second scalar
        timers, e.g. the RRC inactivity timer. NOTE: not every time value routes
        through here — the empirical idle-gap is fractional-second data floored to
        slots (rounding it would alter the calibrated DGP), and Mode B release
        jitter is drawn in seconds. At slot_duration_s = 1 s these coincide."""
        return max(1, round(seconds / self.slot_duration_s))


@dataclass(frozen=True)
class TopologyCfg:
    n_cells: int                    # count (this model: 1)
    initial_nonresident_ues: int    # UEs seeded at t=0 (finite dwell)
    resident_ues: int               # UEs that never exit


# ----------------------------------------------------------------------
# Population dynamics
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ArrivalCfg:
    base_rate: float                # base arrival rate (per slot)
    diurnal_period_slots: int       # slots — one diurnal "day"
    # Gaussian commute bumps: (center_phase in [0,1), width, height-rel-to-base)
    commute_bumps: Tuple[Tuple[float, float, float], ...]
    day_scale_range: Tuple[float, float]  # per-day Uniform multiplier
    day_scale_buffer_days: int      # extra days appended to the day-scale vector
    # rectangular demand surges: (start_slot, duration_slots, extra_lambda)
    burst_events: Tuple[Tuple[int, int, float], ...]
    global_event_prob: float        # per-slot P(whole-cell surge active)
    global_event_access_boost: float  # x — access-prob multiplier during a surge
    # ---- Phase-1 (real 24h axis) fields ----
    profile: str = "comprehensive"  # site profile: resident|office|transport|comprehensive
    shape_minutes: Tuple[float, ...] = ()   # 1440-pt per-minute diurnal shape, mean == 1
    weekday_calendar: Tuple[int, ...] = ()  # weekday index per simulated day (0=Mon..6=Sun)
    weekend_factor: float = 1.0     # multiplier applied on weekend days (T5b only)
    bursts_per_day: int = 1         # random flash-crowd events per recorded day
    burst_amp_rel: float = 1.4      # burst extra-lambda, relative to base_rate
    burst_duration_s: Tuple[float, float] = (90.0, 20.0)  # N(mean, std) seconds


@dataclass(frozen=True)
class MobilityCfg:
    # dwell class name -> {"weight": w, "mean_dwell": slots}
    dwell_classes: Dict[str, Dict[str, float]]
    min_dwell_slots: int            # floor on any sampled dwell
    resident_exit_pad_slots: int    # residents' exit = total_slots + pad


# ----------------------------------------------------------------------
# Traffic / RRC / paging
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficCfg:
    # usage class -> {"weight": w, "access_prob": p}
    usage_intensity: Dict[str, Dict[str, float]]
    # service name -> {"weight", "duration_mean", "duration_std", "multi_rrc", ...}
    service_types: Dict[str, dict]
    service_mix_name: str
    min_service_duration_slots: int
    # per-hour user-activity multiplier (24 values). Used for user-initiated
    # access AND MT-voice arrival; night suppressed (Falaki 2010 diurnal use).
    activity_factor: Tuple[float, ...] = ()


@dataclass(frozen=True)
class RRCCfg:
    inactivity_timer_s: float       # s — connected->idle timeout with no traffic
    ra_retry_prob: float            # P(random-access preamble retry)
    reject_prob: float              # P(RRC setup rejected)
    detach_on_exit_prob: float      # P(explicit DETACH on leaving)
    meas_report_prob_per_slot: float  # per-slot P(connected UE emits meas report)
    nas_ul_on_attach: int           # UL NAS msgs on initial attach
    nas_dl_on_attach: int           # DL NAS msgs on initial attach


@dataclass(frozen=True)
class RACHCfg:
    """4-message RACH per 3GPP TR 37.868 assumptions: contention preambles,
    per-attempt detection prob 1-exp(-i) (power ramping), Msg3 also sent by
    collided UEs the eNB could not discern at preamble stage, contention
    resolved at Msg4, give-up after preamble_trans_max attempts."""
    n_preambles: int                # contention preambles per RAO (TR 37.868: 54)
    raos_per_slot: int              # RACH opportunities per 1 s slot (PRACH cfg 6: 200)
    preamble_trans_max: int         # max preamble transmissions (preambleTransMax: 10)
    collision_msg3: bool            # collided-but-detected UEs still transmit Msg3


@dataclass(frozen=True)
class PagingCfg:
    """MT-cause paging (Phase-1 M10). Paging volume is ATTRIBUTED to the
    empirically-timed background reconnect process (fraction f_mt is
    network-initiated / push-triggered) instead of an independent Bernoulli, so
    the srsRAN idle-gap anchor keeps fixing total reconnection rate."""
    f_mt_base: float                # P(a bg reconnect was MT / push-triggered)
    p_engage: float                 # P(user opens the app after an MT wakeup)
    mt_voice_per_day: float         # incoming (MT) voice calls per UE per active day
    response_prob: float            # P(UE answers one page attempt)
    page_retry_max: int             # MME paging repetitions (T3413 retries)
    intensity_factor: Dict[str, float]  # x, by usage class (heavier user => more MT)


@dataclass(frozen=True)
class RadioCfg:
    """Radio geometry / RSRP. GENERATED per UE but NOT consumed by any feature or
    model in this edition (a hook for future radio-aware work). Kept so the RNG
    stream — and therefore every result — is identical to the reference run; the
    draws are part of UE creation. Removing them is a separate, DGP-changing
    decision, not a cosmetic one."""
    dist_beta: Tuple[float, float]  # Beta(a,b) for normalized distance in [0,1]
    rsrp_at_center_dbm: float       # dBm at cell centre
    rsrp_edge_drop_db: float        # dB drop centre->edge
    shadowing_std_db: float         # dB Gaussian shadowing


# ----------------------------------------------------------------------
# Background reconnection
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BackgroundCfg:
    empirical_idle_gap_file: str    # required per-class idle-gap pools (srsRAN epc.log)
    # bg class -> {"weight", "bg_burst_mean", "bg_burst_std"} (slots)
    device_bg_class: Dict[str, Dict[str, float]]


# ----------------------------------------------------------------------
# Observation model (truth -> what each mode can see)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ReleaseInferenceCfg:
    mean_delay_s: float             # s — mean detection lag after true release
    delay_std_s: float              # s — jitter on the lag
    false_negative_prob: float      # P(a true release is missed)
    false_positive_rate: float      # per-connected-UE spurious-release rate


@dataclass(frozen=True)
class StmsiCfg:
    enabled: bool
    realloc_mean_slots: int         # mean S-TMSI reallocation interval (slots)


@dataclass(frozen=True)
class ObservationCfg:
    # Mode A: cooperative eNB — full observable counters (incl. n_paging_response).
    mode_a_features: Tuple[str, ...]
    # Mode B: passive sniffer — recoverable subset.
    mode_b_features: Tuple[str, ...]
    drx_miss_prob: float            # fraction of connected UEs invisible per slot (DRX)
    release_inference: ReleaseInferenceCfg
    stmsi: StmsiCfg


# ----------------------------------------------------------------------
# Models / estimator hyper-parameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PerUeCfg:
    gc_interval: int                # slots — prune OUT UEs every N
    prune_elapsed: int              # slots — per-UE feature horizon
    subsample: int                  # keep 1/N per-UE samples
    n_clusters: int                 # UE behaviour clusters
    survival_max_elapsed: int       # slots — max elapsed in survival table


@dataclass(frozen=True)
class ModelCfg:
    xgb_reg: dict                   # XGBoost regressor params
    xgb_clf: dict                   # XGBoost classifier params
    mode_b_xgb_overrides: dict      # Mode-B-specific XGB overrides
    lstm: dict                      # LSTM hyper-parameters
    per_ue: PerUeCfg


@dataclass(frozen=True)
class FeatureCfg:
    """Feature-engineering hyper-parameters (affect model inputs -> results)."""
    ewma_alphas: Tuple[float, ...]
    lag_offsets: Tuple[int, ...]
    surge_window: int               # slots — trailing window for the surge baseline
    surge_percentile: float         # train-only percentile defining a surge
    release_spike_percentile: float # train-only percentile defining a release spike
    turnover_windows: Tuple[int, ...]
    flow_cumulative_window: int
    paging_windows: Tuple[int, ...]


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """Everything the TRUTH simulator needs (no observation noise here)."""
    time: TimeCfg
    topology: TopologyCfg
    arrival: ArrivalCfg
    mobility: MobilityCfg
    traffic: TrafficCfg
    rrc: RRCCfg
    paging: PagingCfg
    radio: RadioCfg
    background: BackgroundCfg
    rach: RACHCfg = None            # set by defaults.build_config()
    gc_interval_slots: int = 500    # slots — prune OUT UEs every N (housekeeping; no result impact)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: Tuple[int, ...]
    modes: Tuple[str, ...]
    rrc_timer_sweep_s: Tuple[int, ...]   # sweep spec (driven via CLI/shell, documented here)
    scale_profile: str
    simulation: SimConfig
    observation: ObservationCfg
    features: FeatureCfg
    model: ModelCfg
