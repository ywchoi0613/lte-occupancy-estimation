"""
tests/test_correctness.py — invariants guarding the correctness properties.

Run with pytest, or standalone:  python -m tests.test_correctness
(Set a short horizon; the golden-hash test pins LTE_TOTAL_TIME itself.)
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from lte_occupancy.config.defaults import build_config
from lte_occupancy.simulation.engine import Sim
from lte_occupancy.observation.mode_a import ModeAObserver
from lte_occupancy.observation.mode_b import (
    ModeBObserver, install_stmsi_realloc_patch, _drx_degraded_connected,
    split_events_by_stmsi,
)
from lte_occupancy.estimation.survival import _gather_survival_rows
from lte_occupancy.config.schema import (
    ObservationCfg, ReleaseInferenceCfg, StmsiCfg,
)
from lte_occupancy.event_order import event_key

# golden truth hashes: seed=7, large, balanced, timer=10, LTE_TOTAL_TIME=2000,
# reject_prob=0.0. The dataframe hash was regenerated when reject_prob moved 0.01->0.0
# (only the n_rrc_reject/n_rrc_setup columns changed); ue_events/ue_meta are identical
# because the reject draw does not feed back into the truth trajectory.
# Phase 1 re-baseline (2026-07-27): new DGP — 10-day calendar axis + warm-up,
# Xu-calibrated arrival profiles, TR 37.868 joint RACH chain, persona-derived
# access rates, M10 paging redesign, one-enqueue-per-slot guard, boundary
# left-censoring. Legacy RNG-compat shims intentionally removed.
# Fixture pins: warmup=600, profile=comprehensive, scale=large, mix=balanced.
# v2 (review fixes): +n_elig_{low,medium,high} columns; events/meta verified
# unchanged, so only the dataframe hash moved.
GOLDEN = {"dataframe": "41f6d3b847877a73",
          "ue_events": "c8b9bb598a264c06",
          "ue_meta": "efcb09105af3f3d6"}


def _canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _truth_hashes(total_slots=2000, seed=7):
    os.environ["LTE_TOTAL_TIME"] = str(total_slots)
    os.environ["LTE_WARMUP_SLOTS"] = "600"          # pin: boundary logic in-hash
    os.environ["LTE_ARRIVAL_PROFILE"] = "comprehensive"  # pin against default drift
    os.environ["LTE_SCALE_PROFILE"] = "large"
    os.environ["LTE_SERVICE_MIX"] = "balanced"
    cfg = build_config()
    sim = Sim(seed=seed, cfg=cfg.simulation, rrc_timer=10)
    df = sim.run()
    return {"dataframe": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16],
            "ue_events": _canonical(sim.ue_events),
            "ue_meta": _canonical(sim.ue_meta)}


def test_truth_golden_hash():
    """The truth DGP must match the frozen golden hashes for the DataFrame AND the
    per-UE event log AND the per-UE metadata (not just the aggregate counters)."""
    assert _truth_hashes() == GOLDEN


def test_train_boundary_is_exclusive():
    """Survival must ignore events at t == train_end (test's first slot)."""
    train_end = 100
    # UE whose only activity is exactly at the boundary -> must be excluded
    events = {1: [(train_end, "connect"), (train_end + 5, "gt_exit")]}
    meta = {1: {"is_resident": False, "enter_time": 90, "exit_time": 200}}
    rows = _gather_survival_rows(events, meta, train_end)
    assert len(rows) == 0, "event at t==train_end leaked into the training rows"
    # UE active before the boundary -> present and right-censored (event=0)
    events2 = {2: [(50, "connect"), (60, "disconnect")]}
    meta2 = {2: {"is_resident": False, "enter_time": 40, "exit_time": 500}}
    rows2 = _gather_survival_rows(events2, meta2, train_end)
    assert len(rows2) == 1 and int(rows2.iloc[0]["event"]) == 0


def test_mode_b_never_uses_clean_connected_count():
    """Mode B naive baseline must use the DRX-degraded count, not clean n_connected."""
    cfg = build_config()
    df = pd.DataFrame({"n_connected": [50, 50, 50], "n_connected_b": [40, 41, 39],
                       "n_release": [1, 0, 2]})
    obs_b = ModeBObserver(cfg.observation, seed=7)
    df = obs_b.annotate(df)          # recomputes n_connected_b from truth
    naive = obs_b.naive_connected(df)
    assert np.array_equal(naive, df["n_connected_b"].values)
    assert not np.array_equal(naive, df["n_connected"].values)
    # Mode A, by contrast, legitimately uses the clean count
    obs_a = ModeAObserver(cfg.observation)
    assert np.array_equal(obs_a.naive_connected(df), df["n_connected"].values)


def test_release_inference_is_causal():
    """Inferred releases never precede the true release (non-negative lag)."""
    cfg = build_config()
    rel = np.zeros(200, dtype=int); rel[100] = 30    # 30 releases at t=100
    ncon = np.zeros(200, dtype=int)                  # no FPs
    obs_b = ModeBObserver(cfg.observation, seed=3)
    df = pd.DataFrame({"n_release": rel, "n_connected": ncon})
    df = obs_b.annotate(df)
    inf = df["n_release_inferred"].values
    assert inf[:100].sum() == 0, "a release was 'observed' before it happened"


def test_stmsi_does_not_change_truth():
    """S-TMSI reallocation must leave the cell trajectory identical (isolated RNG)."""
    os.environ["LTE_TOTAL_TIME"] = "800"
    cfg = build_config()
    df_plain = Sim(seed=11, cfg=cfg.simulation, rrc_timer=10).run()
    install_stmsi_realloc_patch(Sim, cfg.observation.stmsi.realloc_mean_slots)
    sim = Sim(seed=11, cfg=cfg.simulation, rrc_timer=10)
    sim._stmsi_realloc_mean = cfg.observation.stmsi.realloc_mean_slots
    df_stmsi = sim.run()
    for col in ["n_present", "n_connected", "n_new_connections", "n_release"]:
        assert np.array_equal(df_plain[col].values, df_stmsi[col].values), \
            f"S-TMSI perturbed truth column {col}"


def test_mode_b_per_ue_events_are_delayed_not_dropped():
    """Mode B's per-track releases are DELAYED (detection lag) but EVENTUALLY resolved:
    the episode-aware observer applies lag only, so no in-horizon session loses its
    release. Per-track FN/FP are intentionally NOT injected here — those noise sources
    live at the cell-counter level (n_release_inferred). This replaces the former
    'per-event FN drop' behaviour, which could leave tracks stuck connected."""
    cfg = build_config()
    obs_b = ModeBObserver(cfg.observation, seed=5)
    truth = {i: [(0, "connect"), (10, "disconnect"), (500, "gt_exit")] for i in range(400)}
    meta = {i: {"is_resident": False, "enter_time": 0, "exit_time": 500} for i in range(400)}
    obs_events, _ = obs_b.observe_ue_events(truth, meta, total_time=1000)
    disc_times = [t for evs in obs_events.values() for (t, et) in evs if et == "disconnect"]
    assert len(disc_times) == 400, "an in-horizon per-track release was dropped (must be lag-only)"
    assert any(t > 10 for t in disc_times), "no disconnect was delayed (detection lag not applied)"
    assert all(t >= 10 for t in disc_times), "a disconnect was observed before it happened"


def test_lstm_and_xgb_share_test_targets():
    """LSTM test targets must equal the full y[split_idx:] (same protocol as XGB)."""
    import torch
    if "stub" in getattr(torch, "__version__", ""):
        print("  (skipped: torch stub)"); return
    from lte_occupancy.estimation.models import train_lstm_model
    import numpy as _np
    n, split, seq = 200, 160, 10
    X = _np.random.RandomState(0).randn(n, 4).astype("float32")
    y = _np.arange(n).astype("float32")
    lstm_cfg = dict(hidden=4, dense=4, dropout=0.0, lr=1e-2, batch=16, epochs=1, patience=1)
    _, yte = train_lstm_model(X, y, split, seq, seed=0, device="cpu", lstm_cfg=lstm_cfg)
    assert len(yte) == n - split, "LSTM evaluated on a different-length test set than XGB"
    assert yte[0] == y[split], "LSTM test targets are misaligned with y[split_idx:]"
    """--timer/--seeds/--modes must be reflected in the effective config."""
    from types import SimpleNamespace
    from lte_occupancy.experiments.runner import _effective_config
    cfg = build_config()
    args = SimpleNamespace(timer=30, seeds=[7], modes=["B"])
    eff = _effective_config(cfg, args)
    assert eff.simulation.rrc.inactivity_timer_s == 30.0
    assert eff.seeds == (7,) and eff.modes == ("B",)


def test_unknown_scenario_names_raise():
    """Typos in scale/mix must fail loudly, not fall back silently."""
    for var, bad in [("LTE_SCALE_PROFILE", "huge"), ("LTE_SERVICE_MIX", "typo")]:
        old = os.environ.get(var)
        os.environ[var] = bad
        try:
            raised = False
            try:
                build_config()
            except ValueError:
                raised = True
            assert raised, f"bad {var}={bad!r} did not raise"
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


# ======================================================================
# Mode-B episode-aware per-track observation (observe_ue_events)
# ======================================================================
def _observer_fixed_delay(delay, seed=0):
    """A ModeBObserver whose per-track detection lag is DETERMINISTIC (= delay),
    achieved with delay_std_s=0. Per-track FN/FP are irrelevant to the episode-aware
    path (it never drops/injects per track), so they are set to 0."""
    cfg = ObservationCfg(
        mode_a_features=(), mode_b_features=(), drx_miss_prob=0.0,
        release_inference=ReleaseInferenceCfg(
            mean_delay_s=float(delay), delay_std_s=0.0,
            false_negative_prob=0.0, false_positive_rate=0.0),
        stmsi=StmsiCfg(enabled=False, realloc_mean_slots=10800),
    )
    return ModeBObserver(cfg, seed=seed)


def _ends_connected(events):
    """Replay the estimator's per-track state machine (canonical order) and return
    whether the track is left CONNECTED at the end of its event stream."""
    connected = False
    for (t, et) in sorted(events, key=event_key):
        if et == "connect":
            connected = True
        elif et == "disconnect":
            connected = False
    return connected


def test_delayed_release_does_not_terminate_new_session():
    """A previous session's delayed release must NOT close the next session; it is
    clamped to the next connect (episode-aware, not flat state-machine)."""
    obs = _observer_fixed_delay(delay=10)
    truth = {1: [(0, "connect"), (10, "disconnect"),
                 (15, "connect"), (30, "disconnect")]}
    ev, _ = obs.observe_ue_events(truth, {1: {"enter_time": 0, "exit_time": 100}},
                                  total_time=1000)
    # sess1 release 10+10=20 -> clamped to next connect 15; sess2 release 30+10=40
    assert ev[1] == [(0, "connect"), (15, "disconnect"),
                     (15, "connect"), (40, "disconnect")]


def test_release_crossing_next_connect_is_clamped():
    """An arbitrarily large lag cannot push a release past the next setup."""
    obs = _observer_fixed_delay(delay=50)
    truth = {1: [(0, "connect"), (10, "disconnect"),
                 (15, "connect"), (60, "disconnect")]}
    ev, _ = obs.observe_ue_events(truth, {1: {"enter_time": 0, "exit_time": 200}},
                                  total_time=1000)
    assert ev[1] == [(0, "connect"), (15, "disconnect"),
                     (15, "connect"), (110, "disconnect")]


def test_same_slot_disconnect_precedes_connect():
    """When a clamped release and the next setup coincide, disconnect comes first."""
    obs = _observer_fixed_delay(delay=5)
    truth = {1: [(0, "connect"), (10, "disconnect"),
                 (15, "connect"), (30, "disconnect")]}
    ev, _ = obs.observe_ue_events(truth, {1: {"enter_time": 0, "exit_time": 100}},
                                  total_time=1000)
    at15 = [et for (t, et) in ev[1] if t == 15]
    assert at15 == ["disconnect", "connect"], f"same-slot order wrong: {at15}"


def test_completed_in_horizon_session_not_stuck_connected():
    """A completed session whose delayed release is within the horizon MUST emit that
    release and end IDLE — this is the guard against the former stuck-connected bug."""
    obs = _observer_fixed_delay(delay=3)
    truth = {1: [(0, "connect"), (10, "disconnect")]}
    ev, _ = obs.observe_ue_events(truth, {1: {"enter_time": 0, "exit_time": 50}},
                                  total_time=1000)
    assert (13, "disconnect") in ev[1]
    assert _ends_connected(ev[1]) is False


def test_final_censored_session_may_remain_connected():
    """A completed session whose delayed release lands beyond the horizon is legitimate
    right-censoring: no release is observed yet, so the track may end CONNECTED. This is
    NOT the stuck bug (contrast with the in-horizon test above)."""
    obs = _observer_fixed_delay(delay=5)
    truth = {1: [(0, "connect"), (4998, "disconnect")]}
    ev, _ = obs.observe_ue_events(truth, {1: {"enter_time": 0, "exit_time": 5000}},
                                  total_time=5000)
    assert all(et != "disconnect" for (_, et) in ev[1])   # 4998+5=5003 >= horizon
    assert _ends_connected(ev[1]) is True


def test_no_illegal_transitions_after_episode_aware_observation():
    """Statistical sanity: on a real simulation, the episode-aware observed stream has
    ZERO connect-while-connected and ZERO disconnect-while-disconnected transitions
    (both were >1400 under the old per-event lag/FN observer)."""
    os.environ["LTE_TOTAL_TIME"] = "3000"
    os.environ["LTE_SCALE_PROFILE"] = "large"
    os.environ["LTE_SERVICE_MIX"] = "balanced"
    cfg = build_config()
    sim = Sim(seed=7, cfg=cfg.simulation, rrc_timer=10)
    sim.run()
    obs = ModeBObserver(cfg.observation, seed=7)
    ev, _ = obs.observe_ue_events(sim.ue_events, sim.ue_meta,
                                  total_time=cfg.simulation.time.total_slots)
    cwc = dwd = 0
    for _pid, evs in ev.items():
        connected = False
        for (t, et) in sorted(evs, key=event_key):
            if et == "connect":
                if connected:
                    cwc += 1
                connected = True
            elif et == "disconnect":
                if not connected:
                    dwd += 1
                connected = False
    assert cwc == 0, f"{cwc} connect-while-connected after episode-aware observation"
    assert dwd == 0, f"{dwd} disconnect-while-disconnected after episode-aware observation"


def test_reject_prob_zero_preserves_trajectory():
    """Phase 1 (M5): the legacy per-request reject/retry draws are replaced by
    the joint TR 37.868 RACH procedure, so `rrc.reject_prob` is an inert
    field — flipping it must leave EVERYTHING byte-identical. Access failure
    now arises only from the RACH itself, giving the chain inequality
    n_rach_trigger >= n_rar >= n_rrc_request(Msg3) >= n_rrc_setup(Msg4)
    instead of an identity between request and setup: Msg3 and Msg4 are two
    distinct observables, and Table S1 lists them separately."""
    from dataclasses import replace
    os.environ["LTE_TOTAL_TIME"] = "3000"
    os.environ["LTE_SCALE_PROFILE"] = "large"
    os.environ["LTE_SERVICE_MIX"] = "balanced"
    cfg = build_config()

    def _run(rp):
        sim_cfg = replace(cfg.simulation, rrc=replace(cfg.simulation.rrc, reject_prob=rp))
        sim = Sim(seed=7, cfg=sim_cfg, rrc_timer=10)
        return sim.run(), sim

    d01, s01 = _run(0.01)
    d00, s00 = _run(0.0)
    assert (d01["n_present"].values == d00["n_present"].values).all()
    assert (d01["n_connected"].values == d00["n_connected"].values).all()
    assert (d01["n_rrc_setup"].values == d00["n_rrc_setup"].values).all()
    assert _canonical(s01.ue_events) == _canonical(s00.ue_events)
    assert _canonical(s01.ue_meta) == _canonical(s00.ue_meta)
    a = d00[["n_rach_trigger", "n_rar", "n_rrc_request", "n_rrc_setup"]].to_numpy()
    assert (a[:, 0] >= a[:, 1]).all()
    # (no RAR>=Msg3 assert: RAR counts detected groups, Msg3 counts UEs,
    #  so collided groups legitimately give Msg3 > RAR)
    assert (a[:, 2] >= a[:, 3]).all()           # collided Msg3 >= Msg4 winners
    assert int((a[:, 2] - a[:, 3]).sum()) >= 0


def test_mode_a_ignores_stmsi_reallocation_events():
    """S-TMSI must affect Mode B tracks only. Mode A reads the raw ue_events, which
    carry s_tmsi_realloc when S-TMSI is ON; injecting those events must NOT change any
    Mode A per-UE output (survival fit, raw survival estimate, raw per-UE estimate)."""
    from lte_occupancy.estimation.survival import fit_clustered_empirical_survival
    from lte_occupancy.estimation.per_ue import compute_per_ue
    T, split = 400, 320
    base, meta = {}, {}
    for i in range(30):
        c = 10 + i; d = c + 40; x = min(d + 50, T - 1)
        base[i] = [(c, "connect"), (d, "disconnect"), (x, "gt_exit")]
        meta[i] = {"is_resident": False, "enter_time": c, "exit_time": x}
    with_tmsi = {pid: sorted(evs + [(50, "s_tmsi_realloc"), (150, "s_tmsi_realloc")], key=event_key)
                 for pid, evs in base.items()}

    def _run(ev):
        gs, pc, _ = fit_clustered_empirical_survival(ev, meta, split, n_clusters=3, max_elapsed=500)
        return compute_per_ue(ev, meta, gs, pc, None, T, 20, 300, None)

    s0, x0 = _run(base)
    s1, x1 = _run(with_tmsi)
    assert np.allclose(s0, s1), "Mode A survival estimate changed when s_tmsi_realloc events were added"
    assert np.allclose(x0, x1), "Mode A per-UE estimate changed when s_tmsi_realloc events were added"


if __name__ == "__main__":
    # tiny standalone runner (no pytest dependency)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} tests passed")
