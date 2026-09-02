"""Phase-1 invariants: arrival shapes, RACH counter chain, MT-attribution anchor
preservation, warm-up/day-split alignment. Run: pytest tests/test_phase1.py -q
"""
import importlib
import os

import numpy as np
import pytest

DAY = 86_400


def _fresh_config(**env):
    keep = {k: os.environ.get(k) for k in list(env)}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        import lte_occupancy.config.defaults as d
        importlib.reload(d)
        return d.build_config()
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


SMOKE = dict(LTE_TOTAL_TIME=7_200, LTE_WARMUP_SLOTS=600,
             LTE_SCALE_PROFILE="small", LTE_TOTAL_DAYS=10, LTE_WARMUP_DAYS=1)


def _run_sim(seed=7, **env):
    cfg = _fresh_config(**{**SMOKE, **env})
    from lte_occupancy.simulation.engine import Sim
    sim = Sim(seed, cfg.simulation)
    df = sim.run()
    return sim, df


# ---------------------------------------------------------------- shapes (T3)
def test_arrival_profiles_hit_xu_targets():
    from lte_occupancy.simulation import arrival_profiles as ap
    for r in ap.all_reports():
        assert abs(r["mean"] - 1.0) < 1e-9, r
        for a, t in zip(r["peaks_h_achieved"], r["peaks_h_target"]):
            assert abs(a - t) <= 0.30, r          # peak within +/-18 min
        ratio = r["pv_achieved"] / r["pv_target"]
        assert 1 / 1.5 <= ratio <= 1.5, r          # gate tolerance x/1.5
        assert r["valley_ok"], r                   # 3-6 h window holds the min


# ------------------------------------------------------- RACH chain (M5, T2)
def test_rach_counter_chain_and_smoke():
    sim, df = _run_sim()
    assert len(df) == SMOKE["LTE_TOTAL_TIME"]
    assert df["t"].iloc[0] == 0 and df["t"].iloc[-1] == SMOKE["LTE_TOTAL_TIME"] - 1
    a = df[["n_rach_trigger", "n_rar", "n_rrc_request",
            "n_rrc_setup", "n_rrc_setup_complete", "n_rach_success",
            "n_new_connections"]].to_numpy()
    assert (a[:, 0] >= a[:, 1]).all()              # preambles >= RAR
    # (no fixed RAR-vs-Msg3 order: RAR counts detected groups, Msg3 counts
    #  UEs, so collided groups legitimately give Msg3 > RAR)
    assert (a[:, 2] >= a[:, 3]).all()              # Msg3 >= Setup
    assert (a[:, 3] == a[:, 4]).all()              # Setup == SetupComplete
    assert (a[:, 4] == a[:, 5]).all()              # == rach_success
    assert (a[:, 5] == a[:, 6]).all()              # == new connections
    assert sim.rach_stats["success"] == int(df["n_new_connections"].sum())
    for cname in ("n_elig_low", "n_elig_medium", "n_elig_high"):
        assert cname in df.columns                 # persona-eligibility columns
    # elig is counted at queue-collection time (mid-slot); the class state
    # breakdown is end-of-slot, so a UE can be eligible then CONNECT within
    # the same slot. The correct invariant: eligible <= class present.
    assert (df["n_elig_low"].to_numpy()
            <= (df["n_idle_low"] + df["n_conn_low"]).to_numpy()).all()
    assert df["n_present"].min() > 0


def test_rach_collisions_emerge_under_load():
    """Force 300 simultaneous attempts through the joint RACH: collisions and
    Msg3>Setup must appear; give-up allowed but bounded."""
    cfg = _fresh_config(**SMOKE)
    from lte_occupancy.simulation.engine import Sim
    sim = Sim(1, cfg.simulation)
    # A single fixed-seed draw can legitimately yield 0 collisions
    # (~4 expected collision groups, ~63% detection at attempt 1), so we
    # assert emergence over an aggregate while keeping per-call invariants.
    agg = dict(preamble_tx=0, collided_tx=0, msg3=0, success=0)
    for _ in range(10):
        winners, st = sim._rach_procedure(300)
        assert st["preamble_tx"] >= 300
        assert st["rar"] <= st["preamble_tx"]
        assert st["msg3"] >= st["success"]
        assert sum(winners) == st["success"]
        assert st["giveup"] <= 300 - st["success"]
        for k in agg:
            agg[k] += st[k]
    assert agg["collided_tx"] > 0                  # collisions emerge under load
    assert agg["msg3"] > agg["success"]            # collided Msg3 path active


# ------------------------------------------- M10 anchor preservation (T2)
def test_fmt_sweep_preserves_truth_when_engage_zero():
    """With p_engage=0, f_mt only relabels bg wakeups MO->MT: paging counters
    change, but the truth trajectory (n_present, connects) must be identical —
    the srsRAN idle-gap anchor keeps fixing reconnection timing."""
    import lte_occupancy.config.defaults as d
    base = _fresh_config(**SMOKE)

    def with_paging(cfg, f_mt):
        from dataclasses import replace
        pg = replace(cfg.simulation.paging, f_mt_base=f_mt, p_engage=0.0,
                     mt_voice_per_day=0.0)
        simc = replace(cfg.simulation, paging=pg)
        return replace(cfg, simulation=simc)

    from lte_occupancy.simulation.engine import Sim
    dfs, pages = [], []
    for f in (0.0, 0.6):
        c = with_paging(base, f)
        s = Sim(11, c.simulation)
        df = s.run()
        dfs.append(df); pages.append(int(df["n_paging"].sum()))
    d0, d1 = dfs
    assert (d0["n_present"].to_numpy() == d1["n_present"].to_numpy()).all()
    assert (d0["n_new_connections"].to_numpy() == d1["n_new_connections"].to_numpy()).all()
    assert pages[0] == 0 and pages[1] > 0
    assert int(d1["n_paging_response"].sum()) > 0


# --------------------------------------------------- time axis (M1) / split
def test_warmup_exclusion_and_day_alignment():
    cfg = _fresh_config(LTE_TOTAL_DAYS=10, LTE_WARMUP_DAYS=1)
    t = cfg.simulation.time
    assert t.total_slots == 10 * DAY and t.warmup_slots == 1 * DAY
    train_end = t.train_ratio * t.total_slots
    assert train_end == 8 * DAY                    # 0.8 x 10 d is day-aligned
    cal = cfg.simulation.arrival.weekday_calendar
    assert cal[1] == 0                             # recorded day 0 == Monday
    assert cal[1 + 5] == 5 and cal[1 + 6] == 6     # Sat/Sun inside train days
    assert cfg.simulation.arrival.diurnal_period_slots == DAY  # t_sin/cos wiring


def test_personas_derive_from_targets():
    cfg = _fresh_config()
    ui = cfg.simulation.traffic.usage_intensity
    act = sum(cfg.simulation.traffic.activity_factor) * 3600.0
    for cls, tgt in (("low", 15.0), ("medium", 55.0), ("high", 150.0)):
        assert ui[cls]["sessions_per_day_target"] == tgt
        assert ui[cls]["access_prob"] == pytest.approx(tgt / act)


def test_paging_events_logged_for_perue_pipeline():
    sim, df = _run_sim(seed=3)
    kinds = {et for evs in sim.ue_events.values() for _, et in evs}
    assert "paging" in kinds                       # M11 tranche-2 hook is live
    # existing estimators whitelist connect/disconnect, so extra kinds are safe
    assert {"connect", "disconnect"} <= kinds


# ------------------------------------------- queue invariant (engine fix)
def test_one_enqueue_per_ue_per_slot_and_truth_alternation():
    """A UE must reach _finalize_connect at most once per slot (resume takes
    precedence over the suspended bg clock), and the resulting truth event
    stream must strictly alternate connect/disconnect per UE."""
    import collections
    from lte_occupancy.simulation import engine as E

    cfg = _fresh_config(**SMOKE)
    calls = collections.Counter()
    orig = E.Sim._finalize_connect

    def counting(self, p, u, *a, **k):
        calls[(p.pid, u)] += 1
        return orig(self, p, u, *a, **k)

    E.Sim._finalize_connect = counting
    try:
        sim = E.Sim(seed=7, cfg=cfg.simulation)
        sim.run()
    finally:
        E.Sim._finalize_connect = orig

    dups = {k: v for k, v in calls.items() if v > 1}
    assert not dups, f"{len(dups)} UEs finalized twice in one slot"

    from lte_occupancy.event_order import event_key
    cwc = dwd = 0
    for evs in sim.ue_events.values():
        connected = False
        for _t, et in sorted(evs, key=event_key):
            if et == "connect":
                if connected:
                    cwc += 1
                connected = True
            elif et == "disconnect":
                if not connected:
                    dwd += 1
                connected = False
    assert cwc == 0, f"{cwc} connect-while-connected in truth"
    assert dwd == 0, f"{dwd} disconnect-while-disconnected in truth"
