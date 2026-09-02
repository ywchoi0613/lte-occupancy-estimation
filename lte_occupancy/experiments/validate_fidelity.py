#!/usr/bin/env python
"""Phase 1 fidelity validation & calibration — gates G2/G3/G4 (+G5 pointer).

Modes
-----
calibrate : iterate LTE_ARRIVAL_SCALE on ONE scenario until mean n_present
            hits --target-nbar (Little's-law update, measured-occupancy
            criterion); optionally nudge LTE_ACCESS_SCALE back to the nominal
            persona targets (--calibrate-access). Writes
            <out>/calibration_<scenario>.json (consumed by gate/retune).
gate      : run ONE scenario with the stored calibration and score the gates;
            writes <out>/<scenario>/metrics.json (+ console PASS/FAIL lines).
report    : aggregate all metrics.json -> <out>/GATE_REPORT.md, add the
            cross-scenario n-bar gate, print OVERALL verdict; exit 1 on FAIL.

Phase-1 sequence (run_phase1_pilot.sh drives this):
    python experiments/validate_fidelity.py --mode calibrate --scenario comprehensive
    for s in resident office transport comprehensive; do
        python experiments/validate_fidelity.py --mode gate --scenario $s --days 2
    done
    python experiments/validate_fidelity.py --mode report

Conventions
-----------
* Warm-up is always WHOLE days (engine M1 contract); recorded day 0 = Monday.
* Pilot/gate runs use LTE_WEEKEND=0 and balanced composition (T5a discipline:
  arrival *shape* is the only moving part; the weekly calendar belongs to the
  headline DGP and T14). Pilot/gate/calibration runs additionally set
  LTE_DETERMINISTIC_DAYS=1 (day_scale=(1,1), no flash bursts) so single-day
  arrival calibration and the daily-drift gate are well-posed; the headline
  DGP re-enables both.
* Persona gate is mechanism-exact: the engine exports per-class eligibility
  counts (n_elig_*), and E[MO_cls] = access_prob_cls * sum_t afac(t) *
  n_elig_cls(t) holds by construction (a UE cannot initiate while CONNECTED
  or in service-idle). Residents' sessions/day vs the nominal Falaki targets
  are reported alongside (ordering gated; level informational).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

DAY = 86400
PERSONAS = ("low", "medium", "high")
SCENARIOS = ("resident", "office", "transport", "comprehensive")
VALLEY_H = (3.0, 6.0)


# --------------------------------------------------------------------------
# environment / run helpers
# --------------------------------------------------------------------------
def _apply_env(scenario: str, days: int, warmup_days: int,
               arrival_scale: float, access_scale: float) -> None:
    os.environ["LTE_ARRIVAL_PROFILE"] = scenario
    os.environ["LTE_TOTAL_TIME"] = str(days * DAY)
    os.environ["LTE_WARMUP_SLOTS"] = str(warmup_days * DAY)
    os.environ["LTE_ARRIVAL_SCALE"] = repr(float(arrival_scale))
    os.environ["LTE_ACCESS_SCALE"] = repr(float(access_scale))
    os.environ["LTE_WEEKEND"] = "0"            # T5a discipline (see docstring)
    os.environ["LTE_DETERMINISTIC_DAYS"] = "1"  # freeze day_scale / flash bursts
    os.environ["LTE_SITE_COMPOSITION"] = "0"   # balanced composition
    os.environ["LTE_SCALE_PROFILE"] = os.environ.get("LTE_SCALE_PROFILE", "large")
    os.environ["LTE_SERVICE_MIX"] = os.environ.get("LTE_SERVICE_MIX", "balanced")


def _run(seed: int):
    from lte_occupancy.config.defaults import build_config
    from lte_occupancy.simulation.engine import Sim
    cfg = build_config()
    t0 = time.time()
    sim = Sim(seed=seed, cfg=cfg.simulation)
    df = sim.run()
    dt = time.time() - t0
    return cfg, sim, df, dt


def _circ_dist_h(a: float, b: float) -> float:
    d = abs(a - b) % 24.0
    return min(d, 24.0 - d)


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def measure(cfg, sim, df, days: int, elapsed_s: float, scenario: str) -> dict:
    m: dict = {"scenario": scenario, "days": days}
    npres = df["n_present"].to_numpy(dtype=float)
    m["nbar"] = float(npres.mean())
    day_means = [float(npres[d * DAY:(d + 1) * DAY].mean()) for d in range(days)]
    m["nbar_by_day"] = day_means
    m["drift_rel"] = (abs(day_means[-1] / day_means[0] - 1.0)
                      if days >= 2 and day_means[0] > 0 else None)

    # ---- personas (G2-a): mechanism-exact expectation via eligibility ----
    # The engine draws user_mo with prob access_prob*afac for exactly the UEs
    # counted in n_elig_<cls> (IDLE, not service-idle, not already triggered),
    # so E[MO_cls] = access_prob_cls * sum_t afac(t)*n_elig_cls(t) holds by
    # construction (global-event boost, ~2e-5/slot, is negligible).
    tr = cfg.simulation.traffic
    af = np.asarray(tr.activity_factor, dtype=float)
    af_slot = np.tile(np.repeat(af, 3600), days)[: len(df)]
    residents = [p for p in sim.ppl if getattr(p, "is_resident", False)]
    cls_of = {pid: mm.get("usage_intensity") for pid, mm in sim.ue_meta.items()}
    mo_total = {c: 0 for c in PERSONAS}
    for pid, cc in sim.cause_counts.items():
        c0 = cls_of.get(pid)
        if c0 in mo_total:
            mo_total[c0] += int(cc.get("user_mo", 0))
    per = {}
    for cls in PERSONAS:
        ap_ = float(tr.usage_intensity[cls]["access_prob"])
        elig = df["n_elig_" + cls].to_numpy(dtype=float)
        expected = ap_ * float((af_slot * elig).sum())
        res_rates = [sim.cause_counts.get(p.pid, {}).get("user_mo", 0) / days
                     for p in residents if p.usage_intensity == cls]
        per[cls] = {
            "target_nominal": float(
                tr.usage_intensity[cls]["sessions_per_day_target"]),
            "expected_mo_total": expected,
            "measured_mo_total": float(mo_total[cls]),
            "mech_ratio": (mo_total[cls] / expected
                           if expected > 0 else float("nan")),
            "resident_sessions_per_day": (float(np.mean(res_rates))
                                          if res_rates else float("nan")),
            "n_resident_ues": int(len(res_rates)),
        }
    m["personas"] = per

    # ---- RACH (G2-b) ----
    rs = {k: int(v) for k, v in sim.rach_stats.items()}
    rs["collision_rate"] = rs["collided_tx"] / max(rs["preamble_tx"], 1)
    rs["detect_ratio"] = rs["rar"] / max(rs["preamble_tx"], 1)
    rs["success_per_day"] = rs["success"] / days
    m["rach"] = rs

    # ---- paging (G2-c) ----
    ps = {k: int(v) for k, v in sim.page_stats.items()}
    wake_mt = ps["mt_data"] + ps["engage"]
    wake = wake_mt + ps["mo_bg"]
    ps["mt_share_of_bg_wakeups"] = wake_mt / max(wake, 1)
    # Mechanism-exact expectation: the engine draws the MT-voice hazard only
    # for UEs that reach the eligibility point of loop (b) (plain IDLE — not
    # service-idle, not queued, not bg/mt-triggered this slot), i.e. exactly
    # the population counted in n_elig_<cls> (UEs that fired mt_voice are the
    # sole exclusion, ~0.003% of eligible UE-slots — negligible). Using
    # n_idle here would overcount by the service-idle share (~45%), which is
    # precisely the v2 G2a bug reappearing in G2c. The nominal
    # mt_voice_per_day is therefore the plain-IDLE-conditioned rate; realized
    # completed calls/UE/day = nominal x eligible-time fraction (a stated
    # modeling choice: incoming calls terminate only in plain IDLE).
    exp_mtv = 0.0
    for cls in PERSONAS:
        rate = float(sim._mt_voice_rate[cls])
        el = df["n_elig_" + cls].to_numpy(dtype=float)
        exp_mtv += float((rate * af_slot * el).sum())
    ps["mt_voice_expected_triggers"] = exp_mtv
    ps["mt_voice_ratio"] = ps["mt_voice"] / exp_mtv if exp_mtv > 0 else float("nan")
    ps["mt_voice_completed_per_ue_day"] = ps["mt_voice"] / max(m["nbar"] * days, 1e-9)
    ps["pages_per_ue_day"] = ps["sent"] / max(m["nbar"] * days, 1e-9)
    ps["retry_hist"] = {str(k): int(v)
                        for k, v in sorted(getattr(sim, "page_k", {}).items())}
    m["paging"] = ps

    # ---- extended fidelity (T2): causes, chain ratios, service sessions ----
    causes: dict = {}
    for cc in sim.cause_counts.values():
        for k, v in cc.items():
            causes[k] = causes.get(k, 0) + int(v)
    m["cause_breakdown"] = causes
    rs["msg3_per_setup"] = rs["msg3"] / max(rs["success"], 1)
    rs["preamble_per_setup"] = rs["preamble_tx"] / max(rs["success"], 1)
    svc: dict = {}
    for r in getattr(sim, "service_sessions", []):
        d = svc.setdefault(r["service"], {"n": 0, "censored": 0, "wall": [],
                                          "active": [], "bursts": []})
        d["n"] += 1
        if r["censored"]:
            d["censored"] += 1
        else:
            d["wall"].append(r["wall_s"])
            d["active"].append(r["active_s"])
            d["bursts"].append(r["bursts"])
    m["service_sessions"] = {
        s: {"n": d["n"], "censored": d["censored"],
            "wall_mean_s": float(np.mean(d["wall"])) if d["wall"] else None,
            "wall_p50_s": float(np.median(d["wall"])) if d["wall"] else None,
            "active_mean_s": (float(np.mean(d["active"]))
                              if d["active"] else None),
            "bursts_mean": float(np.mean(d["bursts"])) if d["bursts"] else None}
        for s, d in svc.items()}

    # ---- realized arrival shape (G3) ----
    from lte_occupancy.simulation.arrival_profiles import profile_report
    rep = profile_report(scenario)
    ent = [meta["enter_time"] for meta in sim.ue_meta.values()
           if not meta.get("is_resident", False) and meta["enter_time"] >= 0]
    nb = 48                                    # 30-min bins
    cnt = np.zeros(nb)
    for e in ent:
        cnt[int((int(e) % DAY) // 1800)] += 1.0
    sm = (np.roll(cnt, 1) + cnt + np.roll(cnt, -1)) / 3.0
    hours = (np.arange(nb) + 0.5) * 0.5
    peaks_real = []
    for pk in rep["peaks_h_target"]:
        mask = np.array([_circ_dist_h(h, pk) <= 3.0 for h in hours])
        peaks_real.append(float(hours[int(np.argmax(np.where(mask, sm, -np.inf)))]))
    vwin = (hours >= VALLEY_H[0]) & (hours <= VALLEY_H[1])
    m["shape"] = {
        "lut_report": rep,
        "peaks_h_realized": peaks_real,
        "pv_realized": float(sm.max() / max(sm.min(), 1e-9)),
        "arrivals_per_day": len(ent) / days,
        "valley_window_arrivals": float(cnt[vwin].sum()),
    }

    # ---- timing (G4) ----
    total_slots = int(cfg.simulation.time.warmup_slots) + days * DAY
    sps = total_slots / max(elapsed_s, 1e-9)
    m["timing"] = {"elapsed_s": elapsed_s, "total_slots": total_slots,
                   "slots_per_s": sps,
                   "full_run_est_s": (int(cfg.simulation.time.warmup_slots)
                                      + 10 * DAY) / sps}
    m["env"] = {k: os.environ.get(k, "") for k in
                ("LTE_ARRIVAL_PROFILE", "LTE_ARRIVAL_SCALE", "LTE_ACCESS_SCALE",
                 "LTE_WEEKEND", "LTE_SITE_COMPOSITION", "LTE_SCALE_PROFILE",
                 "LTE_SERVICE_MIX", "LTE_TOTAL_TIME", "LTE_WARMUP_SLOTS")}
    return m


# --------------------------------------------------------------------------
# gate scoring
# --------------------------------------------------------------------------
def score(m: dict, cfg, args) -> dict:
    g: dict = {}

    per = m["personas"]
    ok = True
    details = []
    for cls in PERSONAS:
        p = per[cls]
        if not np.isfinite(p["mech_ratio"]):
            ok = False
            details.append(cls + ": no data")
            continue
        rel = p["mech_ratio"] - 1.0
        details.append("%s: mech %+.1f%% | res/day %.1f (nom %.0f)"
                       % (cls, 100 * rel, p["resident_sessions_per_day"],
                          p["target_nominal"]))
        if abs(rel) > 0.10:
            ok = False
    meas = [per[c]["resident_sessions_per_day"] for c in PERSONAS]
    order_ok = bool(np.isfinite(meas).all() and meas[0] < meas[1] < meas[2])
    g["G2a_persona_sessions"] = {"pass": bool(ok and order_ok),
                                 "order_ok": order_ok, "detail": details}

    rs = m["rach"]
    g["G2b_rach"] = {"pass": bool(rs["collision_rate"] < 0.01 and rs["giveup"] == 0),
                     "collision_rate": rs["collision_rate"],
                     "giveup": rs["giveup"]}

    ps = m["paging"]
    f_mt = float(cfg.simulation.paging.f_mt_base)
    share_ok = (0.5 * f_mt) <= ps["mt_share_of_bg_wakeups"] <= (2.0 * f_mt)
    mtv_ok = (np.isfinite(ps["mt_voice_ratio"])
              and 0.75 <= ps["mt_voice_ratio"] <= 1.33)
    g["G2c_paging"] = {"pass": bool(share_ok and mtv_ok),
                       "mt_share": ps["mt_share_of_bg_wakeups"],
                       "f_mt_base": f_mt, "mt_voice_ratio": ps["mt_voice_ratio"]}

    sh = m["shape"]
    rep = sh["lut_report"]
    peak_ok = all(_circ_dist_h(rz, tg) <= 0.5
                  for rz, tg in zip(sh["peaks_h_realized"], rep["peaks_h_target"]))
    lut_pv_ok = (rep["pv_achieved"] <= rep["pv_target"] * 1.5
                 and rep["pv_achieved"] >= rep["pv_target"] / 1.5)
    pv_real_note = "n/a"
    if sh["valley_window_arrivals"] >= 20:
        r = sh["pv_realized"] / max(rep["pv_achieved"], 1e-9)
        pv_real_note = "%.2fx of LUT (WARN only)" % r
    g["G3_shape"] = {"pass": bool(peak_ok and lut_pv_ok
                                  and rep.get("valley_ok", True)),
                     "peaks_realized": sh["peaks_h_realized"],
                     "peaks_target": list(rep["peaks_h_target"]),
                     "lut_pv": rep["pv_achieved"], "pv_target": rep["pv_target"],
                     "pv_realized_note": pv_real_note}

    nbar_ok = abs(m["nbar"] / args.target_nbar - 1.0) <= 0.05
    drift = m["drift_rel"]
    drift_ok = True if drift is None else drift <= 0.03
    g["G3_nbar"] = {"pass": bool(nbar_ok and drift_ok), "nbar": m["nbar"],
                    "target": args.target_nbar, "drift_rel": drift}

    t = m["timing"]
    total_wall_days = (t["full_run_est_s"] * args.total_runs
                       / max(args.workers, 1)) / 86400.0
    g["G4_compute"] = {"pass": True, "slots_per_s": t["slots_per_s"],
                       "full_run_est_min": t["full_run_est_s"] / 60.0,
                       "runs": args.total_runs, "workers": args.workers,
                       "est_wall_days": total_wall_days,
                       "note": ("consider 5-seed cut for T5b/T11/T12"
                                if total_wall_days > 5 else "budget OK")}
    return g


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def mode_calibrate(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scale = args.arrival_scale_init
    acc = args.access_scale_init
    hist = []
    cfg = sim = df = None
    for it in range(1, args.max_iter + 1):
        _apply_env(args.scenario, args.cal_days, args.warmup_days, scale, acc)
        cfg, sim, df, dt = _run(args.seed)
        n_res = int(cfg.simulation.topology.resident_ues)
        nbar = float(df["n_present"].mean())
        err = nbar / args.target_nbar - 1.0
        hist.append({"iter": it, "arrival_scale": scale, "nbar": nbar,
                     "rel_err": err, "elapsed_s": dt})
        print("[calibrate] it=%d scale=%.5f nbar=%.2f err=%+.2f%% (%.0fs)"
              % (it, scale, nbar, 100 * err, dt))
        if abs(err) <= 0.02:
            break
        scale *= (args.target_nbar - n_res) / max(nbar - n_res, 1.0)

    if args.calibrate_access and cfg is not None:
        mm = measure(cfg, sim, df, args.cal_days, hist[-1]["elapsed_s"],
                     args.scenario)
        ratios = []
        for cls in PERSONAS:
            p = mm["personas"][cls]
            r = p["resident_sessions_per_day"]
            if np.isfinite(r) and r > 0:
                # restore nominal Falaki resident rates: scale access_prob by
                # target/measured (mechanism ratio stays exact by construction)
                ratios.append(p["target_nominal"] / r * acc)
        if ratios:
            acc = float(np.mean(ratios))
            print("[calibrate] access_scale -> %.4f "
                  "(nominal-target restoration)" % acc)

    payload = {"scenario": args.scenario, "target_nbar": args.target_nbar,
               "LTE_ARRIVAL_SCALE": scale, "LTE_ACCESS_SCALE": acc,
               "achieved_nbar": hist[-1]["nbar"], "iters": len(hist),
               "history": hist, "warmup_days": args.warmup_days,
               "cal_days": args.cal_days, "seed": args.seed}
    path = out / ("calibration_%s.json" % args.scenario)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print("[calibrate] wrote %s" % path)
    converged = abs(hist[-1]["rel_err"]) <= 0.02
    if not converged:
        print("[calibrate] WARNING: not converged within max-iter")
    return 0 if converged else 1


def _load_calibration(out: Path, scenario: str) -> tuple[float, float, str]:
    for name in ("calibration_%s.json" % scenario,
                 "calibration_comprehensive.json"):
        p = out / name
        if p.exists():
            d = json.loads(p.read_text())
            return (float(d["LTE_ARRIVAL_SCALE"]),
                    float(d["LTE_ACCESS_SCALE"]), name)
    return 1.0, 1.0, "none (defaults 1.0)"


def mode_gate(args) -> int:
    out = Path(args.out)
    (out / args.scenario).mkdir(parents=True, exist_ok=True)
    a_scale, x_scale, src = _load_calibration(out, args.scenario)
    print("[gate:%s] calibration source: %s" % (args.scenario, src))
    _apply_env(args.scenario, args.days, args.warmup_days, a_scale, x_scale)
    cfg, sim, df, dt = _run(args.seed)
    m = measure(cfg, sim, df, args.days, dt, args.scenario)
    m["gates"] = score(m, cfg, args)
    m["calibration_source"] = src
    m["seed"] = args.seed
    path = out / args.scenario / "metrics.json"
    path.write_text(json.dumps(m, indent=2, default=str))
    hard = [k for k, v in m["gates"].items() if not v["pass"]]
    for k, v in m["gates"].items():
        print("[gate:%s] %-22s %s" % (args.scenario, k,
                                      "PASS" if v["pass"] else "FAIL"))
    print("[gate:%s] wrote %s%s"
          % (args.scenario, path,
             "" if not hard else "  (FAIL: %s)" % ",".join(hard)))
    return 0


def _md_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |\n"


def mode_report(args) -> int:
    out = Path(args.out)
    metrics = {}
    for sc in SCENARIOS:
        p = out / sc / "metrics.json"
        if p.exists():
            metrics[sc] = json.loads(p.read_text())
    missing = [sc for sc in SCENARIOS if sc not in metrics]
    g1_ok = (out / "G1_PASS").exists()
    cal_ok = (out / "calibration_comprehensive.json").exists()
    if not metrics:
        print("[report] no metrics found under %s" % out)
        return 1

    nbars = {sc: m["nbar"] for sc, m in metrics.items()}
    cross_ok = True
    if len(nbars) >= 2:
        vals = list(nbars.values())
        cross_ok = (max(vals) / min(vals) - 1.0) <= 0.05

    lines = []
    lines.append("# Phase 1 Gate Report\n\n")
    lines.append("Generated: %s\n\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("- G1 (tests): driven by run_phase1_pilot.sh "
                 "(pytest gates the pipeline; see logs_phase1/g1_pytest.log)\n")
    lines.append("- G5 (retune readiness): s2 re-tuning APPROVED on the new "
                 "DGP. Launch via `CONFIRM=1 bash run_phase1_retune.sh` "
                 "after this report is PASS.\n\n")
    lines.append("## Preconditions\n\n")
    lines.append("- G1 marker (fidelity_out/G1_PASS): **%s**\n"
                 % ("present" if g1_ok else "MISSING"))
    lines.append("- calibration_comprehensive.json: **%s**\n"
                 % ("present" if cal_ok else "MISSING"))
    lines.append("- scenario metrics: %d/%d%s\n\n"
                 % (len(metrics), len(SCENARIOS),
                    "" if not missing else " — MISSING: " + ", ".join(missing)))

    lines.append("## G3 — mean occupancy (target %.0f, +/-5%%; drift <=3%%)\n\n"
                 % args.target_nbar)
    lines.append(_md_row(["scenario", "n-bar", "by day", "drift", "gate"]))
    lines.append(_md_row(["---"] * 5))
    for sc, m in metrics.items():
        gg = m["gates"]["G3_nbar"]
        lines.append(_md_row([
            sc, "%.2f" % m["nbar"],
            ", ".join("%.1f" % v for v in m["nbar_by_day"]),
            ("%.2f%%" % (100 * m["drift_rel"])) if m["drift_rel"] is not None
            else "n/a",
            "PASS" if gg["pass"] else "FAIL"]))
    lines.append("\nCross-scenario max/min - 1 <= 5%%: **%s**\n\n"
                 % ("PASS" if cross_ok else "FAIL"))

    lines.append("## G3 — arrival shape (peak +/-30 min; LUT P/V x//1.5)\n\n")
    lines.append(_md_row(["scenario", "peaks target (h)", "peaks realized (h)",
                          "LUT P/V (target)", "realized P/V note", "gate"]))
    lines.append(_md_row(["---"] * 6))
    for sc, m in metrics.items():
        s = m["gates"]["G3_shape"]
        lines.append(_md_row([
            sc,
            ", ".join("%.2f" % p for p in s["peaks_target"]),
            ", ".join("%.2f" % p for p in s["peaks_realized"]),
            "%.1f (%.1f)" % (s["lut_pv"], s["pv_target"]),
            s["pv_realized_note"],
            "PASS" if s["pass"] else "FAIL"]))

    lines.append("\n## G2 — behavioral fidelity\n\n")
    lines.append(_md_row(["scenario", "personas (low/med/high, meas vs adj)",
                          "order", "RACH coll.", "giveup",
                          "MT share (f_mt)", "MT-voice ratio", "gate"]))
    lines.append(_md_row(["---"] * 8))
    for sc, m in metrics.items():
        a = m["gates"]["G2a_persona_sessions"]
        b = m["gates"]["G2b_rach"]
        c = m["gates"]["G2c_paging"]
        ok = a["pass"] and b["pass"] and c["pass"]
        lines.append(_md_row([
            sc, "; ".join(a["detail"]),
            "ok" if a["order_ok"] else "BROKEN",
            "%.3f%%" % (100 * b["collision_rate"]), b["giveup"],
            "%.3f (%.3f)" % (c["mt_share"], c["f_mt_base"]),
            "%.2f" % c["mt_voice_ratio"] if np.isfinite(c["mt_voice_ratio"])
            else "n/a",
            "PASS" if ok else "FAIL"]))

    lines.append("\n## G4 — compute budget\n\n")
    any_m = next(iter(metrics.values()))
    g4 = any_m["gates"]["G4_compute"]
    lines.append("- measured: %.0f slots/s -> full 11-day run ~= %.1f min\n"
                 % (g4["slots_per_s"], g4["full_run_est_min"]))
    lines.append("- %d truth runs / %d workers ~= **%.1f wall days** (%s)\n"
                 % (g4["runs"], g4["workers"], g4["est_wall_days"], g4["note"]))

    lines.append("\n## T2 extended — services / paging / causes"
                 " (informational)\n\n")
    for sc, m in metrics.items():
        lines.append("### %s\n\n" % sc)
        ss = m.get("service_sessions", {})
        if ss:
            lines.append(_md_row(["service", "n", "wall mean (s)", "wall p50",
                                  "active mean (s)", "bursts mean",
                                  "censored"]))
            lines.append(_md_row(["---"] * 7))
            for s, d in sorted(ss.items()):
                lines.append(_md_row([
                    s, d["n"],
                    "%.0f" % d["wall_mean_s"] if d["wall_mean_s"] else "-",
                    "%.0f" % d["wall_p50_s"] if d["wall_p50_s"] else "-",
                    "%.0f" % d["active_mean_s"] if d["active_mean_s"] else "-",
                    "%.1f" % d["bursts_mean"] if d["bursts_mean"] else "-",
                    d["censored"]]))
        cb = m.get("cause_breakdown", {})
        lines.append("\ncauses: %s\n" % ", ".join(
            "%s=%d" % (k, v) for k, v in sorted(cb.items())))
        rr = m["rach"]
        lines.append("RACH chain/day: preamble %.0f : rar %.0f : msg3 %.0f :"
                     " setup %.0f (msg3/setup %.4f)\n"
                     % (rr["preamble_tx"] / m["days"], rr["rar"] / m["days"],
                        rr["msg3"] / m["days"], rr["success"] / m["days"],
                        rr["msg3_per_setup"]))
        pp = m["paging"]
        lines.append("paging: %.2f pages/UE/day; retry hist %s\n\n"
                     % (pp["pages_per_ue_day"], pp.get("retry_hist", {})))

    hard_fail = ((not cross_ok) or bool(missing)
                 or (not g1_ok) or (not cal_ok))
    for m in metrics.values():
        for k, v in m["gates"].items():
            if k != "G4_compute" and not v["pass"]:
                hard_fail = True
    lines.append("\n---\n\n**OVERALL: %s**\n"
                 % ("FAIL" if hard_fail else "PASS"))
    rp = out / "GATE_REPORT.md"
    rp.write_text("".join(lines))
    print("[report] wrote %s — OVERALL %s"
          % (rp, "FAIL" if hard_fail else "PASS"))
    return 1 if hard_fail else 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=("calibrate", "gate", "report"))
    ap.add_argument("--scenario", default="comprehensive", choices=SCENARIOS)
    ap.add_argument("--days", type=int, default=2,
                    help="recorded days for gate runs")
    ap.add_argument("--cal-days", type=int, default=1,
                    help="recorded days per calibration iteration")
    ap.add_argument("--warmup-days", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-nbar", type=float, default=320.0)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--arrival-scale-init", type=float, default=1.0)
    ap.add_argument("--access-scale-init", type=float, default=1.0)
    ap.add_argument("--calibrate-access", action="store_true",
                    help="also restore nominal persona targets via "
                         "LTE_ACCESS_SCALE (default: mechanism-consistency "
                         "gate only)")
    ap.add_argument("--workers", type=int, default=4,
                    help="planned parallel truth workers (G4 estimate)")
    ap.add_argument("--total-runs", type=int, default=220,
                    help="planned Phase-2 truth runs (G4 estimate)")
    ap.add_argument("--out", default="fidelity_out")
    args = ap.parse_args()

    if args.mode == "calibrate":
        return mode_calibrate(args)
    if args.mode == "gate":
        return mode_gate(args)
    return mode_report(args)


if __name__ == "__main__":
    sys.exit(main())
