#!/usr/bin/env python3
"""
make_robust5d_tables.py — assemble every robustness table of the 5-day controlled
protocol from the merged scenario directories written by
run_robustness5d_controlled_seeds.sh.

WHY A SECOND SCRIPT INSTEAD OF EXTENDING make_paper_tables.py
    make_paper_tables.py knows exactly three robustness axes (scale, service mix,
    S-TMSI) and its Table 2 cross-check against final_test_summary.json is the
    validated path for the headline. The 5-day plan has fifteen axes. Bolting
    twelve more onto that file would put the headline's protocol guard at risk for
    no benefit, so this script owns the robustness tables only and leaves Table 2
    to the existing, already-trusted code.

WHAT IT EMITS  (into --out, default paper_tables_<prefix>)
    robust5d_long.csv        one tidy row per (axis, scenario, model, mode) with
                             MAE mean/std, NMAE, n_present, Delta vs anchor,
                             relative degradation, and the paired A-B interval
    T5a_arrival.tex          R1   arrival temporal shape only
    T5b_site.tex             R2   joint site profile
    scale.tex                R3   network scale          (NMAE, not MAE)
    service_mix.tex          R4   service composition
    rrc_timer.tex            R5   RRC inactivity timer
    stmsi.tex                R6   S-TMSI reallocation    (paired)
    modeb_noise.tex          R7   the four Mode-B noise sweeps, one block each
    dwell.tex                R8+R9 dwell scale and dwell composition
    usage_resident_bg.tex    R10+R11+R12
    PROTOCOL.txt             the protocol fingerprint every table shares

FAIL-LOUD, in the same spirit as make_paper_tables.py
    * a required scenario that is missing stops the run — no silently short table
    * every scenario must share one protocol fingerprint (study prefix, comparison,
      train ratio, calibration variant, tuned-parameter hashes)
    * every scenario must share one seed set, or the Delta-vs-anchor columns and the
      paired intervals would be unpaired
    * S-TMSI: the cell branches must be EXACTLY equal on and off, per seed
    * Mode-B noise: the truth hashes must be identical to the anchor's, since
      observation noise may not move the physical trajectory
    * scipy is required for the Student-t interval; there is deliberately no normal
      fallback, which would change the published statistic without changing its name

    python make_robust5d_tables.py --prefix robust_s3c5_cal \\
        --params-dir tune_out_s3c5 --seeds "101 102 103 104 105"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
STEMS = ("Cell", "PerUE", "Hybrid")
FAMILIES = ("XGB", "LSTM")
MODES = ("A", "B")
PRETTY = {f"{s}_{f}": f"{s}-{f}" for s in STEMS for f in FAMILIES}

ANCHOR_AB = "main"
ANCHOR_B = "main_bonly"

# axis -> (title, anchor, [(scenario, label), ...], anchor_label or None)
# The anchor is listed explicitly as the first row of each table so a reader sees the
# baseline in the same table as the perturbation, rather than having to look it up.
AXES: dict[str, dict] = {
    "T5a_arrival": dict(
        title="Arrival temporal shape (site composition held at comprehensive)",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("main", "Comprehensive"), ("arr_resident", "Residential"),
              ("arr_office", "Office"), ("arr_transport", "Transport")]),
    "T5b_site": dict(
        title="Joint site profile (arrival, dwell, usage and service move together)",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("main", "Comprehensive"), ("site_resident", "Residential"),
              ("site_office", "Office"), ("site_transport", "Transport")]),
    "scale": dict(
        title="Network scale (NMAE: absolute MAE necessarily grows with cell size)",
        anchor=ANCHOR_AB, nmae=True,
        rows=[("scale_tiny", "Tiny"), ("scale_small", "Small"),
              ("scale_medium", "Medium"), ("main", "Large"), ("scale_xlarge", "XLarge")]),
    "service_mix": dict(
        title="Service composition",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("main", "Balanced"), ("mix_voice_heavy", "Voice-heavy"),
              ("mix_streaming_heavy", "Streaming-heavy"),
              ("mix_browsing_heavy", "Browsing-heavy")]),
    "rrc_timer": dict(
        title="RRC inactivity timer",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("timer_5", "5 s"), ("main", "10 s (baseline)"),
              ("timer_15", "15 s"), ("timer_30", "30 s")]),
    "dwell": dict(
        title="Dwell-time scale and dwell composition",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("dwell_scale_05", r"scale $\times$0.5"), ("main", "baseline"),
              ("dwell_scale_20", r"scale $\times$2.0"),
              ("dwellmix_transient_heavy", "transient-heavy"),
              ("dwellmix_stationary_heavy", "stationary-heavy"),
              ("dwellmix_homogeneous", "homogeneous")]),
    "usage_resident_bg": dict(
        title="Usage composition, resident share and background reconnect mix",
        anchor=ANCHOR_AB, nmae=False,
        rows=[("main", "baseline"),
              ("usagemix_low_heavy", "usage: low-heavy"),
              ("usagemix_high_heavy", "usage: high-heavy"),
              ("usagemix_homogeneous", "usage: homogeneous"),
              ("resident_15", "residents 15"), ("resident_60", "residents 60"),
              ("bg_chatty_heavy", "bg: chatty-heavy"),
              ("bg_quiet_heavy", "bg: quiet-heavy")]),
}

# Mode-B-only sweeps: four independent one-factor blocks sharing one anchor.
NOISE_BLOCKS = [
    ("DRX miss probability", [("noise_drx_000", "0.00"), (ANCHOR_B, "0.05 (baseline)"),
                              ("noise_drx_010", "0.10"), ("noise_drx_020", "0.20")]),
    ("Release false negative", [("noise_relfn_000", "0.00"), (ANCHOR_B, "0.05 (baseline)"),
                                ("noise_relfn_010", "0.10"), ("noise_relfn_020", "0.20")]),
    ("Release false positive", [("noise_relfp_0000", "0.000"), ("noise_relfp_0005", "0.005"),
                                ("noise_relfp_0010", "0.010"), (ANCHOR_B, "0.020 (baseline)")]),
    ("Release detection delay", [("noise_reldelay_00", "0.0 s"), (ANCHOR_B, "1.5 s (baseline)"),
                                 ("noise_reldelay_30", "3.0 s"), ("noise_reldelay_50", "5.0 s")]),
]

# Scalar protocol fields: compared verbatim across scenarios.
PROTOCOL_KEYS = ("study_prefix", "base_study_prefix", "comparison", "train_ratio",
                 "perue_calib", "calib_scope")
# Parameter fingerprints: compared only AFTER collapsing mode/branch -> branch. See
# _branch_fingerprint for why a verbatim comparison is wrong here.
FINGERPRINT_KEYS = ("parameter_hashes", "param_file_sha256")


def _branch_fingerprint(d: dict, where: str, field: str) -> dict:
    """Collapse a {mode}/{branch} fingerprint map to {branch}.

    training_tuned records these keyed f"{mode}/{branch}", so an A+B scenario carries six
    entries and a Mode-B-only scenario carries three. Comparing the raw dicts would call
    `main` and `main_bonly` a protocol mismatch even though, under `controlled`, both read
    exactly the same best_<prefix>_<branch>_AB.json. That is a false alarm on the very
    baseline the Mode-B noise sweep depends on.

    Collapsing to the branch is not merely a workaround: it is the stronger check. Under
    `controlled` the A and B entries for one branch MUST be identical, so a disagreement
    means the study was not actually controlled — a worse fault than any table mismatch,
    and one a verbatim dict comparison would never surface.
    """
    out: dict[str, str] = {}
    for key, value in (d or {}).items():
        mode, sep, branch = key.partition("/")
        if not sep:
            mode, branch = "", key
        if branch in out and out[branch] != value:
            raise SystemExit(
                f"{where}: {field} disagrees between modes for branch {branch!r} "
                f"({out[branch]!r} vs {value!r}).\n"
                "  Under comparison=controlled both modes read the same "
                "best_<prefix>_<branch>_AB.json, so these must be equal. Two different "
                "values mean Mode A and Mode B were given DIFFERENT hyperparameters, which "
                "invalidates the A/B comparison itself — not just this table.")
        out[branch] = value
    return out


# --------------------------------------------------------------------------------------
def _std(values) -> float:
    """SAMPLE standard deviation (ddof=1) — the convention runner.py, tune_runner and
    make_paper_tables all use. Mixing ddof would print two different '+/-' for one run."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _t_crit(df: int) -> float:
    """Student-t 0.975 quantile. No normal fallback: at n=5 the normal quantile is 29%
    too small (t_{.975,4} = 2.776 vs 1.96), so silently substituting it would narrow every
    published interval while leaving the label 'Student-t 95% CI' intact."""
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            "scipy is required for the paired Student-t intervals in these tables. "
            "Install scipy rather than reverting to a normal approximation."
        ) from exc
    return float(student_t.ppf(0.975, df=df))


def _load(prefix: str, name: str):
    d = Path(f"{prefix}_{name}")
    res, prov = d / "results.csv", d / "tuned_model_provenance.json"
    if not res.exists() or not prov.exists():
        return None
    return pd.read_csv(res), json.loads(prov.read_text()), d


def _models_present(df: pd.DataFrame) -> list[str]:
    """Which of the six models this run actually holds. With --families xgb the LSTM
    columns are absent by construction, and reporting them as '--' would be misread as a
    failed run rather than a deliberate restriction."""
    out = []
    for fam in FAMILIES:
        for stem in STEMS:
            m = f"{stem}_{fam}"
            if any(f"{m}_{mode}" in df.columns for mode in MODES):
                out.append(m)
    return out


def _cell(df: pd.DataFrame, model: str, mode: str, nmae: bool):
    col = f"{model}_{mode}" + ("_nmae" if nmae else "")
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna().values
    return (float(v.mean()), _std(v)) if len(v) else None


def _paired(df: pd.DataFrame, model: str):
    """Seed-paired A - B with a Student-t 95% interval, or None."""
    ca, cb = f"{model}_A", f"{model}_B"
    if ca not in df.columns or cb not in df.columns:
        return None
    j = df[["seed", ca, cb]].dropna()
    if len(j) < 2:
        return None
    d = pd.to_numeric(j[ca], errors="coerce").values - pd.to_numeric(j[cb], errors="coerce").values
    se = d.std(ddof=1) / np.sqrt(len(d))
    crit = _t_crit(len(d) - 1)
    return dict(mean=float(d.mean()), lo=float(d.mean() - crit * se),
                hi=float(d.mean() + crit * se), n=int(len(d)))


NA = "N/A"              # printed where a factor has no path to the estimator
_BSL = chr(92)          # backslash, kept out of string literals


def _fmt(ms, digits=2):
    return "--" if ms is None else f"{ms[0]:.{digits}f} $\\pm$ {ms[1]:.{digits}f}"


def _pct(cur, base) -> str:
    """Relative change, with negative zero normalised. '-0%' is a formatting artefact that
    reads as a real (tiny, negative) effect; at this precision it is indistinguishable
    from no change and must print as +0%."""
    if not cur or not base or base[0] <= 0:
        return "--"
    p = 100.0 * (cur[0] / base[0] - 1.0)
    if abs(p) < 0.5:
        p = 0.0
    return f"{p:+.0f}\\%"


# --------------------------------------------------------------------------------------
def collect(prefix: str, needed: list[str]):
    """Load every required scenario and verify they describe ONE experiment."""
    frames, provs, dirs, seeds, missing = {}, {}, {}, {}, []
    for name in sorted(set(needed)):
        got = _load(prefix, name)
        if got is None:
            missing.append(f"{prefix}_{name}/")
            continue
        frames[name], provs[name], dirs[name] = got
        seeds[name] = tuple(sorted(int(s) for s in frames[name]["seed"]))
    if missing:
        raise SystemExit(
            "Missing robustness scenarios — the tables would be silently incomplete:\n  "
            + "\n  ".join(missing)
            + "\n\nRun them:  SCENARIOS='<names>' CONFIRM=1 "
              "bash run_robustness5d_controlled_seeds.sh")

    # Reference = an A+B scenario, so its fingerprint covers every branch any scenario has.
    ref_name = ANCHOR_AB if ANCHOR_AB in frames else sorted(frames)[0]
    ref = {k: provs[ref_name].get(k) for k in PROTOCOL_KEYS}
    ref_fp = {k: _branch_fingerprint(provs[ref_name].get(k), ref_name, k)
              for k in FINGERPRINT_KEYS}
    for name in sorted(frames):
        cur = {k: provs[name].get(k) for k in PROTOCOL_KEYS}
        diff = {k: (ref[k], cur[k]) for k in PROTOCOL_KEYS if ref[k] != cur[k]}
        for k in FINGERPRINT_KEYS:
            fp = _branch_fingerprint(provs[name].get(k), name, k)
            # A Mode-B-only scenario legitimately holds a SUBSET of the branches, so the
            # test is "agrees wherever both define it, and adds nothing new", not equality.
            extra = sorted(set(fp) - set(ref_fp[k]))
            if extra:
                diff[k] = (f"branches {sorted(ref_fp[k])}", f"unexpected branches {extra}")
            clash = sorted(b for b in fp if b in ref_fp[k] and fp[b] != ref_fp[k][b])
            if clash:
                diff[k] = ({b: ref_fp[k][b] for b in clash}, {b: fp[b] for b in clash})
            if not set(fp) & set(ref_fp[k]):
                diff[k] = (sorted(ref_fp[k]), "no shared branch — vacuous check")
        if diff:
            lines = [f"    {k}: {ref_name}={a!r}  vs  {name}={b!r}" for k, (a, b) in diff.items()]
            raise SystemExit("Protocol mismatch between robustness scenarios:\n"
                             + "\n".join(lines)
                             + "\n\nEvery cell of a robustness table must come from one "
                               "frozen estimator definition. Re-run the offending scenario.")
    uniq = set(seeds.values())
    if len(uniq) > 1:
        lines = [f"    {n:28} seeds={list(s)}" for n, s in sorted(seeds.items())]
        raise SystemExit("Scenarios do not share one seed set, so Delta-vs-anchor and the "
                         "paired intervals would be UNPAIRED:\n" + "\n".join(lines))
    return frames, dirs, {**ref, **ref_fp, "seeds": list(next(iter(uniq)))}


def check_stmsi(frames: dict):
    """Reallocation only fragments S-TMSI identities: it runs on an isolated RNG stream and
    the cell branches never read identity, so with matched seeds the cell MAEs must be
    EXACTLY equal — not merely close. Any difference means the pairing broke or an identity
    signal leaked into the cell features, and the S-TMSI table would be wrong either way."""
    off, on = frames.get(ANCHOR_AB), frames.get("stmsi_on")
    if off is None or on is None:
        return
    a, b = off.set_index("seed"), on.set_index("seed")
    for model in [m for m in _models_present(off) if m.startswith("Cell_")]:
        for mode in MODES:
            col = f"{model}_{mode}"
            if col not in a.columns or col not in b.columns:
                continue
            j = a[[col]].join(b[[col]], lsuffix="_off", rsuffix="_on", how="inner")
            d = (pd.to_numeric(j[f"{col}_on"], errors="coerce")
                 - pd.to_numeric(j[f"{col}_off"], errors="coerce")).abs()
            bad = j[d > 1e-6]
            if len(bad):
                rows = "\n".join(f"      seed {int(i)}: {r[f'{col}_off']:.6f} -> "
                                 f"{r[f'{col}_on']:.6f}" for i, r in bad.iterrows())
                raise SystemExit(f"{col} changed under S-TMSI reallocation:\n{rows}\n"
                                 "  This must be exactly 0. Check the seeds/config match and "
                                 "that no identity-derived signal reached the cell features.")


def check_noise_truth(dirs: dict, seeds: list[int]):
    """Mode-B observation noise is injected at OBSERVATION time; it must never move the
    physical trajectory. Comparing truth hashes against the Mode-B anchor is what proves a
    noise sweep measured observability and not a different simulation.

    A MISSING hash file is a failure, not a skip. Silently passing when the evidence is
    absent is the precise opposite of the guarantee this function is here to give: the
    suite would look verified while nothing had been compared.

    All three trajectory hashes are checked. `truth_dataframe` matters as much as the event
    streams — a change there means the recorded cell occupancy itself moved, which is what
    the estimator is being scored against.
    """
    base = dirs.get(ANCHOR_B)
    noisy = sorted(n for n in dirs if n.startswith("noise_"))
    if base is None:
        if noisy:
            raise SystemExit(
                f"Mode-B noise scenarios are present ({len(noisy)}) but the Mode-B anchor "
                f"'{ANCHOR_B}' is missing, so their truth trajectories cannot be verified "
                "and their relative degradations have no baseline. Run it.")
        return
    keys = ("truth_dataframe", "physical_ue_events", "ue_meta")
    bad, missing = [], []
    for seed in seeds:
        ref_p = base / f"truth_hashes_seed{seed}.json"
        if not ref_p.exists():
            missing.append(str(ref_p))
            continue
        ref = json.loads(ref_p.read_text())
        for name in noisy:
            q = dirs[name] / f"truth_hashes_seed{seed}.json"
            if not q.exists():
                missing.append(str(q))
                continue
            cur = json.loads(q.read_text())
            for k in keys:
                if ref.get(k) is None or cur.get(k) is None:
                    missing.append(f"{q}: key {k!r} absent")
                elif ref[k] != cur[k]:
                    bad.append(f"    seed {seed} {name}: {k} {ref[k]} != {cur[k]}")
    if missing:
        raise SystemExit(
            "Truth hashes missing, so the Mode-B noise sweep is UNVERIFIED:\n  "
            + "\n  ".join(missing[:20])
            + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
            + "\n\nEvery scenario writes truth_hashes_seed<seed>.json. Their absence means "
              "the run did not complete or the merge dropped them; either way the "
              "invariance claim cannot be made.")
    if bad:
        raise SystemExit("Mode-B noise changed the TRUTH trajectory:\n" + "\n".join(bad)
                         + "\n  Observation noise must not touch the simulator. The noise "
                           "sweep would be measuring a different cell, not a worse sniffer.")


def check_seed_set(protocol: dict, requested: list[int]):
    """--seeds is used to look up per-seed artifacts, so a mismatch against the seeds the
    runs actually contain would verify one set while tabulating another."""
    actual = sorted(int(x) for x in protocol["seeds"])
    if sorted(requested) != actual:
        raise SystemExit(
            f"--seeds {sorted(requested)} does not match the seeds in the scenario "
            f"results: {actual}.\n  Pass the seeds that were actually run, or the "
            "truth-hash and pairing checks would cover a different set than the tables.")


# --------------------------------------------------------------------------------------
# Table 2 — the headline, generated HERE rather than by make_paper_tables.py
#
# The existing make_paper_tables.py cannot produce it under this protocol, for two
# independent reasons, and neither is worth "fixing" in a file whose guards are already
# trusted:
#   1. its main() hard-requires the OLD scenario names (_default, _tiny, _voice_heavy,
#      _stmsi, ...). The 5-day runner writes _main, _scale_tiny, _mix_voice_heavy,
#      _stmsi_on, so it exits on "Missing required paper scenarios" before reaching
#      Table 2 at all;
#   2. its table2_main() demands that the headline and the robustness runs cover the SAME
#      (mode, branch) set, and treats any key present on one side only as a hard error.
#      With an XGB-only robustness suite the Main LSTM entries are exactly such keys, so
#      even with matching directory names it would refuse.
# make_paper_tables.py is therefore left completely untouched, as the record of the
# archived earlier per-mode-tuned protocol.
# --------------------------------------------------------------------------------------
def table2_main(params_dir: Path, protocol: dict, robust_dir: Path) -> tuple[str, list[str]]:
    """Headline Mode A/B table from final_test_summary.json, cross-checked against the
    robustness protocol. Returns (latex, models_reported)."""
    src = params_dir / "final_test_summary.json"
    if not src.exists():
        raise SystemExit(
            f"Table 2 needs {src}.\n  Run stage 3 of run_paper5d_controlled.sh "
            "(tune_runner --final ... then --merge-shards).")
    blob = json.loads(src.read_text())
    results, meta = blob.get("results", blob), blob.get("meta")
    if meta is None:
        raise SystemExit(f"{src} predates variant tracking (no 'meta'). It cannot be shown "
                         "to describe the same estimator as the robustness suite; re-run "
                         "--final under the 5-day controlled protocol.")

    # ---- scalar protocol fields must agree exactly
    problems = []
    for key in ("study_prefix", "base_study_prefix", "comparison",
                "perue_calib", "calib_scope"):
        got, want = meta.get(key), protocol.get(key)
        if got is not None and want is not None and got != want:
            problems.append(f"    {key}: headline={got!r}  robustness={want!r}")
    if meta.get("train_ratio") is not None and protocol.get("train_ratio") is not None:
        if abs(float(meta["train_ratio"]) - float(protocol["train_ratio"])) > 1e-9:
            problems.append(f"    train_ratio: headline={meta['train_ratio']!r}  "
                            f"robustness={protocol['train_ratio']!r}")

    # ---- horizon: the plan pins Main and robustness to one 5-day trace
    rc = robust_dir / "resolved_config.json"
    if rc.exists() and meta.get("horizon") is not None:
        try:
            rob_h = json.loads(rc.read_text())["simulation"]["time"]["total_slots"]
        except (KeyError, json.JSONDecodeError):
            rob_h = None
        if rob_h is not None and int(meta["horizon"]) != int(rob_h):
            problems.append(f"    horizon: headline={meta['horizon']}  robustness={rob_h}")
    if meta.get("horizon") is None:
        problems.append("    horizon: headline records null — the merge step was not given "
                        "--horizon, so its provenance is unusable")

    # ---- parameter SHAs: INTERSECTION only.
    # Main reports six branches, robustness three. The LSTM SHAs existing only on the
    # headline side is correct and expected; what must hold is that every branch the two
    # sides share was loaded from the same file, and that robustness introduces no branch
    # the headline lacks.
    h_fp = _branch_fingerprint(meta.get("param_file_sha256"), "headline", "param_file_sha256")
    r_fp = protocol.get("param_file_sha256") or {}
    shared = sorted(set(h_fp) & set(r_fp))
    if h_fp and r_fp:
        if not shared:
            problems.append(f"    param_file_sha256: no shared branch "
                            f"(headline {sorted(h_fp)} vs robustness {sorted(r_fp)}) — "
                            f"the check would be vacuous")
        only_r = sorted(set(r_fp) - set(h_fp))
        if only_r:
            problems.append(f"    param_file_sha256: robustness uses branches the headline "
                            f"never reports: {only_r}")
        clash = [b for b in shared if h_fp[b] != r_fp[b]]
        if clash:
            problems.append(f"    param_file_sha256: differing files for {clash}")
    else:
        print("  (info) parameter-file hashes unavailable on one side; Table 2 protocol "
              "verified on prefixes/variant only.")

    if problems:
        raise SystemExit("Protocol mismatch between Table 2 and the robustness tables:\n"
                         + "\n".join(problems)
                         + "\n\nBoth must come from one tuning study, one split, one "
                           "calibration variant and one horizon.")

    models = [m for m in (f"{s}_{f}" for f in FAMILIES for s in STEMS) if m in results]
    if not models:
        raise SystemExit(f"{src} holds no model results.")
    lines = [r"% Table 2 -- main held-out performance (all tuned models)",
             f"% headline branches: {sorted(h_fp) or 'n/a'}",
             f"% robustness branches: {sorted(r_fp) or 'n/a'}  (shared: {shared})",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & Mode A (MAE) & Mode B (MAE) & $A-B$ [95\% CI] \\",
             r"\midrule"]
    for m in models:
        a, b = results[m].get("A"), results[m].get("B")
        pa = results[m].get("paired_A_minus_B")
        ci = (f"{pa['mean']:+.2f} [{pa['ci95'][0]:+.2f}, {pa['ci95'][1]:+.2f}]"
              if pa else "--")
        lines.append(f"{PRETTY[m]} & {_fmt((a['mean'], a['std'])) if a else '--'} & "
                     f"{_fmt((b['mean'], b['std'])) if b else '--'} & {ci} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    n_seeds = len(meta.get("test_seeds") or [])
    print(f"  Table 2: {len(models)} models, {n_seeds} test seeds "
          f"(seeds {meta.get('test_seeds')})")
    return "\n".join(lines) + "\n", models


# --------------------------------------------------------------------------------------
def long_rows(frames: dict, axis_name: str, spec: dict, models: list[str]) -> list[dict]:
    anchor = frames[spec["anchor"]]
    nmae = spec["nmae"]
    out = []
    for name, label in spec["rows"]:
        df = frames[name]
        col = "mean_present_test" if "mean_present_test" in df.columns else "mean_present"
        npres = float(pd.to_numeric(df[col], errors="coerce").mean())
        pr = {m: _paired(df, m) for m in models}
        for m in models:
            for mode in MODES:
                cur, base = _cell(df, m, mode, nmae), _cell(anchor, m, mode, nmae)
                if cur is None:
                    continue
                d_abs = cur[0] - base[0] if base else np.nan
                d_rel = 100.0 * (cur[0] / base[0] - 1.0) if base and base[0] > 0 else np.nan
                p = pr[m] if mode == "A" else None
                out.append(dict(
                    axis=axis_name, scenario=name, label=label, model=m, mode=mode,
                    n_seeds=int(len(df)), mean_present=round(npres, 2),
                    metric="NMAE" if nmae else "MAE",
                    value_mean=round(cur[0], 6), value_std=round(cur[1], 6),
                    delta_vs_anchor=round(d_abs, 6) if np.isfinite(d_abs) else "",
                    rel_degradation_pct=round(d_rel, 2) if np.isfinite(d_rel) else "",
                    # n is in the column name on purpose: these are 5-seed subset values,
                    # not the 10-seed headline in Table 2.
                    **({f"paired_A_minus_B_n{p['n']}": round(p["mean"], 6),
                        f"ci95_lo_n{p['n']}": round(p["lo"], 6),
                        f"ci95_hi_n{p['n']}": round(p["hi"], 6)} if p else {})))
    return out


def axis_table(frames: dict, spec: dict, models: list[str]) -> str:
    """One row per (condition, mode); one column per model; Mode A rows carry the paired
    A - B interval so the reader never has to pair two rows by eye."""
    nmae, dg = spec["nmae"], 3 if spec["nmae"] else 2
    anchor = frames[spec["anchor"]]
    head = " & ".join(PRETTY[m] for m in models)
    lines = [r"% " + spec["title"],
             r"\begin{tabular}{ll r " + "c" * len(models) + "}", r"\toprule",
             f"Condition & Mode & $\\bar n$ & {head} \\\\", r"\midrule"]
    for name, label in spec["rows"]:
        df = frames[name]
        col = "mean_present_test" if "mean_present_test" in df.columns else "mean_present"
        npres = float(pd.to_numeric(df[col], errors="coerce").mean())
        for i, mode in enumerate(MODES):
            cells = [_fmt(_cell(df, m, mode, nmae), dg) for m in models]
            lines.append(f"{label if i == 0 else ''} & {mode} & "
                         f"{f'{npres:.1f}' if i == 0 else ''} & " + " & ".join(cells) + r" \\")
        # Delta vs anchor, Mode B only (the constrained regime is where degradation matters)
        if name != spec["anchor"]:
            ds = [_pct(_cell(df, m, "B", nmae), _cell(anchor, m, "B", nmae)) for m in models]
            lines.append(r" & $\Delta$B & & " + " & ".join(ds) + r" \\")
        lines.append(r"\addlinespace")
    # NO paired A-B interval here, deliberately.
    #
    # This suite runs five seeds; the headline runs ten, and the five are a SUBSET of the
    # ten. A Student-t interval on the subset is therefore not independent evidence, and
    # at n=5 it is 1.74x wider per unit of sd yet can still land on the opposite verdict
    # by sampling: on this data the anchor gives Hybrid-XGB -0.084 [-0.152, -0.017],
    # which excludes zero, while the ten-seed headline gives -0.044 [-0.122, +0.034],
    # which does not. Printing both would put "Mode A is significantly better" in seven
    # robustness tables and "parity" in Table 2, for the same quantity, in one paper.
    #
    # The robustness tables exist to report DEGRADATION RATIOS across conditions. The
    # A-B comparison has exactly one authoritative source, and it is Table 2.
    lines += [r"\midrule",
              r"\multicolumn{" + str(3 + len(models)) + r"}{p{0.8\linewidth}}{\footnotesize "
              r"Cells report absolute error per condition and, for Mode B, degradation "
              r"relative to the anchor. The authoritative Mode A vs Mode B comparison is "
              r"Table~\ref{tab:main} ($n=10$ held-out seeds); the five seeds used here are "
              r"a subset of those ten and are reported for degradation ratios only.} " + _BSL + _BSL]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _block_invariant(frames: dict, block, model: str) -> bool:
    """True if `model` is bit-identical to the Mode-B anchor at EVERY level of this noise
    block, for every seed.

    Such a cell must not be printed as a 0% degradation. In this simulator the per-track
    observation path is delay-only by design: release false negatives and false positives
    are injected into the aggregate counters, not into per-UE session boundaries, because
    a tracker with identity continuity resolves a missed release at the next RRC setup
    (see the module docstring of observation/mode_b.py). So a per-UE row here is not
    evidence of robustness to release noise — the factor never reaches that estimator's
    inputs at all. Printing "+0%" invites exactly the opposite reading.

    The test is exact equality per seed rather than a tolerance: an estimator that merely
    happened to score similarly would differ in the last digits, and should be reported as
    a real (small) effect.
    """
    a = frames[ANCHOR_B].set_index("seed")
    col = f"{model}_B"
    if col not in a.columns:
        return False
    for name, _label in block:
        b = frames[name].set_index("seed")
        if col not in b.columns:
            return False
        j = a[[col]].join(b[[col]], lsuffix="_a", rsuffix="_b", how="inner")
        if len(j) == 0 or not (j[f"{col}_a"] == j[f"{col}_b"]).all():
            return False
    return True


def noise_table(frames: dict, models: list[str]) -> str:
    """Mode B only, four independent one-factor blocks. Absolute MAE plus relative
    degradation against the Mode-B anchor, which the plan asks for explicitly."""
    head = " & ".join(PRETTY[m] for m in models)
    lines = [r"% Mode-B observation-noise sensitivity (Mode B only, one factor at a time)",
             r"\begin{tabular}{ll " + "c" * len(models) + "}", r"\toprule",
             f"Factor & Level & {head} \\\\", r"\midrule"]
    anchor = frames[ANCHOR_B]
    footnoted = False
    for title, rows in NOISE_BLOCKS:
        inert = {m: _block_invariant(frames, rows, m) for m in models}
        footnoted = footnoted or any(inert.values())
        for i, (name, label) in enumerate(rows):
            df = frames[name]
            cells = [(NA if inert[m] else _fmt(_cell(df, m, "B", False))) for m in models]
            lines.append(f"{title if i == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
            if name != ANCHOR_B:
                ds = [(NA if inert[m] else _pct(_cell(df, m, "B", False),
                                                _cell(anchor, m, "B", False)))
                      for m in models]
                lines.append(r" & \quad rel. & " + " & ".join(ds) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule"]
    if footnoted:
        lines += [r"\multicolumn{" + str(2 + len(models)) + r"}{p{0.86\linewidth}}{"
                  r"\footnotesize N/A: this factor is injected into the aggregate "
                  r"sniffer counters only. Per-track session boundaries carry detection "
                  r"lag but no release false negative or false positive, because a "
                  r"tracker with identity continuity resolves a missed release at the "
                  r"next RRC setup. These cells therefore report the absence of an input "
                  r"path, not robustness to release noise.} " + _BSL + _BSL]
    lines += [r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _load_bias(path: Path):
    """Mean signed error per (model, condition) from the figure-data export, or None.

    Kept out of the robustness pipeline proper because it is computed from the per-seed
    prediction arrays rather than from results.csv. Reading it here means the S-TMSI table
    and the S-TMSI figure quote one number with one dispersion, instead of a hand-copied
    mean in the table and a separately computed SD in the figure.
    """
    if not path.exists():
        return None
    b = pd.read_csv(path)
    need = {"model", "condition", "bias"}
    if not need <= set(b.columns):
        raise SystemExit(f"{path} lacks {sorted(need - set(b.columns))}; regenerate it with "
                         "collect_figure_data.sh")
    b["m"] = b.model.str.replace("_B$", "", regex=True)
    has_sd = "bias_std" in b.columns
    out = {}
    for _i, r in b.iterrows():
        out[(r["m"], r["condition"])] = (float(r["bias"]),
                                         float(r["bias_std"]) if has_sd else None)
    return out


def stmsi_table(frames: dict, models: list[str], bias=None) -> str:
    off, on = frames[ANCHOR_AB], frames["stmsi_on"]
    sg = bool(bias)
    lines = [r"% S-TMSI reallocation (paired; cell branches must be identical)."]
    if sg:
        lines.append(r"% Signed error and its SD are the same five-seed test-slice values "
                     r"the S-TMSI figure plots.")
    lines += [r"\begin{tabular}{lcccccc" + ("c" if sg else "") + "}", r"\toprule",
              r"Model & \multicolumn{2}{c}{Mode A MAE (UEs)} "
              r"& \multicolumn{2}{c}{Mode B MAE (UEs)} "
              r"& \multicolumn{2}{c}{Mode B degradation}"
              + (r" & Mode B signed error (UEs)" if sg else "") + r" \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
              + (r"\cmidrule(lr){8-8}" if sg else ""),
              r" & no realloc. & realloc. & no realloc. & realloc. & abs. & rel."
              + (r" & no realloc. $\rightarrow$ realloc." if sg else "") + r" \\",
              r"\midrule"]
    for m in models:
        a0, a1 = _cell(off, m, "A", False), _cell(on, m, "A", False)
        b0, b1 = _cell(off, m, "B", False), _cell(on, m, "B", False)
        d_abs = f"{b1[0] - b0[0]:+.2f}" if b0 and b1 else "--"
        d_rel = _pct(b1, b0)
        row = (f"{PRETTY[m]} & {_fmt(a0)} & {_fmt(a1)} & {_fmt(b0)} & {_fmt(b1)} "
               f"& {d_abs} & {d_rel}")
        if sg:
            # Already inside $...$, so \pm must NOT be wrapped in its own $ $ the way
            # _fmt does for text-mode cells; nesting math delimiters breaks the row.
            def f1(cond):
                v = bias.get((m, cond))
                if v is None:
                    return "--"
                return (f"{v[0]:+.2f}" if v[1] is None
                        else f"{v[0]:+.2f} " + _BSL + f"pm {v[1]:.2f}")
            row += f" & ${f1('off')} " + _BSL + r"rightarrow " + f"{f1('on')}$"
        lines.append(row + " " + _BSL + _BSL)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------

def _check_latex_rows(out: Path) -> int:
    """Every emitted table row must end with exactly two backslashes.

    This exists because a row that ends with ONE backslash is not a LaTeX error you notice:
    the table still compiles, it just silently merges two rows. The generator builds rows
    from a mix of raw and f-strings, and an f-string needs four source backslashes to emit
    two -- an easy off-by-one that no unit test of the numbers would ever catch. Checking
    the rendered output closes that gap at the only place it can be closed.
    """
    STRUCT = tuple(_BSL + w for w in ("begin", "end", "toprule", "midrule", "bottomrule",
                                     "addlinespace", "cmidrule"))
    bad, checked = [], 0
    for f in sorted(out.glob("*.tex")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            t = line.rstrip()
            if not t or t.startswith("%") or any(t.startswith(x) for x in STRUCT):
                continue
            if "&" not in t and (_BSL + "multicolumn") not in t:
                continue
            checked += 1
            n = len(t) - len(t.rstrip(_BSL))
            if n != 2:
                bad.append(f"    {f.name}:{i} ends in {n} backslash(es): ...{t[-40:]!r}")
    if bad:
        raise SystemExit(
            f"Malformed LaTeX rows ({len(bad)}); the tables would render with merged or "
            "broken rows:\n" + "\n".join(bad[:15])
            + (f"\n    ... and {len(bad) - 15} more" if len(bad) > 15 else ""))
    return checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="robust_s3c5_cal",
                    help="prefix of the merged scenario dirs (<prefix>_main, ...)")
    ap.add_argument("--params-dir", default="tune_out_s3c5")
    ap.add_argument("--seeds", default="101 102 103 104 105")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stmsi-bias", default="figure_data/stmsi_bias.csv",
                    help="signed-error CSV from collect_figure_data.sh. When present, the "
                         "S-TMSI table gains the mean signed error and its SD, matching "
                         "the S-TMSI figure exactly.")
    ap.add_argument("--skip-table2", action="store_true",
                    help="build the robustness tables only, without Table 2. Use while the "
                         "Main evaluation is still outstanding. The cross-check that the "
                         "headline and the robustness suite share one protocol is SKIPPED, "
                         "so the output is for inspection, not for the paper.")
    args = ap.parse_args()
    out = Path(args.out or f"paper_tables_{args.prefix.replace('robust_', '')}")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split()]

    needed = [ANCHOR_AB, ANCHOR_B, "stmsi_on"]
    for spec in AXES.values():
        needed += [n for n, _ in spec["rows"]]
    for _t, rows in NOISE_BLOCKS:
        needed += [n for n, _ in rows]

    frames, dirs, protocol = collect(args.prefix, needed)
    check_seed_set(protocol, seeds)
    check_stmsi(frames)
    check_noise_truth(dirs, seeds)

    ab_models = _models_present(frames[ANCHOR_AB])
    b_models = _models_present(frames[ANCHOR_B])
    if not ab_models:
        raise SystemExit(f"{args.prefix}_{ANCHOR_AB}/results.csv holds no model columns.")

    # Table 2 first: if the headline and the robustness suite are not the same experiment,
    # nothing downstream is worth generating.
    if args.skip_table2:
        main_models = []
        print("  Table 2: SKIPPED (--skip-table2). The protocol cross-check against the "
              "headline did NOT run — do not put these tables in the paper until a full "
              "build succeeds.")
    else:
        t2, main_models = table2_main(Path(args.params_dir), protocol, dirs[ANCHOR_AB])
        (out / "table2_main.tex").write_text(t2)

    rows: list[dict] = []
    for axis_name, spec in AXES.items():
        (out / f"{axis_name}.tex").write_text(axis_table(frames, spec, ab_models))
        rows += long_rows(frames, axis_name, spec, ab_models)

    bias = _load_bias(Path(args.stmsi_bias))
    if bias is None:
        print(f"  (info) {args.stmsi_bias} not found — S-TMSI table omits the signed-error "
              "column; the figure will still plot it from its own copy.")
    (out / "stmsi.tex").write_text(stmsi_table(frames, ab_models, bias))
    rows += long_rows(frames, "stmsi",
                      dict(anchor=ANCHOR_AB, nmae=False,
                           rows=[("main", "no realloc."), ("stmsi_on", "realloc.")]),
                      ab_models)

    (out / "modeb_noise.tex").write_text(noise_table(frames, b_models))
    for title, blk in NOISE_BLOCKS:
        rows += long_rows(frames, f"noise:{title}",
                          dict(anchor=ANCHOR_B, nmae=False, rows=blk), b_models)

    df_long = pd.DataFrame(rows).drop_duplicates(
        subset=["axis", "scenario", "model", "mode"], keep="first")
    df_long.to_csv(out / "robust5d_long.csv", index=False)

    (out / "PROTOCOL.txt").write_text(
        "\n".join(f"{k}: {protocol.get(k)}"
                  for k in (*PROTOCOL_KEYS, *FINGERPRINT_KEYS, "seeds"))
        + f"\nmodels_table2: {main_models or 'SKIPPED — provisional output'}\n"
        + f"models_robust_ab: {ab_models}\n"
        + f"models_robust_b: {b_models}\n"
        + f"n_scenarios: {len(frames)}\nn_seeds: {len(seeds)}\n")

    n_rows = _check_latex_rows(out)
    print(f"wrote {len(list(out.glob('*.tex')))} LaTeX tables ({n_rows} content rows) "
          f"+ robust5d_long.csv -> {out}/")
    print(f"  Table 2 models:  {main_models or 'skipped'}")
    print(f"  robustness:      {ab_models}")
    print(f"  seeds:           {protocol['seeds']}")
    if main_models and len(ab_models) < len(main_models):
        print(f"  NOTE: Table 2 reports {len(main_models)} models, robustness "
              f"{len(ab_models)}. That asymmetry is the --families restriction and MUST be "
              "stated in the paper, not left implicit.")
    if len(seeds) < 10:
        crit = _t_crit(len(seeds) - 1)
        print(f"  NOTE: robustness n={len(seeds)} -> t_.975 = {crit:.3f}; its paired "
              f"intervals are {crit / np.sqrt(len(seeds)) / (2.262 / np.sqrt(10)):.2f}x "
              "wider than the 10-seed headline. Read them as degradation ratios, not as "
              "significance tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
