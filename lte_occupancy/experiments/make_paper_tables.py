"""
experiments/make_paper_tables.py — assemble the paper's result tables from tuned artifacts.

Table layout (one experimental axis per table, so each maps to one research question):

    Table 1  Experimental setup / protocol        (written by hand; not generated here)
    Table 2  Main Mode A/B performance            <- tune_out_<prefix>/final_test_summary.json
    Table 3  Network-scale robustness             <- robust_<prefix>_{tiny,small,medium,default,xlarge}
    Table 4  Traffic-composition robustness       <- robust_<prefix>_{default,voice_heavy,streaming_heavy,browsing_heavy}
    Table 5  S-TMSI robustness (paired)           <- robust_<prefix>_{default,stmsi}

`default` (large + balanced + no realloc) is the shared anchor: it is the "Large" column of
Table 3, the "Balanced" column of Table 4, and the "no realloc" side of Table 5, so all
comparisons use the same runner, split and seeds.

Reported figures:
  * MAE  = mean +/- SAMPLE std (ddof=1) over the run's seeds (test slice = last 20%)
  * NMAE = MAE / mean(n_present) on the SAME test slice — required for Table 3, where the
    absolute MAE necessarily grows with cell size.
Table 2 additionally carries the paired Mode A - Mode B difference with a 95% interval.

Fail-loud by construction (a paper table must not quietly mix things):
  * every required scenario must exist, else exit — no silently skipped rows
  * all scenarios must share one protocol (study prefix, comparison, train ratio,
    calibration variant, and the tuned parameter hashes), else exit
  * all scenarios must share one seed set, else exit — Table 5 is only meaningful paired
  * Table 2's headline variant must match the scenarios' variant, else exit
Legacy (Survival-based) numbers are never read here.

    python -m lte_occupancy.experiments.make_paper_tables \
        --robust-prefix robust_s3c5 --params-dir tune_out_s3c5 --out paper_tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Cell_XGB", "PerUE_XGB", "Hybrid_XGB", "Cell_LSTM", "PerUE_LSTM", "Hybrid_LSTM"]
PRETTY = {"Cell_XGB": "Cell-XGB", "PerUE_XGB": "PerUE-XGB", "Hybrid_XGB": "Hybrid-XGB",
          "Cell_LSTM": "Cell-LSTM", "PerUE_LSTM": "PerUE-LSTM", "Hybrid_LSTM": "Hybrid-LSTM"}
MODES = ("A", "B")
SCALES = [("tiny", "Tiny"), ("small", "Small"), ("medium", "Medium"),
          ("default", "Large"), ("xlarge", "XLarge")]
MIXES = [("default", "Balanced"), ("voice_heavy", "Voice-heavy"),
         ("streaming_heavy", "Streaming-heavy"), ("browsing_heavy", "Browsing-heavy")]


# --------------------------------------------------------------------------------------
def _load_scenario(prefix: str, name: str) -> pd.DataFrame | None:
    p = Path(f"{prefix}_{name}") / "results.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_provenance(prefix: str, name: str) -> dict | None:
    p = Path(f"{prefix}_{name}") / "tuned_model_provenance.json"
    return json.loads(p.read_text()) if p.exists() else None


def _std(values) -> float:
    """SAMPLE standard deviation (ddof=1) — same convention as runner.py's summary.csv
    (pandas default) and tune_runner._std. Mixing ddof would print two different '+/-'
    values for the same experiment."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _cells(df: pd.DataFrame, model: str, mode: str, nmae: bool):
    """(mean, std) of MAE (or NMAE) for one model/mode, over the run's seeds."""
    col = f"{model}_{mode}" + ("_nmae" if nmae else "")
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna().values
    return (float(v.mean()), _std(v)) if len(v) else None


def _fmt(ms, digits=2):
    if ms is None:
        return "--"
    return f"{ms[0]:.{digits}f} $\\pm$ {ms[1]:.{digits}f}"


# --------------------------------------------------------------------------------------
# fail-loud protocol checks: a paper table must never mix estimators or seed sets
# --------------------------------------------------------------------------------------
# Fields that define "the same estimator under a different scenario". parameter_hashes
# pins the actual tuned params, so a scenario accidentally run against another tuning
# directory is caught even if the prefix string matches.
PROTOCOL_KEYS = ("study_prefix", "base_study_prefix", "comparison", "train_ratio",
                 "perue_calib", "calib_scope", "parameter_hashes", "param_file_sha256")


def _check_protocol(prefix: str, needed: list[str]):
    """Every required scenario must exist and share one protocol and one seed set.
    Returns (frames, protocol) or exits."""
    frames, provs, seeds = {}, {}, {}
    missing = []
    for name in needed:
        df = _load_scenario(prefix, name)
        if df is None:
            missing.append(f"{prefix}_{name}/results.csv")
            continue
        pv = _load_provenance(prefix, name)
        if pv is None:
            missing.append(f"{prefix}_{name}/tuned_model_provenance.json")
            continue
        frames[name], provs[name] = df, pv
        seeds[name] = tuple(sorted(int(s) for s in df["seed"]))
    if missing:
        raise SystemExit(
            "Missing required paper scenarios (tables would be silently incomplete):\n  "
            + "\n  ".join(missing)
            + "\n\nRun them first:  bash run_robustness_tuned.sh")

    ref_name = needed[0]
    ref = {k: provs[ref_name].get(k) for k in PROTOCOL_KEYS}
    for name in needed[1:]:
        cur = {k: provs[name].get(k) for k in PROTOCOL_KEYS}
        diff = {k: (ref[k], cur[k]) for k in PROTOCOL_KEYS if ref[k] != cur[k]}
        if diff:
            lines = [f"    {k}: {ref_name}={a!r}  vs  {name}={b!r}" for k, (a, b) in diff.items()]
            raise SystemExit(
                f"Protocol mismatch between scenarios '{ref_name}' and '{name}':\n"
                + "\n".join(lines)
                + "\n\nAll cells of a table must come from ONE estimator definition. "
                  "Re-run the offending scenario with matching settings.")
    uniq_seeds = set(seeds.values())
    if len(uniq_seeds) > 1:
        lines = [f"    {n:16} seeds={list(s)}" for n, s in seeds.items()]
        raise SystemExit(
            "Scenarios do not share one seed set, so cross-scenario comparisons would be "
            "UNPAIRED:\n" + "\n".join(lines)
            + "\n\nRe-run with the same SEEDS (the S-TMSI table in particular is only "
              "meaningful paired: the cell branches must land on identical trajectories).")
    return frames, {**ref, "seeds": list(next(iter(uniq_seeds)))}


def _assert_cell_invariant_under_stmsi(off: pd.DataFrame, on: pd.DataFrame):
    """S-TMSI reallocation only fragments identities; it is driven by an isolated RNG stream
    and never perturbs the physical trajectory, and the cell branches never read identity.
    With matched seeds the cell MAEs must therefore be EXACTLY equal, per seed — not merely
    close on average. A difference means the pairing broke or identity leaked into the cell
    features, and either way the S-TMSI table would be wrong."""
    a = off.set_index("seed"); b = on.set_index("seed")
    for model in ("Cell_XGB", "Cell_LSTM"):
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
                raise SystemExit(
                    f"{col} changed under S-TMSI reallocation:\n{rows}\n"
                    "  Cell features do not use UE identity and reallocation uses an isolated "
                    "RNG stream, so this must be exactly 0. Check that both runs used the same "
                    "seeds/config, and that no identity-derived signal reached the cell "
                    "features.")


# --------------------------------------------------------------------------------------
def table2_main(params_dir: Path, protocol: dict) -> str:
    """Main held-out performance from the tuned final summary (test seeds).

    The headline and the robustness tables must describe the SAME estimator: if the
    scenarios were produced with isotonic calibration while final_test_summary.json is the
    raw protocol, Table 2 and Tables 3-5 would silently be different models."""
    p = params_dir / "final_test_summary.json"
    if not p.exists():
        raise SystemExit(f"Table 2 needs {p}. Run: tune_runner --final --final-variant "
                         f"<variant> --test-seeds ...")
    blob = json.loads(p.read_text())
    # new format {"meta":..., "results":...}; older summaries are flat = raw protocol
    s = blob.get("results", blob)
    meta = blob.get("meta")
    if meta is None:
        meta = {"final_variant": "R_raw", "perue_calib": "none", "calib_scope": "none"}
        print("  (info) final_test_summary.json predates variant tracking; assuming the raw "
              "protocol (R_raw).")
    want_calib = protocol.get("perue_calib", "none")
    want_scope = protocol.get("calib_scope", "none")
    got_calib, got_scope = meta.get("perue_calib", "none"), meta.get("calib_scope", "none")
    if (got_calib, got_scope) != (want_calib, want_scope):
        variant = {("none", "none"): "R_raw",
                   ("isotonic", "perue_only"): "S_perue_cal",
                   ("linear", "perue_only"): "S_perue_cal"}.get((want_calib, want_scope),
                                                                "C_full_cal")
        raise SystemExit(
            f"Protocol mismatch between Table 2 and Tables 3-5:\n"
            f"    headline ({p}): perue_calib={got_calib!r} scope={got_scope!r}\n"
            f"    robustness scenarios:  perue_calib={want_calib!r} scope={want_scope!r}\n"
            f"Re-run `tune_runner --final --final-variant {variant}` or re-run the "
            f"robustness suite with matching --perue-calib/--calib-scope.")
    # The headline must also come from the SAME tuning study and split as the scenarios:
    # an controlled Table 2 next to equal-budget robustness would be two different
    # experiments in one paper.
    for key in ("study_prefix", "base_study_prefix", "comparison", "train_ratio"):
        got, want = meta.get(key), protocol.get(key)
        if got is None or want is None:
            continue
        if key == "train_ratio":
            same = abs(float(got) - float(want)) < 1e-9
        else:
            same = got == want
        if not same:
            raise SystemExit(
                f"Protocol mismatch between Table 2 and Tables 3-5: {key}: "
                f"headline={got!r}, robustness={want!r}. Both must come from the same tuned "
                f"study and the same temporal split.")
    # Strongest check: the exact tuned parameter FILES behind both sides. An empty or
    # partial key overlap is itself a failure — it means the two sides do not even describe
    # the same set of (mode, branch) models, so "no differing hash" would be vacuous.
    hs, hr = meta.get("param_file_sha256"), protocol.get("param_file_sha256")
    if hs and hr:
        only_h, only_r = sorted(set(hs) - set(hr)), sorted(set(hr) - set(hs))
        if not (set(hs) & set(hr)):
            raise SystemExit(
                "Table 2 and the robustness scenarios share NO tuned-parameter entries "
                f"(headline keys: {sorted(hs)[:4]}..., robustness keys: {sorted(hr)[:4]}...). "
                "They were built from different studies or different modes; the protocol "
                "check cannot be satisfied vacuously.")
        if only_h or only_r:
            raise SystemExit(
                "Table 2 and the robustness scenarios cover different (mode/branch) models:\n"
                f"    only in headline:   {only_h}\n"
                f"    only in robustness: {only_r}\n"
                "Both must report the same six models for the same modes.")
        diff = sorted(k for k in hs if hs[k] != hr[k])
        if diff:
            raise SystemExit(
                "Table 2 and the robustness scenarios used DIFFERENT tuned parameter files "
                f"for: {diff}. The prefixes match but the file contents do not — re-run the "
                "headline and the scenarios against one tuning directory.")
    else:
        print("  (info) parameter-file hashes unavailable on one side; protocol verified on "
              "prefixes/variant only. Re-run --final to record them.")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & Mode A (MAE) & Mode B (MAE) & A $-$ B [95\% CI] \\", r"\midrule"]
    for m in MODELS:
        if m not in s:
            continue
        a, b = s[m].get("A"), s[m].get("B")
        pa = s[m].get("paired_A_minus_B")
        ci = (f"{pa['mean']:+.2f} [{pa['ci95'][0]:+.2f}, {pa['ci95'][1]:+.2f}]"
              if pa else "--")
        lines.append(f"{PRETTY[m]} & {_fmt((a['mean'], a['std'])) if a else '--'} & "
                     f"{_fmt((b['mean'], b['std'])) if b else '--'} & {ci} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _axis_table(frames: dict, axis: list, axis_label: str, nmae: bool) -> str:
    """Shared body for Tables 3 and 4: one row per (condition, mode), one column per model."""
    head = " & ".join(PRETTY[m] for m in MODELS)
    lines = [r"\begin{tabular}{ll r " + "c" * len(MODELS) + "}", r"\toprule",
             f"{axis_label} & Mode & Mean present & {head} \\\\", r"\midrule"]
    for key, label in axis:
        df = frames[key]
        col = "mean_present_test" if "mean_present_test" in df.columns else "mean_present"
        mp = float(pd.to_numeric(df[col], errors="coerce").mean())
        for i, mode in enumerate(MODES):
            cells = [_fmt(_cells(df, m, mode, nmae), 3 if nmae else 2) for m in MODELS]
            first = label if i == 0 else ""
            mpc = f"{mp:.1f}" if i == 0 else ""
            lines.append(f"{first} & {mode} & {mpc} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table5_stmsi(frames: dict) -> str:
    """Paired S-TMSI degradation. Seeds are guaranteed matched by _check_protocol, so the
    cell branches MUST come out identical: reallocation is driven by an isolated RNG stream
    and never touches the physical trajectory. A non-zero cell delta therefore indicates a
    broken pairing (or a leaked identity signal into the cell features), not an effect."""
    off, on = frames["default"], frames["stmsi"]
    _assert_cell_invariant_under_stmsi(off, on)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & \multicolumn{2}{c}{Mode A} & \multicolumn{2}{c}{Mode B} "
             r"& \multicolumn{2}{c}{Mode B degradation} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & no realloc. & realloc. & no realloc. & realloc. & abs. & rel. \\",
             r"\midrule"]
    for m in MODELS:
        a0, a1 = _cells(off, m, "A", False), _cells(on, m, "A", False)
        b0, b1 = _cells(off, m, "B", False), _cells(on, m, "B", False)
        if b0 and b1:
            d_abs, d_rel = f"{b1[0] - b0[0]:+.2f}", f"{100 * (b1[0] / b0[0] - 1):+.0f}\\%"
        else:
            d_abs = d_rel = "--"
        lines.append(f"{PRETTY[m]} & {_fmt(a0)} & {_fmt(a1)} & {_fmt(b0)} & {_fmt(b1)} "
                     f"& {d_abs} & {d_rel} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-prefix", default="robust_s3c5",
                    help="prefix of the robustness output dirs (…_default, …_tiny, …)")
    ap.add_argument("--params-dir", default="tune_out_s3c5",
                    help="tuned run dir holding final_test_summary.json")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()

    # every scenario the five tables need; missing ones are a hard error, not a skipped row
    needed = ["default"] + [k for k, _ in SCALES if k != "default"] \
                         + [k for k, _ in MIXES if k != "default"] + ["stmsi"]
    frames, protocol = _check_protocol(args.robust_prefix, needed)
    print(f"protocol: prefix={protocol['study_prefix']} comparison={protocol['comparison']} "
          f"train_ratio={protocol['train_ratio']} "
          f"perue_calib={protocol['perue_calib']}/{protocol['calib_scope']} "
          f"seeds={protocol['seeds']}")

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    tables = {
        "table2_main.tex": table2_main(Path(args.params_dir), protocol),
        "table3_scale_mae.tex": _axis_table(frames, SCALES, "Scale", nmae=False),
        "table3_scale_nmae.tex": _axis_table(frames, SCALES, "Scale", nmae=True),
        "table4_mix_mae.tex": _axis_table(frames, MIXES, "Traffic mix", nmae=False),
        "table5_stmsi.tex": table5_stmsi(frames),
    }
    for name, body in tables.items():
        (outd / name).write_text(body)
        print(f"\n===== {name} =====")
        print(body)
    (outd / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"saved -> {outd}/  (protocol.json records the verified single protocol)")


if __name__ == "__main__":
    main()
