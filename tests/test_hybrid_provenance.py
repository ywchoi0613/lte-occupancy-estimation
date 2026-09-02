"""tests/test_hybrid_provenance.py — the tuned-hybrid provenance guard.

The guard's whole job is to refuse hybrid params that were tuned for inputs other than the
ones the consuming run will feed them. Each case below is a mistake that would otherwise
land silently in a paper table.

    python -m pytest tests/test_hybrid_provenance.py -q      (or: python tests/test_hybrid_provenance.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lte_occupancy.experiments import tuning as TU     # noqa: E402

BRANCHES = ("hybrid_xgb", "hybrid_lstm")


def _study(d: Path, prefix: str, *, meta: dict | None, modes=("A", "B")):
    """Write one hybrid study: best params for each mode + one run_meta per worker."""
    for branch in BRANCHES:
        for m in modes:
            (d / f"best_{prefix}_{branch}_{m}.json").write_text(json.dumps({"params": {}}))
        if meta is None:
            continue                                   # orphaned params: no provenance
        for w in range(4):
            (d / f"run_meta_{prefix}_{branch}_w{w}.json").write_text(
                json.dumps({"kind": branch, "branch": branch, **meta}))


def _raw_meta(**over):
    """A modern raw-tuned study (records perue_calib explicitly)."""
    return {"comparison": "equal_budget", "selection": "val", "dev_seeds": [7, 13, 42],
            "study_prefix": "s1e", "base_study_prefix": "s1e", "perue_calib": "none", **over}


def _cal_meta(**over):
    base = dict(study_prefix="s1e_cal", base_study_prefix="s1e", perue_calib="isotonic")
    base.update(over)
    return _raw_meta(**base)


def _legacy_meta(**over):
    """A study tuned BEFORE --perue-calib existed: no perue_calib, no base_study_prefix.
    This is what tune_out_s1e actually holds today."""
    m = _raw_meta(**over)
    m.pop("perue_calib"); m.pop("base_study_prefix")
    return m


def _check(d, prefix, variant, *, base=None, comparison="equal_budget", selection="val"):
    TU.check_hybrid_provenance(
        d, prefix,
        perue_calib=("none" if variant == "R_raw" else "isotonic"),
        calib_scope=TU.VARIANT_SCOPE[variant],
        base_study_prefix=base or prefix, comparison=comparison, selection=selection,
        context=f"--final-variant {variant}")


def _raises(fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except SystemExit as e:
        return str(e)
    raise AssertionError("guard did NOT fire but should have")


def run(tmp: Path):
    ok = []

    # ---- 1. legacy raw study + R_raw / S_perue_cal -> must PASS -------------------------
    # The existing tune_out_s1e has no perue_calib field. R and S feed the hybrid raw bases,
    # so those params are valid and the guard must not block today's working headline.
    d = tmp / "legacy"; d.mkdir()
    _study(d, "s1e", meta=_legacy_meta())
    _check(d, "s1e", "R_raw")
    _check(d, "s1e", "S_perue_cal")
    ok.append("legacy raw study passes R_raw and S_perue_cal")

    # ---- 2. legacy raw study + C_full_cal -> must FAIL (THE HOLE) -----------------------
    # Old code: rec is None -> `continue` -> silently accepted. This is the exact command
    # `MODE=final FINAL_VARIANT=C_full_cal PREFIX=s1e` that must never run.
    msg = _raises(_check, d, "s1e", "C_full_cal")
    assert "hybrid_xgb" in msg and "perue_calib" in msg, msg
    assert "BRANCHES=" in msg and "PREFIX=s1e_cal" in msg and "BASE_PREFIX=s1e" in msg, msg
    ok.append("legacy raw study is REFUSED for C_full_cal (was silently accepted)")

    # ---- 3. properly re-tuned calibrated study + C_full_cal -> must PASS ----------------
    d = tmp / "cal"; d.mkdir()
    _study(d, "s1e", meta=_legacy_meta())               # base study stays put
    _study(d, "s1e_cal", meta=_cal_meta())              # hybrids re-tuned beside it
    _check(d, "s1e_cal", "C_full_cal", base="s1e")
    ok.append("re-tuned s1e_cal study passes C_full_cal")

    # ---- 4. prefix isolation: s1e_cal artifacts must not leak into the s1e check --------
    _check(d, "s1e", "R_raw")
    ok.append("prefix 's1e' does not pick up 's1e_cal' artifacts")

    # ---- 5. calibrated hybrids used for R/S -> must FAIL (reverse direction, NEW) -------
    msg = _raises(_check, d, "s1e_cal", "R_raw", base="s1e")
    assert "perue_calib" in msg, msg
    ok.append("calibrated hybrids are REFUSED for R_raw (reverse direction)")

    # ---- 6. tuned params with no run_meta at all -> must FAIL ---------------------------
    d = tmp / "orphan"; d.mkdir()
    _study(d, "s1e", meta=None)
    msg = _raises(_check, d, "s1e", "C_full_cal")
    assert "provenance is unknown" in msg, msg
    ok.append("orphaned params (no run_meta) are REFUSED")

    # ---- 7. workers disagree on the protocol -> must FAIL -------------------------------
    d = tmp / "split"; d.mkdir()
    _study(d, "s1e_cal", meta=_cal_meta())
    (d / "run_meta_s1e_cal_hybrid_xgb_w1.json").write_text(
        json.dumps(_cal_meta(perue_calib="none")))      # one worker tuned on raw
    msg = _raises(_check, d, "s1e_cal", "C_full_cal", base="s1e")
    assert "disagree on perue_calib" in msg, msg
    ok.append("workers disagreeing on perue_calib are REFUSED")

    # ---- 8. base_study_prefix / comparison mismatches -> must FAIL ----------------------
    d = tmp / "wrongbase"; d.mkdir()
    _study(d, "s1e_cal", meta=_cal_meta(base_study_prefix="s1c"))
    msg = _raises(_check, d, "s1e_cal", "C_full_cal", base="s1e")
    assert "base_study_prefix" in msg, msg
    ok.append("hybrid tuned on s1c bases is REFUSED for an s1e run")

    d = tmp / "wrongcmp"; d.mkdir()
    _study(d, "s1c_cal", meta=_cal_meta(comparison="controlled"))
    msg = _raises(_check, d, "s1c_cal", "C_full_cal", base="s1c", comparison="equal_budget")
    assert "comparison" in msg, msg
    ok.append("controlled-tuned hybrids are REFUSED for an equal_budget run")

    # ---- 9. test_oracle-tuned hybrids reported as a val-selected headline -> FAIL -------
    d = tmp / "oracle"; d.mkdir()
    _study(d, "s1e_cal", meta=_cal_meta(selection="test_oracle"))
    msg = _raises(_check, d, "s1e_cal", "C_full_cal", base="s1e", selection="val")
    assert "selection" in msg, msg
    ok.append("test_oracle-tuned hybrids are REFUSED for a val-selected headline")

    # ---- 10. a study with no hybrid params at all is skipped, not crashed ---------------
    d = tmp / "empty"; d.mkdir()
    _check(d, "s1e", "C_full_cal")
    ok.append("study with no hybrid params is skipped (loader reports it downstream)")

    # ---- 11. controlled comparison writes best_*_AB.json, not _A/_B ---------------------
    d = tmp / "ctrl"; d.mkdir()
    _study(d, "s1c", meta=_legacy_meta(comparison="controlled", study_prefix="s1c"),
           modes=("AB",))
    _check(d, "s1c", "R_raw", comparison="controlled")
    msg = _raises(_check, d, "s1c", "C_full_cal", comparison="controlled")
    assert "perue_calib" in msg, msg
    ok.append("controlled studies (best_*_AB.json) are matched by the same globs")

    return ok


def test_hybrid_provenance(tmp_path):
    run(Path(tmp_path))


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        for line in run(Path(t)):
            print(f"  PASS  {line}")
    print("\nall hybrid-provenance cases pass")
