#!/usr/bin/env python3
"""
trial_times.py — per-branch trial durations from Optuna journal files.

Takes ONE OR MORE journal paths. An earlier version took a single path plus an optional
study-name filter as the second argument, which meant `trial_times.py a.journal b.journal`
silently produced no output: the second journal was read as a filter that matched nothing.
The filter is now an explicit --study flag, so a path can never be mistaken for one.

    python trial_times.py tune_s3c5.journal tune_s3c5_cal.journal
    python trial_times.py tune_s3c5.journal --study cell_lstm
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

CREATE_STUDY, COMPLETE = 0, 1


def parse_dt(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def report(path: Path, want: str | None):
    names, nid, starts, comps, tstudy, ntr = {}, 0, {}, {}, {}, 0
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("op_code") == CREATE_STUDY and "study_name" in rec:
            if rec["study_name"] not in names.values():
                names[nid] = rec["study_name"]
                nid += 1
        elif "datetime_start" in rec and "study_id" in rec:
            tstudy[ntr] = rec["study_id"]
            starts[ntr] = parse_dt(rec["datetime_start"])
            ntr += 1
        elif "datetime_complete" in rec and "trial_id" in rec:
            if rec.get("state", COMPLETE) == COMPLETE:
                comps[rec["trial_id"]] = parse_dt(rec["datetime_complete"])

    print(f"\n{path}")
    if not names:
        print("  no studies found")
        return
    shown = 0
    for sid, nm in sorted(names.items()):
        if want and want not in nm:
            continue
        ts = [t for t, s in tstudy.items() if s == sid]
        durs = sorted((comps[t] - starts[t]).total_seconds() / 60
                      for t in ts if starts.get(t) and comps.get(t))
        if not durs:
            print(f"  {nm}: no completed trials")
            shown += 1
            continue
        st = [starts[t] for t in ts if starts.get(t)]
        en = [comps[t] for t in ts if comps.get(t)]
        total = (max(en) - min(st)).total_seconds() / 3600
        print(f"  {nm}")
        print(f"    {len(durs)} trials   first start {min(st):%m/%d %H:%M}   "
              f"last finish {max(en):%m/%d %H:%M}   span {total:.1f}h")
        print(f"    per trial  median {durs[len(durs) // 2]:.1f} min   "
              f"min {durs[0]:.1f}   max {durs[-1]:.1f}   total {sum(durs) / 60:.1f}h")
        shown += 1
    if want and shown == 0:
        print(f"  no study matches '{want}'")


def main() -> int:
    args = sys.argv[1:]
    want = None
    if "--study" in args:
        i = args.index("--study")
        want = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    paths = [Path(a) for a in args]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        if p.exists():
            report(p, want)
        else:
            print(f"\n{p}  not found")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
