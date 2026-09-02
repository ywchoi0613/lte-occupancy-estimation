#!/usr/bin/env bash
# collect_paper_bundle.sh — gather everything needed to verify the results and write the
# paper, and nothing else.
#
# WHAT IS DELIBERATELY LEFT OUT
#   trace_seed*.parquet / .csv.gz   ~50 MB x 215 units. These are the raw per-slot traces.
#   preds_seed*.npz                 per-seed predictions.
#   robust_*__s<seed>/              the per-seed shard dirs, already merged into the
#                                   scenario dirs. Keeping both doubles the size.
# None of these are read by the table generator or by any provenance check. They stay on
# the server; if a reviewer ever wants them, the GitHub repo plus the seeds reproduces
# them exactly.
#
# WHAT IS KEPT, AND WHY
#   tables/       the actual paper tables and the tidy CSV behind them
#   main/         final_test_summary.json (Table 2) + the 10 shards + the 6 tuned
#                 parameter files, i.e. the exact estimator definitions
#   robustness/   per scenario: results.csv, resolved_config.json, provenance, truth
#                 hashes. This is what every correctness claim in the paper rests on.
#   provenance/   Optuna journals (full tuning history), fidelity gate, calibration
#   analysis/     the scripts that turn the above back into the tables
#   TIMINGS.txt   per-branch trial durations, for the compute-cost paragraph
#   MANIFEST.txt  file counts and sizes, so it is obvious if something is missing
#
# USAGE
#   bash collect_paper_bundle.sh            # dry run: shows what would be copied + size
#   CONFIRM=1 bash collect_paper_bundle.sh  # build the zip
set -uo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-paper_bundle}
ZIP=${ZIP:-lte_paper_bundle_$(date +%Y%m%d).zip}
TUNE=${TUNE:-tune_out_s3c5}
TABLES=${TABLES:-paper_tables_s3c5_cal}
ROBUST=${ROBUST:-robust_s3c5_cal}
PREFIX=${PREFIX:-s3c5}
CAL=${CAL:-s3c5_cal}

say () { printf '  %-46s %s\n' "$1" "$2"; }
sz  () { du -sh "$1" 2>/dev/null | cut -f1; }

echo "=== checking what will be collected"
missing=0
for p in "$TABLES" "$TUNE/final_test_summary.json" "tune_${PREFIX}.journal" \
         "tune_${CAL}.journal" "${ROBUST}_main"; do
  if [ -e "$p" ]; then say "$p" "OK ($(sz "$p"))"; else say "$p" "MISSING"; missing=1; fi
done
nsc=$(ls -d ${ROBUST}_*/ 2>/dev/null | grep -v "__s" | wc -l | tr -d ' ')
say "robustness scenario dirs" "$nsc"
nsh=$(ls -1 "$TUNE"/final_test_summary.shard_*.json 2>/dev/null | wc -l | tr -d ' ')
say "main shards" "$nsh"

echo
echo "=== excluded (kept only on the compute host)"
# du prints nothing when the glob matches nothing, and `|| echo 0` does not fire because
# the pipeline still succeeds — so the size has to be defaulted after the fact.
esz () { local r; r=$(du -ch $1 2>/dev/null | tail -1 | cut -f1); echo "${r:-0}"; }
say "trace_seed*"  "$(esz "${ROBUST}_*/trace_seed*")"
say "preds_seed*"  "$(esz "${ROBUST}_*/preds_seed*")"
say "shard dirs ${ROBUST}_*__s*/" "$(esz "${ROBUST}_*__s*")"

if [ "${CONFIRM:-0}" != "1" ]; then
  echo
  echo "=== DRY RUN. To actually build it: CONFIRM=1 bash collect_paper_bundle.sh"
  [ "$missing" = 1 ] && echo "    (items are missing - check before proceeding)"
  exit 0
fi

rm -rf "$OUT"
mkdir -p "$OUT"/{tables,main,robustness,provenance,analysis}

# ---- tables
cp -r "$TABLES"/. "$OUT/tables/" 2>/dev/null

# ---- main: headline + shards + the 6 estimator definitions
cp "$TUNE"/final_test_summary*.json "$OUT/main/" 2>/dev/null
cp "$TUNE"/best_${PREFIX}_*_AB.json "$TUNE"/best_${CAL}_*_AB.json "$OUT/main/" 2>/dev/null
cp "$TUNE"/run_meta_*.json "$OUT/main/" 2>/dev/null

# ---- robustness: the small, load-bearing files only
for d in $(ls -d ${ROBUST}_*/ 2>/dev/null | grep -v "__s"); do
  n=$(basename "$d")
  mkdir -p "$OUT/robustness/$n"
  for f in results.csv resolved_config.json tuned_model_provenance.json \
           parameter_registry.csv environment.json manifest.json; do
    [ -f "$d$f" ] && cp "$d$f" "$OUT/robustness/$n/"
  done
  cp "$d"truth_hashes_seed*.json "$OUT/robustness/$n/" 2>/dev/null
done

# ---- provenance: full tuning history + the gate that licensed the DGP
cp tune_${PREFIX}.journal tune_${CAL}.journal "$OUT/provenance/" 2>/dev/null
[ -d fidelity_out ] && { mkdir -p "$OUT/provenance/fidelity_out"
  cp fidelity_out/GATE_REPORT.md fidelity_out/calibration_comprehensive.json \
     fidelity_out/G1_PASS "$OUT/provenance/fidelity_out/" 2>/dev/null; }

# ---- analysis scripts: without these the bundle is data with no way to rebuild the tables
for f in make_robust5d_tables.py check_single_axis.py count_trials.py trial_times.py \
         show_results.py diff_stmsi_truth.py status.sh \
         run_paper5d_controlled.sh run_robustness5d_controlled_seeds.sh \
         topup_lstm.sh patch_family_filter.py memwatch.sh eta.py; do
  [ -f "$f" ] && cp "$f" "$OUT/analysis/"
done

# ---- timings for the compute-cost paragraph, extracted rather than shipping raw logs
{
  echo "Per-branch trial wall-clock, measured from the Optuna journals"
  python trial_times.py tune_${PREFIX}.journal tune_${CAL}.journal 2>/dev/null
  echo
  echo "Main shard wall-clock and peak memory (/usr/bin/time -v)"
  grep -H -E "Elapsed \(wall|Maximum resident" logs_tune_${PREFIX}/final_seed_*.log 2>/dev/null
} > "$OUT/TIMINGS.txt" 2>&1

# ---- manifest: makes an incomplete bundle obvious at a glance
{
  echo "created  $(date '+%F %T')"
  echo "host     $(hostname)"
  echo
  echo "[contents]"
  printf "  tables      %s tex files, %s\n" \
    "$(ls -1 "$OUT"/tables/*.tex 2>/dev/null | wc -l | tr -d ' ')" "$(sz "$OUT/tables")"
  printf "  main        %s shards, %s best-param files, %s\n" \
    "$(ls -1 "$OUT"/main/final_test_summary.shard_*.json 2>/dev/null | wc -l | tr -d ' ')" \
    "$(ls -1 "$OUT"/main/best_*_AB.json 2>/dev/null | wc -l | tr -d ' ')" "$(sz "$OUT/main")"
  printf "  robustness  %s scenarios, %s\n" \
    "$(ls -1d "$OUT"/robustness/*/ 2>/dev/null | wc -l | tr -d ' ')" "$(sz "$OUT/robustness")"
  printf "  provenance  %s\n" "$(sz "$OUT/provenance")"
  printf "  analysis    %s scripts\n" "$(ls -1 "$OUT"/analysis 2>/dev/null | wc -l | tr -d ' ')"
  echo
  echo "[protocol]"
  sed 's/^/  /' "$OUT/tables/PROTOCOL.txt" 2>/dev/null
  echo
  echo "[headline seeds]"
  python - "$OUT/main/final_test_summary.json" <<'PYEOF' 2>/dev/null
import json, sys
m = json.load(open(sys.argv[1]))["meta"]
for k in ("comparison", "selection", "final_variant", "horizon", "train_ratio", "test_seeds"):
    print(f"  {k}: {m.get(k)}")
PYEOF
} > "$OUT/MANIFEST.txt" 2>&1

rm -f "$ZIP"
zip -qr "$ZIP" "$OUT"
echo
echo "=== done: $ZIP  ($(sz "$ZIP"))"
echo
cat "$OUT/MANIFEST.txt"
echo
echo "To rebuild the tables from the bundle and verify them:"
echo "  unzip $ZIP && cd $OUT/analysis"
echo "  (point --prefix at robustness/ and --params-dir at main/)"
