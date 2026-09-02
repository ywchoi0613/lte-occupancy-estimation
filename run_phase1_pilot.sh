#!/usr/bin/env bash
# run_phase1_pilot.sh — Phase 1: G1 tests -> arrival calibration -> 4-scenario
# fidelity gates (parallel, CPU-only) -> GATE_REPORT.md.
#
# Resource policy: CPU-only truth simulation; probes the 1-min load average
# and uses only the FREE cores at nice 15, so running FL jobs keep priority.
#
# Usage:
#   bash run_phase1_pilot.sh
#   DAYS=1 bash run_phase1_pilot.sh                # faster (no drift gate)
#   PHASE1_SKIP_TESTS=1 bash run_phase1_pilot.sh   # skip G1 (not recommended)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# env hygiene: stray LTE_* smoke/sweep knobs must not leak into the gates
for v in $(env | grep '^LTE_' | cut -d= -f1); do unset "$v"; done

OUT=${OUT:-fidelity_out}
LOGS=${LOGS:-logs_phase1}
mkdir -p "$OUT" "$LOGS"
NICE=${NICE:-15}
SEED=${SEED:-7}
DAYS=${DAYS:-2}
TARGET_NBAR=${TARGET_NBAR:-320}
TOTAL_RUNS=${TOTAL_RUNS:-220}
SCENARIOS=${SCENARIOS:-"resident office transport comprehensive"}
PY=${PY:-python}

# fresh gates: never let stale outputs from an earlier run leak into the verdict
for sc in $SCENARIOS; do rm -rf "$OUT/$sc"; done
rm -f "$OUT/GATE_REPORT.md" "$OUT/G1_PASS" "$OUT/calibration_comprehensive.json"

TOTAL=$(nproc)
LOAD=$(awk '{printf "%d", $1 + 0.999}' /proc/loadavg)
FREE=$(( TOTAL - LOAD )); (( FREE < 1 )) && FREE=1
WORKERS=$FREE
NSC=$(wc -w <<< "$SCENARIOS")
(( WORKERS > NSC )) && WORKERS=$NSC
echo "[pilot] cores=$TOTAL load1=$LOAD -> free=$FREE; scenario workers=$WORKERS (nice $NICE)"

# ---------------------------------------------------------------- G1: tests
rm -f "$OUT/G1_PASS"
if [[ "${PHASE1_SKIP_TESTS:-0}" != "1" ]]; then
  echo "[pilot] G1: pytest (phase1 + correctness + provenance)"
  nice -n "$NICE" "$PY" -m pytest \
      tests/test_phase1.py tests/test_correctness.py tests/test_hybrid_provenance.py \
      -q 2>&1 | tee "$LOGS/g1_pytest.log"
  touch "$OUT/G1_PASS"
else
  echo "[pilot] G1 SKIPPED (PHASE1_SKIP_TESTS=1) — no G1_PASS marker, report will FAIL" \
    | tee "$LOGS/g1_pytest.log"
fi

# ------------------------------------------------- calibrate (comprehensive)
echo "[pilot] calibrating arrival scale on 'comprehensive' (target n=$TARGET_NBAR)"
nice -n "$NICE" "$PY" -m lte_occupancy.experiments.validate_fidelity --mode calibrate \
    --scenario comprehensive --seed "$SEED" --target-nbar "$TARGET_NBAR" \
    --out "$OUT" 2>&1 | tee "$LOGS/calibrate.log"

# ------------------------------------------------- gate runs (parallel)
running=0
for sc in $SCENARIOS; do
  echo "[pilot] gate: $sc (${DAYS}d recorded, warm-up 1d)"
  nice -n "$NICE" "$PY" -m lte_occupancy.experiments.validate_fidelity --mode gate \
      --scenario "$sc" --days "$DAYS" --seed "$SEED" \
      --target-nbar "$TARGET_NBAR" --workers "$WORKERS" \
      --total-runs "$TOTAL_RUNS" --out "$OUT" \
      > "$LOGS/gate_$sc.log" 2>&1 &
  running=$((running + 1))
  if (( running >= WORKERS )); then
    wait -n
    running=$((running - 1))
  fi
done
wait

# ------------------------------------------------- report + verdict
"$PY" -m lte_occupancy.experiments.validate_fidelity --mode report --out "$OUT" \
    --target-nbar "$TARGET_NBAR" | tee "$LOGS/report.log"
echo "[pilot] done -> $OUT/GATE_REPORT.md (T1-T3 feed for the interim report)"
