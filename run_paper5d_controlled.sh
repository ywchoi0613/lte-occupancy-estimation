#!/usr/bin/env bash
# run_paper5d_controlled.sh — the final 5-day CONTROLLED protocol, end to end.
#
# This is the experimental protocol reported in the "Experimental protocol" section of
# the paper: the controlled comparison in which Modes A and B share one hyperparameter
# set, so that any A-B difference is attributable to observability rather than to tuning.
#
# WHAT THIS SCRIPT DOES
#   Stage 0  refuse unless the fidelity gate passed and the calibration file exists
#   Stage 1  s3c5      base six-branch Optuna tuning, controlled, 5-day dev traces
#   Stage 2  s3c5_cal  re-tune ONLY the two hybrids on isotonic-calibrated per-UE bases
#   Stage 3  MAIN final evaluation, C_full_cal, held-out seeds, one shard per seed
#
# WHAT IT NEVER TOUCHES
#   Optuna journals and output directories belonging to any other study prefix. The
#   script aborts if a journal for its own prefix already exists, so a second invocation
#   cannot append trials to a finished study and silently inflate its budget
#   (use RESUME=1 to deliberately continue one).
#
# SEED POLICY — fixed before any result is seen
#   Main       ten seeds (101-110), all six models
#   Robustness five seeds (101-105), XGB family  [run_robustness5d_controlled_seeds.sh]
#
#   The ten is committed IN ADVANCE and is not conditional on what the intervals look
#   like. "Run five, inspect the CI, extend to ten if it straddles zero" is optional
#   stopping: the decision to collect more data depends on the observed significance, so
#   the reported interval no longer has its nominal coverage. That the sharding makes the
#   extension cheap is precisely what makes the temptation dangerous, so the choice is
#   removed rather than left to judgement.
#
#   Why ten is affordable here: a paired Student-t half-width is t_{.975,n-1}*sd/sqrt(n),
#   so n=5 gives 1.242*sd against 0.715*sd at n=10 — the interval is 1.74x wider, not
#   1.41x. Main is ONE configuration, so ten seeds costs ten shards against the ~215-unit
#   robustness suite. Buying the headline's power is roughly 5% of the total budget.
#   Robustness keeps five because there the quantity of interest is a degradation ratio
#   across many scenarios, not a single interval carrying the paper's claim.
#
# USAGE
#   bash run_paper5d_controlled.sh                      # dry-run plan, writes nothing
#   CONFIRM=1 nohup bash run_paper5d_controlled.sh > paper5d.out 2>&1 &
#   STAGES="3" CONFIRM=1 bash run_paper5d_controlled.sh # main eval only (after tuning)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PY=${PY:-python}
FID=${FID:-fidelity_out}
CAL="$FID/calibration_comprehensive.json"

PREFIX=${PREFIX:-s3c5}
CAL_PREFIX=${CAL_PREFIX:-${PREFIX}_cal}
OUT=${OUT:-tune_out_${PREFIX}}
LOGDIR=${LOGDIR:-logs_tune_${PREFIX}}

DEV_SEEDS=${DEV_SEEDS:-"7 13 42"}
MAIN_SEEDS=${MAIN_SEEDS:-"101 102 103 104 105 106 107 108 109 110"}
TRIALS_TOTAL=${TRIALS_TOTAL:-48}
SAMPLER_SEED=${SAMPLER_SEED:-42}
STAGES=${STAGES:-"1 2 3"}

# Which estimator families Stages 1 and 2 tune. Default = both, i.e. unchanged.
# TUNE_FAMILIES=xgb runs ONLY the XGB branches, which is what the robustness suite needs:
# it lets the whole robustness run start while the LSTM branches are still outstanding.
# Stage 3 (Main) always needs all six and is unaffected by this knob.
TUNE_FAMILIES=${TUNE_FAMILIES:-"xgb lstm"}
has_fam () { [[ " $TUNE_FAMILIES " == *" $1 "* ]]; }

# Worker counts follow the measured per-worker footprint, not the GPU count. One tuning
# worker holding 3 dev seeds x 2 modes at 432,000 slots peaks near 36 GB of resident
# memory, so on the 188 GB host of the paper three LSTM workers fit (~108 GB) and four do
# not (~11 GB headroom, which is not enough and will be OOM-killed). XGB branches train on
# the CPU with a much smaller footprint, so they keep all four. Retune these for your host.
GPUS_XGB=${GPUS_XGB:-"0 1 2 3"}
GPUS_LSTM=${GPUS_LSTM:-"0 1 2"}

# Stage 3 runs one shard per seed. Ten shards SEQUENTIALLY is ~60 h; the only thing that
# stops them running together is host RAM, not the GPUs — a full six-branch A+B process at
# 432k slots peaked at ~70 GB (139 GB measured at 864k). Two fit; three do not.
MAIN_GPUS=${MAIN_GPUS:-"0 1 2 3"}
MAIN_JOBS=${MAIN_JOBS:-2}

# Scheduled-outage support. A shard killed mid-run is not corrupting (shard JSONs are
# written atomically and a missing one is simply re-run), but finishing cleanly avoids
# wasting a 6 h shard, so new shards stop launching once the reserve no longer fits.
DEADLINE=${DEADLINE:-}                  # e.g. "2026-08-17 06:00" — empty = no limit
SHARD_HOURS=${SHARD_HOURS:-8}           # worst-case shard, used as the launch reserve

if [ -n "$DEADLINE" ]; then
  DEADLINE_EPOCH=$(date -d "$DEADLINE" +%s 2>/dev/null) || {
    echo "[5d] cannot parse DEADLINE='$DEADLINE'"; exit 1; }
else
  DEADLINE_EPOCH=""
fi
time_left_for () {
  [ -z "$DEADLINE_EPOCH" ] && return 0
  [ $(( DEADLINE_EPOCH - $(date +%s) )) -ge $(( ${1%%.*} * 3600 )) ]
}
hours_left () {
  [ -z "$DEADLINE_EPOCH" ] && { echo "unbounded"; return; }
  awk -v d="$DEADLINE_EPOCH" -v n="$(date +%s)" 'BEGIN{printf "%.1f h", (d-n)/3600}'
}

# ---------------------------------------------------------------- Stage 0: gates
if [[ "${FORCE_GATE:-0}" != "1" ]]; then
  ok=1
  grep -q "OVERALL: PASS" "$FID/GATE_REPORT.md" 2>/dev/null || { echo "[5d] gate report missing/failed: $FID/GATE_REPORT.md"; ok=0; }
  [[ -f "$FID/G1_PASS" ]] || { echo "[5d] missing $FID/G1_PASS"; ok=0; }
  [[ -f "$CAL" ]]        || { echo "[5d] missing $CAL"; ok=0; }
  [[ "$ok" == "1" ]] || { echo "[5d] REFUSING. Run 'bash run_phase1_pilot.sh' first (or FORCE_GATE=1)."; exit 1; }
fi

# ---------------------------------------------------------------- canonical 5-day DGP
# Every LTE_* is cleared first: an inherited LTE_STMSI_REALLOC=1 or LTE_SCALE_PROFILE=tiny
# from the caller's shell would silently retune the paper on a different DGP.
for v in $(env | grep '^LTE_' | cut -d= -f1); do unset "$v"; done
export LTE_ARRIVAL_PROFILE=comprehensive
export LTE_SITE_COMPOSITION=0
export LTE_WEEKEND=0                 # weekday-only diurnal shape; no weekday/weekend contrast
export LTE_DETERMINISTIC_DAYS=0
export LTE_SCALE_PROFILE=large
export LTE_SERVICE_MIX=balanced
export LTE_STMSI_REALLOC=0
export LTE_DRX_MISS=0.05
export LTE_RELEASE_FN=0.05
export LTE_RELEASE_FP=0.02
export LTE_RELEASE_DELAY=1.5
export LTE_DWELL_SCALE=1.0
export LTE_BG_MIX=balanced
export LTE_N_RESIDENTS=30
export LTE_TOTAL_DAYS=5
export LTE_WARMUP_DAYS=1
# unset -> the comprehensive-profile defaults; an empty string is treated as unset
export LTE_DWELL_MIX=""
export LTE_USAGE_MIX=""
LTE_ARRIVAL_SCALE=$("$PY" -c "import json;print(json.load(open('$CAL'))['LTE_ARRIVAL_SCALE'])")
LTE_ACCESS_SCALE=$("$PY" -c "import json;print(json.load(open('$CAL'))['LTE_ACCESS_SCALE'])")
export LTE_ARRIVAL_SCALE LTE_ACCESS_SCALE

export TUNE_HORIZON=432000           # 5 days
export FINAL_HORIZON=432000          # main and robustness share one horizon (5 days)
export COMPARISON=controlled
export SELECTION=val
export XGB_FORCE_CPU=1
export LTE_XGB_JOBS=${LTE_XGB_JOBS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p "$OUT" "$LOGDIR"

cat <<PLAN
[5d] FINAL 5-DAY CONTROLLED PROTOCOL
  DGP        comprehensive | site_comp=0 | weekend=0 | day_var=on | large (~320)
             balanced mix | no S-TMSI realloc | 5 recorded days + 1 warm-up day
             ARRIVAL_SCALE=$LTE_ARRIVAL_SCALE  ACCESS_SCALE=$LTE_ACCESS_SCALE
  Horizons   tune=$TUNE_HORIZON  final=$FINAL_HORIZON   (both 432,000 slots)
  Splits     tuning 60/20/20 = 3d/1d/1d ; final train_ratio=0.8 = 4d/1d
  Protocol   comparison=$COMPARISON  selection=$SELECTION  -> A and B SHARE hyperparams
  Stage 1    $PREFIX       six branches, ${TRIALS_TOTAL} trials each, dev seeds: $DEV_SEEDS
  Stage 2    $CAL_PREFIX   hybrids only, isotonic-calibrated bases, base=$PREFIX
  Stage 3    MAIN  variant=C_full_cal  seeds: $MAIN_SEEDS  (one shard per seed)
  Workers    XGB on '$GPUS_XGB'   LSTM on '$GPUS_LSTM'   (LSTM capped for host RAM)
  Stages to run: $STAGES     tuning families: $TUNE_FAMILIES
  Preserved  journals and outputs of every other study prefix are left untouched
PLAN

if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "[5d] DRY-RUN ONLY. Set CONFIRM=1 to launch."
  exit 0
fi

has_stage () { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- Stage 1: base tuning
if has_stage 1; then
  echo "[5d] Stage 1/3 — base tuning ($PREFIX, controlled, 5-day), families: $TUNE_FAMILIES"
  # Split by family so the LSTM branches get the reduced worker count. The two calls write
  # DISJOINT studies into one journal, so RESUME=1 on the second never appends to a
  # finished study — it only permits reopening the journal file.
  if has_fam xgb; then
    SKIP_FINAL=1 PREFIX="$PREFIX" GPU_LIST="$GPUS_XGB" \
      RESUME="${RESUME_TUNE:-0}" \
      BRANCHES="cell_xgb perue_xgb hybrid_xgb" \
      TRIALS_TOTAL="$TRIALS_TOTAL" SAMPLER_SEED="$SAMPLER_SEED" \
      DEV_SEEDS="$DEV_SEEDS" OUT="$OUT" \
      bash run_tuning_parallel.sh
  fi

  if has_fam lstm; then
    # NOTE: run_tuning_parallel.sh gives every branch ceil(TRIALS_TOTAL/workers) per worker
    # and APPENDS to whatever the study already holds, so resuming a partially completed
    # LSTM study overshoots its per-branch budget. Top such a study up to exactly
    # TRIALS_TOTAL first, then run the untouched branches here.
    SKIP_FINAL=1 PREFIX="$PREFIX" GPU_LIST="$GPUS_LSTM" RESUME=1 \
      BRANCHES="${LSTM_BRANCHES:-cell_lstm perue_lstm hybrid_lstm}" \
      TRIALS_TOTAL="$TRIALS_TOTAL" SAMPLER_SEED="$SAMPLER_SEED" \
      DEV_SEEDS="$DEV_SEEDS" OUT="$OUT" \
      bash run_tuning_parallel.sh
  fi

  echo "[5d] Stage 1 shared-A/B params present:"
  ls -1 "$OUT"/best_${PREFIX}_*_AB.json || { echo "[5d] NO _AB.json — comparison was not 'controlled'."; exit 1; }
fi

# ---------------------------------------------------------------- Stage 2: calibrated hybrids
if has_stage 2; then
  echo "[5d] Stage 2/3 — hybrid re-tune on calibrated bases ($CAL_PREFIX), families: $TUNE_FAMILIES"
  # Read the base prefix BEFORE the command prefixes below shadow it. bash applies
  # assignments in a command prefix left to right, so in
  #     PREFIX="$CAL_PREFIX" BASE_PREFIX="$PREFIX" cmd
  # the $PREFIX on the second assignment already expands to CAL_PREFIX. The hybrid re-tune
  # would then take its own study as its base and never find the cell/perue params.
  BASE_FOR_CAL="$PREFIX"
  if has_fam xgb; then
    # RESUME is passed only when the caller knows the calibrated-prefix journal already
    # exists. Without it, run_tuning_parallel.sh refuses to reopen an existing journal;
    # that refusal is the guard against accidentally doubling a branch's trial budget.
    SKIP_FINAL=1 PREFIX="$CAL_PREFIX" BASE_PREFIX="$BASE_FOR_CAL" GPU_LIST="$GPUS_XGB" \
      RESUME="${RESUME_TUNE:-0}" \
      BRANCHES="hybrid_xgb" PERUE_CALIB=isotonic \
      FINAL_VARIANT=C_full_cal CALIB_KIND=isotonic \
      TRIALS_TOTAL="$TRIALS_TOTAL" SAMPLER_SEED="$SAMPLER_SEED" \
      DEV_SEEDS="$DEV_SEEDS" OUT="$OUT" \
      bash run_tuning_parallel.sh
  fi

  if has_fam lstm; then
    SKIP_FINAL=1 PREFIX="$CAL_PREFIX" BASE_PREFIX="$BASE_FOR_CAL" GPU_LIST="$GPUS_LSTM" RESUME=1 \
      BRANCHES="hybrid_lstm" PERUE_CALIB=isotonic \
      FINAL_VARIANT=C_full_cal CALIB_KIND=isotonic \
      TRIALS_TOTAL="$TRIALS_TOTAL" SAMPLER_SEED="$SAMPLER_SEED" \
      DEV_SEEDS="$DEV_SEEDS" OUT="$OUT" \
      bash run_tuning_parallel.sh
  fi

  echo "[5d] Stage 2 calibrated hybrid params:"
  ls -1 "$OUT"/best_${CAL_PREFIX}_hybrid_*_AB.json
  echo "[5d] FREEZE these. Every robustness scenario reuses them unchanged."
fi

# ---------------------------------------------------------------- Stage 3: MAIN, sharded
if has_stage 3; then
  # One FINAL process holds a full 2-mode dataset plus six branches of training state:
  # 139 GB measured at 864k slots, so roughly 70 GB at 432k. Two concurrent invocations of
  # this script would not fit, however small each seed shard is — hence the lock.
  LOCK=${LOCK:-.paper5d_final.lock}
  exec 9>"$LOCK"
  flock -n 9 || { echo "[5d] another final eval holds $LOCK — refusing. (pkill -f tune_runner)"; exit 1; }

  # Main reports all six models, so every branch must exist. Without this the run would
  # burn ~30 h and then die inside the first shard on a missing JSON.
  miss=""
  for br in cell_xgb perue_xgb cell_lstm perue_lstm; do
    [ -f "$OUT/best_${PREFIX}_${br}_AB.json" ] || miss="$miss $PREFIX/$br"
  done
  for br in hybrid_xgb hybrid_lstm; do
    [ -f "$OUT/best_${CAL_PREFIX}_${br}_AB.json" ] || miss="$miss $CAL_PREFIX/$br"
  done
  [ -z "$miss" ] || { echo "[5d] Stage 3 needs all six branches; missing:$miss"; exit 1; }

  echo "[5d] Stage 3/3 — MAIN evaluation, C_full_cal, one shard per seed"
  todo=""
  for s in $MAIN_SEEDS; do
    if [[ -f "$OUT/final_test_summary.shard_${s}.json" && "${FORCE:-0}" != "1" ]]; then
      echo "     seed $s: shard exists, skip"
    else
      todo="$todo $s"
    fi
  done
  echo "[5d] to run:${todo:- (none)}"

  read -r -a MG <<< "$MAIN_GPUS"
  run_shard () {                                  # $1=seed  $2=slot
    local sd="$1" gpu="${MG[$(( $2 % ${#MG[@]} ))]}"
    if CUDA_VISIBLE_DEVICES="$gpu" LTE_GPU_INDEX=0 \
        /usr/bin/time -v "$PY" -m lte_occupancy.experiments.tune_runner --final \
          --final-variant C_full_cal --calib-kind isotonic \
          --test-seeds "$sd" --horizon "$FINAL_HORIZON" --modes A B \
          --comparison "$COMPARISON" --selection "$SELECTION" \
          --study-prefix "$CAL_PREFIX" --base-study-prefix "$PREFIX" \
          --out "$OUT" --out-tag "$sd" \
          --sampler-seed "$SAMPLER_SEED" --worker-index 0 --workers 1 \
          > "$LOGDIR/final_seed_${sd}.log" 2>&1
    then
      grep -E "Elapsed \(wall|Maximum resident" "$LOGDIR/final_seed_${sd}.log" || true
      echo "[5d] seed $sd done [gpu $gpu] -> final_test_summary.shard_${sd}.json"
    else
      echo "[5d] !! seed $sd FAILED [gpu $gpu] — see $LOGDIR/final_seed_${sd}.log" >&2
      : > "$FAILDIR/$sd"
    fi
  }

  FAILDIR="$(mktemp -d)"; trap 'rm -rf "$FAILDIR"' EXIT
  fifo="$(mktemp -u)"; mkfifo "$fifo"; exec 8<>"$fifo"; rm -f "$fifo"
  for ((i = 0; i < MAIN_JOBS; i++)); do echo "$i" >&8; done
  deferred=0
  echo "[5d] $MAIN_JOBS concurrent shard(s) on GPUs '$MAIN_GPUS'; $(hours_left) until deadline"
  for s in $todo; do
    if ! time_left_for "$SHARD_HOURS"; then deferred=$((deferred + 1)); continue; fi
    read -r -u 8 slot
    echo "[5d] -> seed $s (slot $slot, $(hours_left) left)"
    { run_shard "$s" "$slot"; echo "$slot" >&8; } &
  done
  wait
  exec 8>&-

  nf=$(find "$FAILDIR" -type f | wc -l)
  [ "$nf" -eq 0 ] || { echo "[5d] $nf shard(s) failed; NOT merging a partial Main table."; exit 1; }
  if [ "$deferred" -gt 0 ]; then
    echo "[5d] DEADLINE: $deferred seed(s) not started. Merging what exists would record a"
    echo "     test_seeds set smaller than the pre-registered ten, so NOT merging."
    echo "     After the outage re-run the identical command; finished shards are skipped."
    exit 0
  fi

  n_shard=$(ls -1 "$OUT"/final_test_summary.shard_*.json 2>/dev/null | wc -l)
  [[ "$n_shard" != "0" ]] || { echo "[5d] no shards produced — see $LOGDIR/final_seed_*.log; NOT merging."; exit 1; }

  # tune_runner._report rebuilds the canonical meta from args.test_seeds and args.horizon,
  # and merge_shards reuses that same code path. Omitting them does NOT merely leave the
  # fields blank: argparse defaults win, so the merged file would claim
  #     test_seeds = [101..105]      (the argparse default)
  #     horizon    = null
  # regardless of what actually ran. The numbers would be right and the provenance wrong,
  # which is the worst combination. Derive the seed set from the shards ON DISK rather than
  # from $MAIN_SEEDS, so a merge after a partial run describes what it really merged.
  MERGED_SEEDS=$(ls -1 "$OUT"/final_test_summary.shard_*.json \
                 | sed -E 's/.*shard_([0-9]+)\.json/\1/' | sort -n | tr '\n' ' ')
  echo "[5d] merging $n_shard shard(s): $MERGED_SEEDS"
  if [ "$(echo $MERGED_SEEDS)" != "$(echo $MAIN_SEEDS | tr ' ' '\n' | sort -n | tr '\n' ' ' | sed 's/ $//')" ]; then
    echo "[5d] NOTE: merged seed set differs from MAIN_SEEDS='$MAIN_SEEDS'."
    echo "     The merged summary will record what is actually on disk."
  fi
  # shellcheck disable=SC2086
  "$PY" -m lte_occupancy.experiments.tune_runner --final --merge-shards \
    --final-variant C_full_cal --calib-kind isotonic \
    --study-prefix "$CAL_PREFIX" --base-study-prefix "$PREFIX" --out "$OUT" \
    --comparison "$COMPARISON" --selection "$SELECTION" \
    --test-seeds $MERGED_SEEDS --horizon "$FINAL_HORIZON" --modes A B \
    2>&1 | tee "$LOGDIR/final_merge.log"

  # Verify the metadata actually landed, rather than trusting that it did.
  "$PY" - "$OUT/final_test_summary.json" "$FINAL_HORIZON" "$MERGED_SEEDS" <<'CHK'
import json, sys
meta = json.load(open(sys.argv[1]))["meta"]
want_h, want_s = int(sys.argv[2]), [int(x) for x in sys.argv[3].split()]
bad = []
if meta.get("horizon") != want_h:
    bad.append(f"horizon={meta.get('horizon')!r} (expected {want_h})")
if sorted(meta.get("test_seeds") or []) != want_s:
    bad.append(f"test_seeds={meta.get('test_seeds')!r} (expected {want_s})")
if bad:
    raise SystemExit("[5d] MERGE METADATA WRONG: " + "; ".join(bad))
print(f"[5d] merge metadata OK: horizon={meta['horizon']} "
      f"test_seeds={meta['test_seeds']} n={len(meta['test_seeds'])}")
CHK
  echo "[5d] MAIN table source: $OUT/final_test_summary.json"
fi

cat <<'NEXT'

[5d] DONE. Next:
  1. Do NOT change hyperparameters now that the Main numbers are visible, and do NOT add
     seeds in response to an interval that straddles zero — the seed count was fixed in
     advance for exactly that reason. An interval containing zero is a reportable result.
  2. Freeze the params and run the robustness suite:
       PARAMS_DIR=tune_out_s3c5 PREFIX=s3c5_cal BASE_PREFIX=s3c5 \
       CONFIRM=1 nohup bash run_robustness5d_controlled_seeds.sh > robust5d.out 2>&1 &
NEXT
