#!/usr/bin/env bash
# run_tuning_parallel.sh — tuned protocol (Stages 1-4) across GPUs 0-3.
#
# SYMMETRIC six-model design (finalized):
#     Cell-XGB  + PerUE-XGB   -> Hybrid-XGB
#     Cell-LSTM + PerUE-LSTM  -> Hybrid-LSTM
#
# Optuna workers share ONE journal study, so 4 processes (one per GPU) add trials to the
# same study concurrently. XGB branches run on CPU (GPUs idle there); the LSTM branches
# (cell_lstm, perue_lstm, hybrid_lstm) use one GPU per worker. Hyperparameters are
# selected on VALIDATION only; TEST is touched once at the end (held-out seeds) with a
# paired A-B interval, and the final summary now reports up to all SIX models.
#
# Reproducibility: each study uses a seeded multivariate TPE sampler (SAMPLER_SEED). With
# async workers the global trial order still varies, so sampler seed / Optuna version /
# worker count / torch+device are written to run_meta_*.json.
#
# Run twice to get BOTH comparisons the reviewers asked for:
#   COMPARISON=equal_budget PREFIX=s3c5  bash run_tuning_parallel.sh   # per-mode tuned
#   COMPARISON=controlled   PREFIX=s3c5  bash run_tuning_parallel.sh   # identical A/B params
#
# Variant-C re-tune (only if the calibration ablation picks C_full_cal): re-tune ONLY the
# two hybrid branches on calibrated bases, reusing the four base studies as-is. The new
# hybrid studies are written under a new PREFIX into the SAME directory as the base params,
# so nothing is copied and the base params keep their identity:
#   BRANCHES="hybrid_xgb hybrid_lstm" PREFIX=s3c5_cal BASE_PREFIX=s3c5 \
#     PERUE_CALIB=isotonic FINAL_VARIANT=C_full_cal CALIB_KIND=isotonic \
#     bash run_tuning_parallel.sh
# (OUT defaults to tune_out_$BASE_PREFIX in that case. Use SKIP_FINAL=1 to tune only.)
set -euo pipefail

export XGB_FORCE_CPU=1                       # deterministic XGB (paper); GPUs used by LSTM
export LTE_SCALE_PROFILE="${LTE_SCALE_PROFILE:-large}"
export LTE_SERVICE_MIX="${LTE_SERVICE_MIX:-balanced}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LTE_XGB_JOBS="${LTE_XGB_JOBS:-4}"

COMPARISON="${COMPARISON:-equal_budget}"     # equal_budget | controlled
PREFIX="${PREFIX:-s3c5}"                      # prefix this run's studies are written under
BASE_PREFIX="${BASE_PREFIX:-}"               # read BASE params from this prefix instead
PERUE_CALIB="${PERUE_CALIB:-none}"           # none | isotonic | linear (hybrid input calib)
FINAL_VARIANT="${FINAL_VARIANT:-R_raw}"      # R_raw | S_perue_cal | C_full_cal
CALIB_KIND="${CALIB_KIND:-isotonic}"         # map used by S/C at final time
SELECTION="${SELECTION:-val}"                # val (headline) | test_oracle (diagnostic)
# best_*.json are addressed by PREFIX inside ONE directory, so a hybrid re-tune must write
# into the directory that already holds the base params: default OUT to the BASE_PREFIX's
# directory whenever a base prefix is given.
OUT="${OUT:-tune_out_${BASE_PREFIX:-$PREFIX}}"
STORAGE="${STORAGE:-journal:tune_${PREFIX}.journal}"
DEV_SEEDS="${DEV_SEEDS:-7 13 42}"
TEST_SEEDS="${TEST_SEEDS:-101 102 103 104 105 106 107 108 109 110}"
TUNE_HORIZON="${TUNE_HORIZON:-8000}"         # reduced horizon for search speed
FINAL_HORIZON="${FINAL_HORIZON:-30000}"      # full horizon for the reported test numbers
TRIALS_TOTAL="${TRIALS_TOTAL:-48}"           # per study, summed across the 4 workers
SAMPLER_SEED="${SAMPLER_SEED:-42}"
SKIP_FINAL="${SKIP_FINAL:-0}"                # 1 = tune only, run --final yourself later
GPUS=(${GPU_LIST:-0 1 2 3})            # override e.g. GPU_LIST="0 1 2" for memory-heavy branches
PER=$(( (TRIALS_TOTAL + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
NWORKERS=${#GPUS[@]}
LOGDIR="logs_tune_${PREFIX}"
mkdir -p "$LOGDIR" "$OUT"

tune () {                                    # $1 = branch
  local branch="$1"
  echo "=== [$branch] ${TRIALS_TOTAL} trials over ${NWORKERS} GPUs (val-selected) ==="
  local pids=() idx=0 gpu
  for gpu in "${GPUS[@]}"; do
    # worker-index (loop ordinal) offsets the sampler seed so the 4 workers explore with
    # DIFFERENT TPE proposals instead of duplicating the same initial trials.
    CUDA_VISIBLE_DEVICES="$gpu" LTE_GPU_INDEX=0 \
      python -m lte_occupancy.experiments.tune_runner \
        --branch "$branch" --trials "$PER" \
        --comparison "$COMPARISON" --selection "$SELECTION" --modes A B \
        --dev-seeds $DEV_SEEDS --horizon "$TUNE_HORIZON" \
        --storage "$STORAGE" --study-prefix "$PREFIX" --out "$OUT" \
        ${BASE_PREFIX:+--base-study-prefix "$BASE_PREFIX"} \
        --perue-calib "$PERUE_CALIB" \
        --sampler-seed "$SAMPLER_SEED" --worker-index "$idx" --workers "$NWORKERS" \
        > "$LOGDIR/${branch}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  local rc=0 i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "!! [$branch] worker on GPU ${GPUS[$i]} FAILED (see $LOGDIR/${branch}_gpu${GPUS[$i]}.log)" >&2
      rc=1
    fi
  done
  [ "$rc" -eq 0 ] || { echo "ABORT: $branch had a failed worker" >&2; exit 1; }
  echo "    [$branch] done"
}

# Guard against accidentally resuming an existing journal (would ADD trials on top of it,
# breaking the fair per-study budget). Use RESUME=1 to intentionally continue.
RESUME="${RESUME:-0}"
JOURNAL_PATH="${STORAGE#journal:}"
if [ "$RESUME" != "1" ] && [ "${STORAGE#journal:}" != "$STORAGE" ] && [ -e "$JOURNAL_PATH" ]; then
  echo "Journal '$JOURNAL_PATH' already exists. Use RESUME=1 to continue, or a new PREFIX." >&2
  exit 1
fi

# Branches to tune. Base branches MUST precede the hybrids that consume them, and each
# hybrid consumes its OWN family's base pair:
#     cell_xgb  perue_xgb  -> hybrid_xgb
#     cell_lstm perue_lstm -> hybrid_lstm
# Override BRANCHES to run a subset, e.g. the verified XGB-only path:
#     BRANCHES="cell_xgb perue_xgb hybrid_xgb"
BRANCHES="${BRANCHES:-cell_xgb perue_xgb cell_lstm perue_lstm hybrid_xgb hybrid_lstm}"
for br in $BRANCHES; do
  tune "$br"
done

# Stage 4 — final test on held-out seeds at FULL horizon, single process (uses best
# params). Reports every model that was tuned: XGB family always, LSTM family if present.
# --comparison / --selection are passed so the final run_meta records the correct
# provenance (otherwise a controlled run would be logged as equal_budget).
# --perue-calib is a TUNING-stage flag and is deliberately NOT passed here: at final time
# the estimator is chosen by --final-variant (+ --calib-kind for S/C).
if [ "$SKIP_FINAL" = "1" ]; then
  echo "=== SKIP_FINAL=1: tuning done; run --final yourself ==="
else
  echo "=== FINAL test eval (held-out seeds, horizon=$FINAL_HORIZON, variant=$FINAL_VARIANT) ==="
  LTE_GPU_INDEX=0 CUDA_VISIBLE_DEVICES=0 \
    python -m lte_occupancy.experiments.tune_runner --final \
      --final-variant "$FINAL_VARIANT" --calib-kind "$CALIB_KIND" \
      --test-seeds $TEST_SEEDS --horizon "$FINAL_HORIZON" --modes A B \
      --comparison "$COMPARISON" --selection "$SELECTION" \
      --study-prefix "$PREFIX" --out "$OUT" \
      ${BASE_PREFIX:+--base-study-prefix "$BASE_PREFIX"} \
      --sampler-seed "$SAMPLER_SEED" --worker-index 0 --workers "$NWORKERS" | tee "$LOGDIR/final.log"
fi

echo "=== ALL DONE. best params + final_test_summary.json + run_meta_*.json in $OUT/ ==="