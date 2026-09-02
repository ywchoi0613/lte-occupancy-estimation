#!/usr/bin/env bash
# run_robustness5d_controlled_seeds.sh — the full 15-axis robustness suite on the frozen
# 5-day controlled protocol. This produces the robustness results reported in the paper
# (Tables S6-S14 and Figures 5-8).
#
# Unit of work = (scenario, seed). Hyperparameters are FROZEN from the tuning studies; each
# unit re-trains weights on its own scenario's training split. No per-scenario Optuna.
#
# ---------------------------------------------------------------------------------------
# THREE DESIGN CHOICES THAT DIFFER FROM THE MAIN EVALUATION, ALL DELIBERATE
# ---------------------------------------------------------------------------------------
# 1. FAMILIES defaults to "xgb", not both families.
#    Cost: one all-six unit at 432,000 slots takes ~6 h and peaks near 70 GB, which caps
#    a 188 GB host at TWO concurrent units, making the suite a ~3-week job. Restricting
#    robustness to the XGB family cuts it to roughly five days.
#    Why the XGB family is the one kept: in the primary experiments the LSTM family showed
#    substantially higher cross-seed variability, on the order of the scenario and mode
#    effects this suite is trying to resolve, so at five seeds its cells would be dominated
#    by that variability. The MAIN table still reports all six models.
#    Note precisely what that evidence is: the spread is ACROSS TEST SEEDS, which varies
#    the simulator trajectory as well as the training RNG. It is not an isolated
#    measurement of run-to-run training variance, and the paper does not describe it as
#    one; that would need an experiment holding the trace and hyperparameters fixed and
#    varying only the training seed.
#    To override: FAMILIES="xgb lstm" (and budget roughly three weeks).
#
# 2. Concurrency is set by HOST RAM, not by GPU count.
#    An earlier 30,000-slot suite ran four units per GPU token. At 432,000 slots that is
#    14.4x the data and exhausts a 188 GB host. JOBS_AB/JOBS_B below are derived from the
#    measured per-unit footprint; raise them only after watching `free -g` through a full
#    unit on your own hardware.
#
# 3. An extra Mode-B-only anchor (`main_bonly`) is run.
#    The R7 noise sweeps are Mode-B-only, so their baseline must also be a Mode-B-only run
#    under an identical protocol. Reusing the Mode B column of the A+B anchor would assume
#    that adding Mode A leaves Mode B bit-identical, which is untested. Five extra units
#    buys a baseline that is comparable by construction.
#
# USAGE
#   bash run_robustness5d_controlled_seeds.sh                       # dry-run plan
#   PARAMS_DIR=tune_out_s3c5 PREFIX=s3c5_cal BASE_PREFIX=s3c5 \
#     CONFIRM=1 nohup bash run_robustness5d_controlled_seeds.sh > robust5d.out 2>&1 &
#
#   AXES="scale mix"          only those axis groups
#   SCENARIOS="scale_tiny"    only those scenario names
#   SEEDS="101 102 103"       override the seed pool
#   RESUME=0                  re-run units that already have results.csv (default: skip)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export XGB_FORCE_CPU=1
export LTE_XGB_JOBS="${LTE_XGB_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PY=${PY:-python}
PARAMS_DIR="${PARAMS_DIR:-tune_out_s3c5}"
PREFIX="${PREFIX:-s3c5_cal}"
BASE_PREFIX="${BASE_PREFIX:-s3c5}"
COMPARISON="${COMPARISON:-controlled}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
PERUE_CALIB="${PERUE_CALIB:-isotonic}"
CALIB_SCOPE="${CALIB_SCOPE:-full}"
FAMILIES="${FAMILIES:-xgb}"
SEEDS="${SEEDS:-101 102 103 104 105}"
HORIZON="${HORIZON:-432000}"                  # 5 days; main and robustness share one horizon
OUT_PREFIX="${OUT_PREFIX:-robust_${PREFIX}}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
LOGDIR="${LOGDIR:-logs_robust_${PREFIX}_5d}"
RESUME="${RESUME:-1}"

# --- Scheduled-outage support -----------------------------------------------------------
# DEADLINE: stop LAUNCHING new units once fewer than UNIT_HOURS_* remain, then let the
# in-flight ones finish and exit cleanly. A unit killed mid-write by a power cut leaves a
# partial output directory, and runner.py REFUSES to write into a non-empty directory
# (runner.py:198), so that unit would block its own retry. Finishing cleanly avoids the
# whole class of problem; the stale-directory cleanup below is the second line of defence.
DEADLINE="${DEADLINE:-}"                    # e.g. "2026-08-17 06:00" — empty = no limit
UNIT_HOURS_AB="${UNIT_HOURS_AB:-4}"         # worst-case A+B unit, used as the launch reserve
UNIT_HOURS_B="${UNIT_HOURS_B:-2}"           # worst-case B-only unit
CALIBRATE="${CALIBRATE:-0}"                 # run ONE unit alone under /usr/bin/time -v

if [ -n "$DEADLINE" ]; then
  DEADLINE_EPOCH=$(date -d "$DEADLINE" +%s 2>/dev/null) || {
    echo "cannot parse DEADLINE='$DEADLINE' (try \"2026-08-17 06:00\")" >&2; exit 1; }
else
  DEADLINE_EPOCH=""
fi

# true when a unit needing $1 hours can still finish before the deadline
time_left_for () {
  [ -z "$DEADLINE_EPOCH" ] && return 0
  local need_s=$(( ${1%%.*} * 3600 ))
  [ $(( DEADLINE_EPOCH - $(date +%s) )) -ge "$need_s" ]
}
hours_left () {
  [ -z "$DEADLINE_EPOCH" ] && { echo "unbounded"; return; }
  awk -v d="$DEADLINE_EPOCH" -v n="$(date +%s)" 'BEGIN{printf "%.1f h", (d-n)/3600}'
}

# Concurrency, from the measured per-unit footprint at 432,000 slots on a 188 GB host:
#   all six branches, A+B  ~70 GB  -> 2
#   xgb family,       A+B  ~40 GB  -> 3
#   xgb family,       B    ~20 GB  -> 4
if [[ "$FAMILIES" == "xgb" ]]; then
  JOBS_AB="${JOBS_AB:-3}"; JOBS_B="${JOBS_B:-4}"
else
  JOBS_AB="${JOBS_AB:-2}"; JOBS_B="${JOBS_B:-3}"
fi

read -r -a GPUS <<< "$GPU_LIST"
read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p "$LOGDIR"

# =======================================================================================
# Scenario registry:  name | axis | weight | modes | timer | unset-vars | env-overrides
#
#   weight  cost proxy (mean occupancy) used ONLY to order the queue longest-first
#   timer   RRC inactivity timer in seconds -> runner's --timer (NOT an LTE_ env var)
#   unset   vars that must be REMOVED, not reset. LTE_SERVICE_MIX has no valid empty
#           value, so a T5b site scenario has to unset it for the site's own service mix
#           to apply; resetting it to `balanced` would silently break T5b.
# =======================================================================================
ALL=(
  # ---- anchor (reused as the Large / Balanced / no-realloc / 10 s cell of every table)
  "main|anchor|322|A B|10|-|-"
  "main_bonly|anchor|322|B|10|-|-"

  # ---- R1 / T5a: arrival temporal shape ONLY (site composition stays comprehensive)
  "arr_resident|arrival|322|A B|10|-|LTE_ARRIVAL_PROFILE=resident"
  "arr_office|arrival|322|A B|10|-|LTE_ARRIVAL_PROFILE=office"
  "arr_transport|arrival|322|A B|10|-|LTE_ARRIVAL_PROFILE=transport"

  # ---- R2 / T5b: joint site profile (arrival + dwell + usage + service move together)
  #      LTE_WEEKEND MUST be forced to 0: build_config defaults weekend to ON whenever
  #      LTE_SITE_COMPOSITION=1, which would add a second uncontrolled axis.
  #      LTE_SERVICE_MIX/DWELL_MIX/USAGE_MIX are unset so the site drives them.
  "site_resident|site|322|A B|10|LTE_SERVICE_MIX|LTE_ARRIVAL_PROFILE=resident LTE_SITE_COMPOSITION=1 LTE_WEEKEND=0"
  "site_office|site|322|A B|10|LTE_SERVICE_MIX|LTE_ARRIVAL_PROFILE=office LTE_SITE_COMPOSITION=1 LTE_WEEKEND=0"
  "site_transport|site|322|A B|10|LTE_SERVICE_MIX|LTE_ARRIVAL_PROFILE=transport LTE_SITE_COMPOSITION=1 LTE_WEEKEND=0"

  # ---- R3: network scale (large == anchor)
  #      LTE_N_RESIDENTS MUST be unset here. The canonical reset pins it to 30, which is
  #      the LARGE profile's value, and build_config reads the env in preference to the
  #      profile — so a reset-and-override would run `tiny` with 30 of its 45 UEs resident
  #      (md sec.7 R3 asks for 4) and set arrival_base = (45-30)/E[D]. Silent, and it would
  #      have turned the small end of the scale sweep into a nearly static cell.
  "scale_tiny|scale|45|A B|10|LTE_N_RESIDENTS|LTE_SCALE_PROFILE=tiny"
  "scale_small|scale|85|A B|10|LTE_N_RESIDENTS|LTE_SCALE_PROFILE=small"
  "scale_medium|scale|162|A B|10|LTE_N_RESIDENTS|LTE_SCALE_PROFILE=medium"
  "scale_xlarge|scale|642|A B|10|LTE_N_RESIDENTS|LTE_SCALE_PROFILE=xlarge"

  # ---- R4: service mix (balanced == anchor)
  "mix_voice_heavy|mix|321|A B|10|-|LTE_SERVICE_MIX=voice_heavy"
  "mix_streaming_heavy|mix|321|A B|10|-|LTE_SERVICE_MIX=streaming_heavy"
  "mix_browsing_heavy|mix|321|A B|10|-|LTE_SERVICE_MIX=browsing_heavy"

  # ---- R5: RRC inactivity timer (10 s == anchor); passed as --timer, not env
  "timer_5|timer|322|A B|5|-|-"
  "timer_15|timer|322|A B|15|-|-"
  "timer_30|timer|322|A B|30|-|-"

  # ---- R6: S-TMSI reallocation (off == anchor). Cell rows must come out ~identical;
  #      truth_hashes_seed*.json is what proves the physical trajectory did not move.
  "stmsi_on|stmsi|322|A B|10|-|LTE_STMSI_REALLOC=1"

  # ---- R7: Mode B observation noise, one factor at a time (baseline == main_bonly)
  "noise_drx_000|noise|322|B|10|-|LTE_DRX_MISS=0.00"
  "noise_drx_010|noise|322|B|10|-|LTE_DRX_MISS=0.10"
  "noise_drx_020|noise|322|B|10|-|LTE_DRX_MISS=0.20"
  "noise_relfn_000|noise|322|B|10|-|LTE_RELEASE_FN=0.00"
  "noise_relfn_010|noise|322|B|10|-|LTE_RELEASE_FN=0.10"
  "noise_relfn_020|noise|322|B|10|-|LTE_RELEASE_FN=0.20"
  "noise_relfp_0000|noise|322|B|10|-|LTE_RELEASE_FP=0.000"
  "noise_relfp_0005|noise|322|B|10|-|LTE_RELEASE_FP=0.005"
  "noise_relfp_0010|noise|322|B|10|-|LTE_RELEASE_FP=0.010"
  "noise_reldelay_00|noise|322|B|10|-|LTE_RELEASE_DELAY=0.0"
  "noise_reldelay_30|noise|322|B|10|-|LTE_RELEASE_DELAY=3.0"
  "noise_reldelay_50|noise|322|B|10|-|LTE_RELEASE_DELAY=5.0"

  # ---- R8: dwell-time scale. Little's-law arrival_base absorbs E[D], so occupancy stays
  #      near 320 and this measures residence DYNAMICS, not cell size.
  "dwell_scale_05|dwell|322|A B|10|-|LTE_DWELL_SCALE=0.5"
  "dwell_scale_20|dwell|322|A B|10|-|LTE_DWELL_SCALE=2.0"

  # ---- R9: dwell composition  (transient, regular, stationary)
  "dwellmix_transient_heavy|dwellmix|322|A B|10|-|LTE_DWELL_MIX=0.60,0.30,0.10"
  "dwellmix_stationary_heavy|dwellmix|322|A B|10|-|LTE_DWELL_MIX=0.10,0.30,0.60"
  "dwellmix_homogeneous|dwellmix|322|A B|10|-|LTE_DWELL_MIX=0.3333,0.3333,0.3334"

  # ---- R10: usage-intensity composition  (low, medium, high)
  "usagemix_low_heavy|usagemix|322|A B|10|-|LTE_USAGE_MIX=0.60,0.30,0.10"
  "usagemix_high_heavy|usagemix|322|A B|10|-|LTE_USAGE_MIX=0.15,0.25,0.60"
  "usagemix_homogeneous|usagemix|322|A B|10|-|LTE_USAGE_MIX=0.3333,0.3333,0.3334"

  # ---- R11: resident share at ~constant total occupancy
  "resident_15|resident|322|A B|10|-|LTE_N_RESIDENTS=15"
  "resident_60|resident|322|A B|10|-|LTE_N_RESIDENTS=60"

  # ---- R12: background reconnect mix (srsRAN idle-gap classes)
  "bg_chatty_heavy|bg|322|A B|10|-|LTE_BG_MIX=chatty_heavy"
  "bg_quiet_heavy|bg|322|A B|10|-|LTE_BG_MIX=quiet_heavy"
)

# ------------------------------------------------------------------ subset selection
pick=()
if [ -n "${SCENARIOS:-}" ]; then
  for want in $SCENARIOS; do
    hit=""
    for s in "${ALL[@]}"; do
      IFS='|' read -r n _ <<< "$s"
      [ "$n" = "$want" ] && { pick+=("$s"); hit=1; break; }
    done
    [ -n "$hit" ] || { echo "unknown scenario '$want'" >&2; exit 1; }
  done
elif [ -n "${AXES:-}" ]; then
  for s in "${ALL[@]}"; do
    IFS='|' read -r _ ax _ <<< "$s"
    for want in $AXES; do [ "$ax" = "$want" ] && { pick+=("$s"); break; }; done
  done
  [ "${#pick[@]}" -gt 0 ] || { echo "no scenario matched AXES='$AXES'" >&2; exit 1; }
else
  pick=("${ALL[@]}")
fi

# ------------------------------------------------------------------ pre-flight
# Collected rather than raised, so a dry-run still prints the full plan before the tuning
# stage has produced any params. Only CONFIRM=1 treats a problem as fatal.
problems=()
want_branches="cell_xgb perue_xgb hybrid_xgb"
[[ "$FAMILIES" == *lstm* ]] && want_branches="$want_branches cell_lstm perue_lstm hybrid_lstm"
for br in $want_branches; do
  case "$br" in hybrid_*) pre="$PREFIX";; *) pre="$BASE_PREFIX";; esac
  [ -f "$PARAMS_DIR/best_${pre}_${br}_AB.json" ] || \
    problems+=("missing $PARAMS_DIR/best_${pre}_${br}_AB.json  (no _AB file means that study was not tuned with comparison=controlled)")
done
# ------------------------------------------------------------------ build the queue
units_ab=(); units_b=(); skipped=0
for s in "${pick[@]}"; do
  IFS='|' read -r name axis weight modes timer unsetv envs <<< "$s"
  for seed in "${SEED_ARR[@]}"; do
    out="${OUT_PREFIX}_${name}__s${seed}"
    # The sentinel is written by the SHELL after the python process exits 0, so it cannot
    # exist for a unit that was interrupted. Testing results.csv instead would treat a
    # half-written file from a power cut as a finished unit and silently tabulate it.
    if [ "$RESUME" = "1" ] && [ -f "$out/UNIT_DONE" ]; then skipped=$((skipped+1)); continue; fi
    rec="${name}|${modes}|${timer}|${unsetv}|${envs}|${seed}|${weight}"
    if [ "$modes" = "B" ]; then units_b+=("$rec"); else units_ab+=("$rec"); fi
  done
done
# longest-first inside each class
sort_desc () { printf '%s\n' "$@" | awk -F'|' '{print $7"\t"$0}' | sort -rn -k1,1 | cut -f2-; }
[ "${#units_ab[@]}" -gt 0 ] && mapfile -t units_ab < <(sort_desc "${units_ab[@]}")
[ "${#units_b[@]}"  -gt 0 ] && mapfile -t units_b  < <(sort_desc "${units_b[@]}")

free_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo "?")
cat <<PLAN
=== 5-day CONTROLLED robustness suite
    scenarios     ${#pick[@]}   seeds '$SEEDS'   horizon $HORIZON (5 days)
    A+B units     ${#units_ab[@]}   (concurrency $JOBS_AB)
    B-only units  ${#units_b[@]}   (concurrency $JOBS_B)
    skipped       $skipped   (RESUME=$RESUME)
    families      $FAMILIES
    params        $PARAMS_DIR   study=$PREFIX base=$BASE_PREFIX comparison=$COMPARISON
    calibration   $PERUE_CALIB / $CALIB_SCOPE   (variant C_full_cal)
    gpus          '$GPU_LIST'    available RAM ${free_gb} GB
    out           ${OUT_PREFIX}_<scenario>__s<seed>/  -> merged into ${OUT_PREFIX}_<scenario>/
PLAN

if [ "${#problems[@]}" -gt 0 ]; then
  echo "--- PRE-FLIGHT PROBLEMS (${#problems[@]}):"
  printf '      %s\n' "${problems[@]}"
fi

if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "=== DRY-RUN ONLY. Set CONFIRM=1 to launch."
  exit 0
fi
[ "${#problems[@]}" -eq 0 ] || { echo "=== REFUSING to launch with pre-flight problems." >&2; exit 1; }

FAILDIR="$(mktemp -d)"; trap 'rm -rf "$FAILDIR"' EXIT

run_unit () {                       # $1=record  $2=slot
  local rec="$1" slot="$2"
  IFS='|' read -r name modes timer unsetv envs seed _w <<< "$rec"
  local out="${OUT_PREFIX}_${name}__s${seed}"
  local gpu="${GPUS[$(( slot % ${#GPUS[@]} ))]}"

  # No sentinel but a directory present => a previous attempt was interrupted. runner.py
  # refuses a non-empty --out, so the leftovers would make this unit permanently
  # unrunnable. Deleting is safe precisely because the sentinel is absent: nothing in
  # there was ever certified complete.
  if [ -d "$out" ] && [ ! -f "$out/UNIT_DONE" ]; then
    echo "     (clearing interrupted attempt: $out)"
    rm -rf "$out"
  fi

  # Every unit starts from the CANONICAL baseline on ALL FIFTEEN axes and then overrides
  # exactly its own. Without this a leaked LTE_STMSI_REALLOC=1 from the caller's shell
  # would quietly make every "single-factor" scenario a two-factor one.
  local reset=(
    LTE_ARRIVAL_PROFILE=comprehensive
    LTE_SITE_COMPOSITION=0
    LTE_WEEKEND=0
    LTE_DETERMINISTIC_DAYS=0
    LTE_SCALE_PROFILE=large
    LTE_SERVICE_MIX=balanced
    LTE_STMSI_REALLOC=0
    LTE_DRX_MISS=0.05
    LTE_RELEASE_FN=0.05
    LTE_RELEASE_FP=0.02
    LTE_RELEASE_DELAY=1.5
    LTE_DWELL_SCALE=1.0
    LTE_DWELL_MIX=
    LTE_USAGE_MIX=
    LTE_N_RESIDENTS=30
    LTE_BG_MIX=balanced
    LTE_TOTAL_DAYS=5
    LTE_WARMUP_DAYS=1
    LTE_ARRIVAL_SCALE="${LTE_ARRIVAL_SCALE:-1.0}"
    LTE_ACCESS_SCALE="${LTE_ACCESS_SCALE:-1.0}"
  )
  # `env -u VAR VAR=value` does NOT leave VAR unset: env applies its options first and the
  # assignment then puts the variable straight back. So the assignment has to be REMOVED
  # from the reset list, not merely preceded by -u. Getting this wrong is silent — T5b
  # would run every site under the baseline `balanced` service mix and still look fine.
  local unset_args=() keep=() kv var u drop
  if [ "$unsetv" != "-" ]; then
    for u in $unsetv; do unset_args+=(-u "$u"); done
  fi
  for kv in "${reset[@]}"; do
    var="${kv%%=*}"; drop=""
    if [ "$unsetv" != "-" ]; then
      for u in $unsetv; do [ "$u" = "$var" ] && { drop=1; break; }; done
    fi
    [ -n "$drop" ] || keep+=("$kv")
  done
  local extra=(); [ "$envs" != "-" ] && read -r -a extra <<< "$envs"

  # NOT named `timer`: that variable already holds the RRC inactivity timer parsed out of
  # the scenario record a few lines above.
  local measure_cmd=()
  [ "${MEASURE:-0}" = "1" ] && measure_cmd=(/usr/bin/time -v -o "$LOGDIR/${name}__s${seed}.time")

  if env "${unset_args[@]}" "${keep[@]}" "${extra[@]}" \
      CUDA_VISIBLE_DEVICES="$gpu" LTE_GPU_INDEX=0 LTE_TOTAL_TIME="$HORIZON" \
      "${measure_cmd[@]}" "$PY" -u -m lte_occupancy.experiments.runner \
        --model-set tuned --params-dir "$PARAMS_DIR" --study-prefix "$PREFIX" \
        --base-study-prefix "$BASE_PREFIX" \
        --comparison "$COMPARISON" --train-ratio "$TRAIN_RATIO" \
        --perue-calib "$PERUE_CALIB" --calib-scope "$CALIB_SCOPE" \
        --families $FAMILIES --modes $modes --timer "$timer" \
        --seeds "$seed" --out "$out" \
        > "$LOGDIR/${name}__s${seed}.log" 2>&1
  then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$out/UNIT_DONE"
    echo "     done: ${name} seed ${seed}  [gpu ${gpu}]"
  else
    echo "!! FAILED: ${name} seed ${seed}  (see $LOGDIR/${name}__s${seed}.log)" >&2
    : > "$FAILDIR/${name}__s${seed}"
  fi
}

# A plain N-token FIFO per class. Classes run in sequence (A+B first: heavier and longer),
# so a single semaphore never has to serve two different memory footprints — which is what
# makes a weighted semaphore deadlock-prone in shell.
DEFERRED=0
run_phase () {                      # $1=label  $2=jobs  $3=reserve-hours  $4.. = records
  local label="$1" jobs="$2" reserve="$3"; shift 3
  [ "$#" -gt 0 ] || { echo "=== phase $label: nothing to do"; return 0; }
  echo "=== phase $label: $# unit(s), $jobs concurrent, $(hours_left) until deadline"
  local fifo; fifo="$(mktemp -u)"; mkfifo "$fifo"; exec 9<>"$fifo"; rm -f "$fifo"
  local i
  for ((i = 0; i < jobs; i++)); do echo "$i" >&9; done
  for rec in "$@"; do
    IFS='|' read -r n _m _t _u _e sd _w <<< "$rec"
    # Check BEFORE taking a slot: a unit started with too little time left is not merely
    # wasted, it leaves a partial directory that the next run has to clean up.
    if ! time_left_for "$reserve"; then
      DEFERRED=$((DEFERRED + 1))
      continue
    fi
    read -r -u 9 slot
    echo "  -> [$n seed $sd] slot $slot  ($(hours_left) left)"
    { run_unit "$rec" "$slot"; echo "$slot" >&9; } &
  done
  wait
  exec 9>&-
}

if [ "$CALIBRATE" = "1" ]; then
  # One A+B unit, alone, measured. Every concurrency number in this script is extrapolated
  # from a 864k-slot run; this replaces the extrapolation with the real figure for THIS box
  # and THIS scenario set. It is not a throwaway: the unit is a real member of the suite,
  # so the subsequent full run skips it via its sentinel.
  cal="${units_ab[0]:-${units_b[0]:-}}"
  [ -n "$cal" ] || { echo "nothing left to calibrate (all units already done)"; exit 0; }
  IFS='|' read -r cn _cm _ct _cu _ce cs _cw <<< "$cal"
  echo "=== CALIBRATION: one unit ($cn seed $cs) alone, under /usr/bin/time -v"
  MEASURE=1 run_unit "$cal" 0
  log="$LOGDIR/${cn}__s${cs}.time"
  if [ -f "$log" ]; then
    grep -E "Elapsed \(wall|Maximum resident" "$log"
    python3 - "$log" <<'CAL'
import re, sys
t = open(sys.argv[1]).read()
kb = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", t)
wl = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", t)
if kb:
    gb = int(kb.group(1)) / 1024 / 1024
    import subprocess
    free = int(subprocess.check_output(["free","-g"]).split()[12])
    jobs = max(1, int((free - 20) // gb))
    print(f"\n  measured peak RSS : {gb:.1f} GB per A+B unit")
    if wl: print(f"  measured wall     : {wl.group(1)}")
    print(f"  free RAM now      : {free} GB")
    print(f"  => JOBS_AB={jobs}  (leaves 20 GB headroom; JOBS_B can be ~{jobs*2})")
    print(f"     Also cap by free CPU cores / LTE_XGB_JOBS.")
CAL
  fi
  echo "=== calibration done. Re-run without CALIBRATE=1 to start the suite."
  exit 0
fi

run_phase "A+B" "$JOBS_AB" "$UNIT_HOURS_AB" ${units_ab+"${units_ab[@]}"}
run_phase "B-only" "$JOBS_B" "$UNIT_HOURS_B" ${units_b+"${units_b[@]}"}

nfail=$(find "$FAILDIR" -type f | wc -l)
[ "$nfail" -eq 0 ] || { echo "ABORT: $nfail unit(s) failed; nothing merged." >&2
                        echo "Fix them, then re-run with RESUME=1 (completed units are skipped)." >&2
                        exit 1; }

if [ "$DEFERRED" -gt 0 ]; then
  cat <<STOP

=== DEADLINE REACHED — stopped cleanly with $DEFERRED unit(s) not started.
    Every unit that DID run is complete and certified (UNIT_DONE). Nothing is merged yet,
    because merging a partial suite would produce tables with silently missing rows.
    After the outage, re-run the IDENTICAL command: RESUME=1 skips the finished units and
    picks up exactly where this stopped.
STOP
  exit 0
fi

# ------------------------------------------------------------------ merge
names=""
for s in "${pick[@]}"; do IFS='|' read -r n _ <<< "$s"; names="$names $n"; done
echo
echo "=== merging per-seed shards into ${OUT_PREFIX}_<scenario>/"
"$PY" merge_robust_shards.py --prefix "$OUT_PREFIX" --scenarios "$names" --seeds "$SEEDS"

cat <<NEXT

=== ALL DONE.
Build the robustness tables:
    python make_robust5d_tables.py --prefix ${OUT_PREFIX} \\
      --params-dir ${PARAMS_DIR} --seeds "$SEEDS" --out paper_tables_${PREFIX}

Correctness checks that the reported results rest on:
  * single-axis:  diff each scenario's resolved_config.json against ${OUT_PREFIX}_main/
                  and confirm exactly one field moved (site_* and the timer_* group are
                  the intended multi-field exceptions)
  * S-TMSI:       truth_hashes_seed*.json must match between _main and _stmsi_on for
                  physical_ue_events / ue_meta — only stmsi_events may differ
  * noise sweep:  the same truth hashes must hold across every noise_* scenario
NEXT
