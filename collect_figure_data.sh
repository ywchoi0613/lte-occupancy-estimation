#!/usr/bin/env bash
# collect_figure_data.sh — gather exactly the files the figure notebook reads, as flat
# files in one small zip.
#
# The paper bundle is the archive; this is not. It carries only what the plots need, so it
# stays small enough to upload to Colab by hand (~1-2 MB against the bundle's 668 KB plus
# the 21 GB of predictions the bundle deliberately excludes).
#
# Two of the five files cannot be taken from the bundle at all: they are derived from
# preds_seed*.npz, which lives only on this machine. They are regenerated here rather than
# copied, so the numbers in the figures and the numbers in the text come from one pass over
# the same arrays.
#
#   robust5d_long.csv        scale / noise / robustness figures
#   final_test_summary.json  granularity figure (Table 2)
#   stmsi_bias.csv           S-TMSI panel (b), signed error        [regenerated]
#   acf_curves.npy           supplemental period figure            [regenerated]
#   motivation_day.npz       latent vs observable, one test day    [regenerated]
#
#   bash collect_figure_data.sh          # dry run: what would be collected
#   CONFIRM=1 bash collect_figure_data.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
OUT=${OUT:-figure_data}
ZIP=${ZIP:-figure_data.zip}
TABLES=${TABLES:-paper_tables_s3c5_cal}
TUNE=${TUNE:-tune_out_s3c5}
ROBUST=${ROBUST:-robust_s3c5_cal}
SEED=${SEED:-101}                 # seed used for the one-day motivation trace

echo "=== checking inputs"
ok=1
for p in "$TABLES/robust5d_long.csv" "$TUNE/final_test_summary.json" \
         "${ROBUST}_main/preds_seed${SEED}.npz" "${ROBUST}_stmsi_on/preds_seed${SEED}.npz"; do
  if [ -e "$p" ]; then printf '  OK   %s\n' "$p"
  else printf '  MISSING %s\n' "$p"; ok=0; fi
done
n_arr=$(ls -1 ${ROBUST}_arr_*/preds_seed*.npz 2>/dev/null | wc -l | tr -d ' ')
printf '  arrival-profile preds: %s (for the period-recovery figure)\n' "$n_arr"

if [ "${CONFIRM:-0}" != "1" ]; then
  echo
  echo "=== DRY RUN. To actually build it: CONFIRM=1 bash collect_figure_data.sh"
  [ "$ok" = 0 ] && echo "    (some inputs are missing)"
  exit 0
fi
[ "$ok" = 1 ] || { echo "required inputs are missing; aborting."; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$TABLES/robust5d_long.csv" "$TUNE/final_test_summary.json" "$OUT/"

"$PY" - "$OUT" "$ROBUST" "$SEED" <<'PYEOF'
import glob, sys
import numpy as np

out, robust, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])

# ---- S-TMSI signed error, test slice only ------------------------------------------
# Hybrid predictions exist on the test split only (NaN before split_idx), and the split
# is where the reported metrics are computed; averaging the whole trace silently mixes
# training-window values in and does not reproduce the tables.
# Per-seed spread is carried too: the figure draws error bars, and a mean with no
# dispersion invites the reader to treat a 0.5-UE difference as resolved.
rows = ["model,condition,bias,bias_std,mae,mae_std,rmse,n_seeds"]
for scen, lab in (("main", "off"), ("stmsi_on", "on")):
    for m in ("Cell_XGB_B", "PerUE_XGB_B", "Hybrid_XGB_B"):
        b, a, r = [], [], []
        for f in sorted(glob.glob(f"{robust}_{scen}/preds_seed*.npz")):
            z = np.load(f); k = int(z["split_idx"])
            p, y = z[m][k:], z["y"][k:]
            ok = ~np.isnan(p); e = p[ok] - y[ok]
            b.append(e.mean()); a.append(np.abs(e).mean()); r.append(np.sqrt((e ** 2).mean()))
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(f"{m},{lab},{np.mean(b):.4f},{sd(b):.4f},"
                    f"{np.mean(a):.4f},{sd(a):.4f},{np.mean(r):.4f},{len(b)}")
open(f"{out}/stmsi_bias.csv", "w").write("\n".join(rows) + "\n")
print(f"  stmsi_bias.csv        {len(rows)-1} rows")

# ---- diurnal-period autocorrelation, training window only ---------------------------
# Training window only: the claim is that a passive observer recovers the period from what
# it has already seen, so touching the test split would make the demonstration circular.
cur, lines = {}, []
for scen in ("main", "arr_office", "arr_resident", "arr_transport"):
    for f in sorted(glob.glob(f"{robust}_{scen}/preds_seed*.npz")):
        z = np.load(f); k = int(z["split_idx"])
        x = z["n_connected_b"][:k].astype(float); x -= x.mean(); N = len(x)
        n = 1 << int(np.ceil(np.log2(2 * N)))
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2)[:N]; ac /= ac[0]
        p = 21600 + int(np.argmax(ac[21600:min(172800, N - 1)]))
        F = np.abs(np.fft.rfft(x))
        s = f.split("seed")[1][:3]
        # Two different quantities, and conflating them produced a wrong caption once:
        #   acf_*     autocorrelation of the series at that lag
        #   fftamp_*  |F| at that period, divided by the largest |F| in the spectrum
        # The ACF is what identifies the period; the spectrum is what shows the commute
        # harmonic. Both are recorded so a caption can name the right one.
        lines.append(f"{scen},{s},{p},{p/3600:.3f},"
                     f"{ac[43200]:.4f},{ac[86400]:.4f},"
                     f"{F[N//86400]/F.max():.4f},{F[N//43200]/F.max():.4f}")
        cur[f"{scen}_s{s}"] = ac[:172800:60].astype(np.float32)   # 1-min steps, 48 h
np.save(f"{out}/acf_curves.npy", cur, allow_pickle=True)
open(f"{out}/period_recovery.csv", "w").write(
    "scenario,seed,acf_peak_slots,acf_peak_hours,acf_at_12h,acf_at_24h,"
    "fftamp_24h_rel,fftamp_12h_rel\n" + "\n".join(lines) + "\n")
print(f"  acf_curves.npy        {len(cur)} curves")

# ---- one test day of the canonical trace -------------------------------------------
z = np.load(f"{robust}_main/preds_seed{seed}.npz")
k = int(z["split_idx"]); sl = slice(k, k + 86400)
np.savez_compressed(f"{out}/motivation_day.npz",
                    y=z["y"][sl].astype("float32"),
                    n_connected=z["n_connected"][sl].astype("float32"),
                    n_connected_b=z["n_connected_b"][sl].astype("float32"),
                    hybrid_b=z["Hybrid_XGB_B"][sl].astype("float32"),
                    seed=np.int32(seed))
print(f"  motivation_day.npz    seed {seed}, 86,400 slots from the test split")
PYEOF

rm -f "$ZIP"; zip -qrj "$ZIP" "$OUT"
echo
echo "=== done: $ZIP"
ls -la "$OUT"
echo
echo "Unpack and supply these files to the figure notebook:"
ls -1 "$OUT" | sed 's/^/  /'
