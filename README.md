# UE-Aware Cell Occupancy Inference from Partially Observable Cellular Control-Plane Data

Code for the paper of the same name. It estimates how many UEs are physically present in
a single LTE cell (`n_present`), including those sitting in RRC IDLE, from control-plane
signals alone, and measures how much accuracy survives when the observer loses operator
cooperation.

The connected count is not the occupancy. UEs stay in the cell in RRC IDLE between traffic
bursts, so `n_connected` tracks the active fraction rather than the population. Everything
here is built around inferring the latter from the former plus temporal structure.

Two observation regimes are simulated:

- **Mode A** — cooperative network-side observation (serving eNB together with the MME).
  21 aggregate counters, explicit release events, persistent UE identities.
- **Mode B** — passive over-the-air monitoring. Eight counters, releases inferred rather
  than observed, temporary identifiers, DRX-induced blindness, and detection lag.

Six estimators run in each mode, spanning three granularities and two learner families:
`Cell_XGB`, `Cell_LSTM` (aggregate counters), `PerUE_XGB`, `PerUE_LSTM` (per-UE tracks),
and `Hybrid_XGB`, `Hybrid_LSTM` (gated fusion of the two). Modes A and B share one
selected hyperparameter configuration per estimator, so an A−B difference reflects
observability rather than tuning effort.

The headline result is a parity, not a gap. On ten held-out five-day traces of a cell
averaging about 317 UEs, Hybrid-XGB reaches 5.00 ± 0.98 UEs MAE in Mode A and 5.05 ± 0.98
in Mode B. Replacing aggregate counters with per-UE tracks cuts the error by 31.8 % (A) and
28.0 % (B); fusion removes a further 8.3 % and 8.0 %. What separates the estimators is per-UE
temporal continuity, not counter richness — which is why identifier reallocation, not the
counter budget, is the binding constraint on the passive observer.

## Layout

```
lte_occupancy/
  config/         schema.py (frozen dataclasses), defaults.py (single source of truth
                  for every research parameter), registry.py (emits parameter_registry.csv),
                  device.py
  simulation/     state.py, arrival.py, arrival_profiles.py, engine.py
                  TRUTH only: ground-truth counters and n_present, no observation noise
  observation/    mode_a.py, mode_b.py — what each regime can see, derived from the truth
  features/       cell.py, per_ue.py — observable signals only
  estimation/     models.py, per_ue.py, fusion.py, calibration.py, survival.py
  experiments/    runner.py, tune_runner.py, tuning.py, training_tuned.py,
                  validate_fidelity.py, make_paper_tables.py
tests/            correctness, Phase-1 fidelity, hybrid provenance
fidelity_out/     the passed fidelity gate and the calibrated arrival/access scales
```

The truth simulator and the observation layer are deliberately separate. That split is what
lets a result be read as "what actually happened" against "what each observer saw" instead
of conflating the two.

Parameter provenance is machine-checked rather than asserted. `config/registry.py` labels
every value with its source type — empirical anchor, 3GPP standard, design assumption, or
calibrated — and writes `parameter_registry.csv`, which is copied into each run directory
alongside `resolved_config.json`, `environment.json` and per-seed truth hashes.

## Reproducing the paper

Four stages, in order. Stages 1–3 are the expensive part.

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"

# Stage 0 — fidelity gate and arrival calibration.
# Already satisfied in fidelity_out/; re-run only if you change the DGP.
bash run_phase1_pilot.sh

# Stages 1-3 — tuning and the ten-seed held-out evaluation.
bash run_paper5d_controlled.sh                        # dry run: prints the plan, writes nothing
CONFIRM=1 nohup bash run_paper5d_controlled.sh > paper5d.out 2>&1 &

# Stage 4 — the 43-scenario, 5-seed robustness suite on the frozen hyperparameters.
PARAMS_DIR=tune_out_s3c5 PREFIX=s3c5_cal BASE_PREFIX=s3c5 \
  CONFIRM=1 nohup bash run_robustness5d_controlled_seeds.sh > robust5d.out 2>&1 &

# Tables.
python make_robust5d_tables.py --prefix robust_s3c5_cal \
  --params-dir tune_out_s3c5 --seeds "101 102 103 104 105" --out paper_tables
```

Every script refuses rather than degrades. `run_paper5d_controlled.sh` will not start
without a passed gate; it aborts if a journal for its own study prefix already exists, so a
second invocation cannot quietly double a branch's trial budget; and it refuses to merge a
partial set of shards, because a merged summary recording fewer than the ten pre-registered
seeds would report correct numbers with wrong provenance.

Ten seeds were fixed before any result was seen. Running five, inspecting the interval and
extending to ten if it straddled zero would be optional stopping, and the sharding makes
that extension cheap enough to be tempting — so the choice was removed rather than left to
judgement. An interval containing zero is a reportable result, and here it is the result.

### What it costs

On the hardware used for the paper — one 20-core host, 188 GB RAM, four RTX 3090s — a
single held-out seed with all six estimators in both modes takes 4.4–5.2 h of wall clock and
peaks near 70 GB resident. That footprint, not the GPU count, sets the concurrency: two main
shards fit, three do not. The full robustness suite restricted to the XGB family is roughly
five days; with the LSTM family included it is about three weeks.

The robustness suite runs XGB only, and the reason is stated in the paper rather than left
implicit. At five seeds the LSTM family's spread across test seeds is comparable to the
scenario and mode effects the suite is trying to resolve. Note precisely what that evidence
is: spread across *test seeds* varies the simulator trajectory as well as the training RNG,
so it is not an isolated measurement of training variance. The main table still reports all
six estimators.

## Scenario knobs

Scenarios are selected by environment variable, all prefixed `LTE_`. The pipeline scripts
clear every `LTE_*` before setting the canonical values, because an inherited
`LTE_STMSI_REALLOC=1` or `LTE_SCALE_PROFILE=tiny` from the calling shell would silently
retune the whole paper on a different data-generating process.

The axis that matters most is `LTE_STMSI_REALLOC`. With reallocation on and no
re-association, orphaned track segments keep being counted as present: Mode-B PerUE-XGB
goes from 5.79 to 105.09 UEs MAE (mean signed error +100.5 UEs, so almost the entire error
is one-directional over-counting), Hybrid-XGB from 5.29 to 18.50, while the identity-free
Cell-XGB is bit-identical at 7.50. Reallocation uses a dedicated RNG stream
(`default_rng(seed + 777)`) specifically so that turning it on cannot desynchronise the
physical-truth streams; the per-seed truth hashes are what verify this held.

## Data availability

This repository holds the simulator, the estimators and the experiment scripts. It does not
hold the outputs: the full experiment tree is about 26 GB, dominated by per-seed prediction
archives. The accompanying archived bundle contains the resolved configurations, parameter
registries and hashes, tuning journals, truth-trajectory hashes, per-seed held-out results,
robustness outputs and the table-generating scripts. See the paper's Data Availability
statement for its DOI.

No real network traffic was collected or used. All results come from the simulator in
`lte_occupancy/simulation/`, and the paper states the limits of that choice.

## Citation

See `CITATION.cff`.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`. The grant covers the code in this
repository; third-party material retains the terms of its original source.
