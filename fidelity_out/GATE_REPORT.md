# Phase 1 Gate Report

Generated: 2026-07-27 23:51:45

- G1 (tests): driven by run_phase1_pilot.sh (pytest gates the pipeline; see logs_phase1/g1_pytest.log)
- G5 (retune readiness): s2 re-tuning APPROVED on the new DGP. Launch via `CONFIRM=1 bash run_phase1_retune.sh` after this report is PASS.

## Preconditions

- G1 marker (fidelity_out/G1_PASS): **present**
- calibration_comprehensive.json: **present**
- scenario metrics: 4/4

## G3 — mean occupancy (target 320, +/-5%; drift <=3%)

| scenario | n-bar | by day | drift | gate |
| --- | --- | --- | --- | --- |
| resident | 317.64 | 321.5, 313.8 | 2.41% | PASS |
| office | 322.12 | 324.1, 320.1 | 1.24% | PASS |
| transport | 325.09 | 329.3, 320.8 | 2.59% | PASS |
| comprehensive | 318.05 | 317.8, 318.3 | 0.18% | PASS |

Cross-scenario max/min - 1 <= 5%: **PASS**

## G3 — arrival shape (peak +/-30 min; LUT P/V x//1.5)

| scenario | peaks target (h) | peaks realized (h) | LUT P/V (target) | realized P/V note | gate |
| --- | --- | --- | --- | --- | --- |
| resident | 21.50 | 21.25 | 8.3 (8.9) | 0.99x of LUT (WARN only) | PASS |
| office | 10.50 | 10.75 | 24.5 (23.0) | 1.10x of LUT (WARN only) | PASS |
| transport | 8.00, 18.00 | 7.75, 17.75 | 133.2 (133.3) | 0.97x of LUT (WARN only) | PASS |
| comprehensive | 12.00, 21.50 | 11.75, 21.25 | 9.1 (9.5) | 1.01x of LUT (WARN only) | PASS |

## G2 — behavioral fidelity

| scenario | personas (low/med/high, meas vs adj) | order | RACH coll. | giveup | MT share (f_mt) | MT-voice ratio | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resident | low: mech +2.0% | res/day 8.0 (nom 15); medium: mech +0.2% | res/day 18.0 (nom 55); high: mech -4.2% | res/day 30.8 (nom 150) | ok | 0.029% | 0 | 0.045 (0.050) | 1.01 | PASS |
| office | low: mech +1.5% | res/day 7.5 (nom 15); medium: mech -1.0% | res/day 18.6 (nom 55); high: mech -1.9% | res/day 31.2 (nom 150) | ok | 0.038% | 0 | 0.043 (0.050) | 0.97 | PASS |
| transport | low: mech +1.0% | res/day 7.7 (nom 15); medium: mech +2.1% | res/day 18.8 (nom 55); high: mech +1.1% | res/day 31.2 (nom 150) | ok | 0.039% | 0 | 0.044 (0.050) | 1.00 | PASS |
| comprehensive | low: mech -4.3% | res/day 7.7 (nom 15); medium: mech +2.0% | res/day 18.1 (nom 55); high: mech +0.8% | res/day 28.6 (nom 150) | ok | 0.028% | 0 | 0.044 (0.050) | 1.05 | PASS |

## G4 — compute budget

- measured: 1565 slots/s -> full 11-day run ~= 10.1 min
- 220 truth runs / 4 workers ~= **0.4 wall days** (budget OK)

## T2 extended — services / paging / causes (informational)

### resident

| service | n | wall mean (s) | wall p50 | active mean (s) | bursts mean | censored |
| --- | --- | --- | --- | --- | --- | --- |
| browsing | 8265 | 1707 | 1673 | 280 | 37.8 | 2145 |
| streaming | 6440 | 1192 | 1178 | 456 | 16.0 | 1274 |
| voice | 2820 | 129 | 128 | 120 | 1.0 | 102 |

causes: mo_bg=519041, mt_data=18197, mt_engage=6042, mt_voice=617, service_resume=354788, user_mo=10866
RACH chain/day: preamble 646000 : rar 454776 : msg3 454870 : setup 454776 (msg3/setup 1.0002)
paging: 43.39 pages/UE/day; retry hist {'1': 22436, '2': 2210, '3': 237}

### office

| service | n | wall mean (s) | wall p50 | active mean (s) | bursts mean | censored |
| --- | --- | --- | --- | --- | --- | --- |
| browsing | 9828 | 1705 | 1682 | 280 | 37.8 | 2536 |
| streaming | 7611 | 1194 | 1175 | 459 | 16.1 | 1510 |
| voice | 3847 | 127 | 127 | 118 | 1.0 | 116 |

causes: mo_bg=479398, mt_data=16253, mt_engage=5482, mt_voice=830, service_resume=419840, user_mo=14978
RACH chain/day: preamble 665384 : rar 468390 : msg3 468516 : setup 468390 (msg3/setup 1.0003)
paging: 38.92 pages/UE/day; retry hist {'1': 20333, '2': 2037, '3': 222}

### transport

| service | n | wall mean (s) | wall p50 | active mean (s) | bursts mean | censored |
| --- | --- | --- | --- | --- | --- | --- |
| browsing | 9923 | 1705 | 1678 | 281 | 37.8 | 2562 |
| streaming | 7652 | 1194 | 1173 | 459 | 16.1 | 1465 |
| voice | 3820 | 128 | 129 | 119 | 1.0 | 118 |

causes: mo_bg=471992, mt_data=16024, mt_engage=5509, mt_voice=839, service_resume=425088, user_mo=15067
RACH chain/day: preamble 663865 : rar 467260 : msg3 467388 : setup 467260 (msg3/setup 1.0003)
paging: 38.17 pages/UE/day; retry hist {'1': 20214, '2': 1947, '3': 237}

### comprehensive

| service | n | wall mean (s) | wall p50 | active mean (s) | bursts mean | censored |
| --- | --- | --- | --- | --- | --- | --- |
| browsing | 8972 | 1712 | 1709 | 282 | 38.0 | 2307 |
| streaming | 6875 | 1204 | 1182 | 461 | 16.1 | 1377 |
| voice | 3237 | 128 | 128 | 119 | 1.0 | 108 |

causes: mo_bg=495797, mt_data=17081, mt_engage=5749, mt_voice=734, service_resume=386767, user_mo=12616
RACH chain/day: preamble 652572 : rar 459372 : msg3 459464 : setup 459372 (msg3/setup 1.0002)
paging: 41.06 pages/UE/day; retry hist {'1': 21270, '2': 2077, '3': 231}


---

**OVERALL: PASS**
