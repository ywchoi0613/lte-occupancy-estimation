"""Per-test environment isolation.

WHY THIS FILE EXISTS
--------------------
`build_config()` resolves the whole DGP from `LTE_*` environment variables, and several
tests set those variables directly without restoring them:

    tests/test_correctness.py:51    LTE_WARMUP_SLOTS = "600"
    tests/test_correctness.py:113   LTE_TOTAL_TIME   = "800"
    tests/test_correctness.py:276   LTE_TOTAL_TIME   = "3000"
    tests/test_correctness.py:310   LTE_TOTAL_TIME   = "3000"

pytest collects `test_correctness.py` before `test_phase1.py`, so those values are still
in the process environment when `test_phase1.test_warmup_exclusion_and_day_alignment`
runs. That test asks for `LTE_TOTAL_DAYS=10, LTE_WARMUP_DAYS=1` and asserts
`total_slots == 864000`, but `build_config` prefers the explicit slot counts:

    total_slots  = int(os.environ.get("LTE_TOTAL_TIME",    days * DAY))
    warmup_slots = int(os.environ.get("LTE_WARMUP_SLOTS",  warmup_days * DAY))

so it gets 3000 / 600 and fails. `test_phase1._fresh_config` only saves and restores the
keys it was handed, which cannot undo a variable an earlier *module* set.

This is a TEST-ISOLATION defect, not a production defect: nothing in the pipeline sets
`LTE_TOTAL_TIME` and then relies on it being absent later, and every runner script
explicitly clears `LTE_*` before pinning its own DGP.

WHY A CONFTEST FIXTURE RATHER THAN EDITING THE TESTS
----------------------------------------------------
Adding try/finally around ten assignment sites would fix these four leaks and leave the
next one to be discovered the same way — by a confusing failure in an unrelated file.
An autouse fixture makes every test start from the environment the session started with,
whatever any other test did, and it changes no existing test's logic, so it cannot alter
what any of them assert. No test in this suite mutates `LTE_*` at module import time, so
snapshotting per test is safe.

Verify the ordering dependence is really gone:

    python -m pytest -q                          # full suite
    python -m pytest -q -p no:randomly           # fixed order
    python -m pytest -q tests/test_phase1.py     # in isolation — must match the full run
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_lte_env():
    """Snapshot every LTE_* variable before a test and restore it afterwards.

    Restoring means: keys the test added are removed, keys it changed go back to their
    previous value, and keys it deleted come back. Assigning the whole dict back would not
    do the last two, and `os.environ.clear()` would drop PATH along with everything else.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("LTE_")}
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith("LTE_")]:
            if key not in saved:
                del os.environ[key]
        os.environ.update(saved)
