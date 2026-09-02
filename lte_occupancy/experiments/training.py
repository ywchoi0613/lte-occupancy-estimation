"""Backward-compatible shim.

The legacy single-seed pipeline moved to ``training_legacy.py`` (and the tuned six-model
pipeline lives in ``training_tuned.py``). This module is kept only so that older scripts
or notebooks importing ``from lte_occupancy.experiments.training import run_one_seed``
keep working. New code should import from ``training_legacy`` / ``training_tuned`` directly.
"""
from .training_legacy import run_one_seed

__all__ = ["run_one_seed"]
