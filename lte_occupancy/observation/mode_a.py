"""
observation/mode_a.py — Mode A: cooperative eNB (idealized UPPER bound).

A cooperative eNB exposes its full S1AP/NAS/RRC counters, so Mode A observes the
truth counters directly (no degradation). This observer therefore just declares
the observable column set and supplies the Mode-A naive baseline.

Note: the observable set explicitly includes ``n_paging_response`` (21
counters = legacy 20 + ``n_rar``).
A cooperative eNB genuinely sees paging responses, and the feature pipeline uses
them; listing it here keeps the observable count consistent with the model input
(fixes the former "19 observations" mismatch).
"""
from __future__ import annotations

import pandas as pd

from ..config.schema import ObservationCfg


class ModeAObserver:
    mode = "A"

    def __init__(self, cfg: ObservationCfg):
        self.feature_columns = list(cfg.mode_a_features)

    def naive_connected(self, df: pd.DataFrame):
        """Naive occupancy estimate for Mode A = the (clean) connected count,
        which a cooperative eNB observes directly."""
        return df["n_connected"].values
