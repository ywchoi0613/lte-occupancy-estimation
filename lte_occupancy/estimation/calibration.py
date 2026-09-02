"""
calibration.py — Map a raw Per-UE estimate to an occupancy count.

The Per-UE estimators (survival-sum / XGB-sum) produce a raw signal that is
correlated with, but not equal to, the true occupancy. We fit a monotonic
(isotonic) calibration curve  raw -> count  on the training portion and apply
it to the whole sequence. Isotonic regression is non-parametric and only
assumes the mapping is non-decreasing, which is the natural assumption here
(more tracked UEs => more people present).

Reference: Niculescu-Mizil & Caruana (2005) ICML.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def isotonic_correction(raw_est: np.ndarray, actual: np.ndarray, split_idx: int) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(raw_est[:split_idx], actual[:split_idx])
    return iso.predict(raw_est).flatten()
