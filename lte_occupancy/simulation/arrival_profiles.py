"""
simulation/arrival_profiles.py — Literature-calibrated diurnal arrival shapes.

Each site profile is a parametric shape on the 24 h circle (wrapped-Gaussian
bumps over a floor), normalised to mean == 1 over the day, and calibrated to the
tower-level statistics measured by Xu et al., "Understanding Mobile Traffic
Patterns of Large Scale Cellular Towers in Urban Environment", ACM IMC 2015
(9,600 Shanghai towers, 1 month):

  profile        peak time(s)   valley     peak/valley ratio   weekday/weekend
  -----------    ------------   --------   -----------------   ---------------
  resident       21:30          4:00-5:00      8.93                ~1.0
  transport      8:00 & 18:00   4:00-4:30    133.33                1.49
  office         10:30          5:00          22.99                1.79
  comprehensive  12:00 & 21:30  5:00           9.47                ~1.0

(peak/valley and weekday:weekend ratios from their Table 4; peak/valley TIMES
from their Table 5. The same paper shows tower traffic is reconstructed by a few
low-order frequency components — which justifies this low-order parametric form.)

SCOPE / honesty note: Xu et al. measure DATA-TRAFFIC volume, not cell-arrival
rate. We borrow only the *temporal statistics* (peak/valley times, P/V ratio,
weekday:weekend ratio) as the shape of the arrival intensity; absolute cell
scale is set independently (Little's-law initialisation + pilot calibration in
experiments/validate_fidelity.py). Traffic-volume ratios across sites are NOT
used as occupancy ratios.
"""
from __future__ import annotations

import math

import numpy as np

# (center_hour, sigma_hours, height) bumps + floor, hand-calibrated so that the
# post-normalisation shape hits the Xu targets (verified by profile_report(),
# asserted in tests/test_phase1.py within the fidelity-gate tolerance).
_PROFILE_PARAMS: dict[str, dict] = {
    "resident": dict(
        floor=0.140,
        bumps=[(21.5, 2.30, 1.32), (7.5, 1.40, 0.28), (12.5, 2.0, 0.22)],
    ),
    "office": dict(
        floor=0.060,
        bumps=[(10.5, 1.55, 1.30), (15.0, 2.30, 0.70)],
    ),
    "transport": dict(
        floor=0.0118,
        bumps=[(8.0, 1.05, 1.52), (18.0, 1.20, 1.56)],
    ),
    "comprehensive": dict(
        floor=0.125,
        bumps=[(21.5, 2.10, 1.13), (12.0, 2.30, 1.05), (8.0, 1.2, 0.30)],
    ),
}

# Weekday->weekend demand multiplier = 1 / (weekday:weekend ratio), Xu Table 4.
WEEKEND_FACTOR: dict[str, float] = {
    "resident": 1.0,          # ~identical weekday vs weekend
    "transport": 1.0 / 1.49,
    "office": 1.0 / 1.79,
    "comprehensive": 1.0,     # ~identical weekday vs weekend
}

# Gate targets (Xu Tables 4 & 5) used by profile_report() / tests.
TARGETS: dict[str, dict] = {
    "resident":      dict(peaks_h=(21.5,),      pv=8.93),
    "office":        dict(peaks_h=(10.5,),      pv=22.99),
    "transport":     dict(peaks_h=(8.0, 18.0),  pv=133.33),
    "comprehensive": dict(peaks_h=(12.0, 21.5), pv=9.47),
}

VALLEY_WINDOW_H = (3.0, 6.0)   # Xu: valley always 4:00-5:00; gate accepts 3-6 h.


def _wrapped_gauss(h: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Gaussian on the 24 h circle (shortest circular distance)."""
    d = np.abs(h - center)
    d = np.minimum(d, 24.0 - d)
    return np.exp(-(d * d) / (2.0 * sigma * sigma))


def build_shape(profile: str, n_minutes: int = 1440) -> np.ndarray:
    """Per-minute diurnal shape, normalised to mean == 1 over the day."""
    if profile not in _PROFILE_PARAMS:
        raise ValueError(f"Unknown arrival profile: {profile!r}. "
                         f"Valid: {sorted(_PROFILE_PARAMS)}")
    prm = _PROFILE_PARAMS[profile]
    h = (np.arange(n_minutes) + 0.5) * (24.0 / n_minutes)
    g = np.full(n_minutes, float(prm["floor"]))
    for c, s, ht in prm["bumps"]:
        g += ht * _wrapped_gauss(h, c, s)
    g /= g.mean()
    return g


def profile_report(profile: str) -> dict:
    """Achieved-vs-target stats for the fidelity gate (T3)."""
    g = build_shape(profile)
    h = (np.arange(len(g)) + 0.5) * (24.0 / len(g))
    tgt = TARGETS[profile]
    peaks = []
    for pk in tgt["peaks_h"]:
        # local argmax within +/-3 h of each target peak
        m = (np.abs(np.minimum(np.abs(h - pk), 24 - np.abs(h - pk))) <= 3.0)
        idx = np.argmax(np.where(m, g, -np.inf))
        peaks.append(float(h[idx]))
    pv = float(g.max() / g.min())
    # Valley check is plateau-tolerant: the night floor is flat, so we require the
    # 3-6 h window (Xu: valley 4:00-5:00) to CONTAIN the (near-)global minimum.
    win = (h >= VALLEY_WINDOW_H[0]) & (h <= VALLEY_WINDOW_H[1])
    win_min = float(g[win].min())
    valley_ok = win_min <= float(g.min()) * 1.02
    valley_h = float(h[win][int(np.argmin(g[win]))])
    return {
        "profile": profile,
        "peaks_h_target": tgt["peaks_h"], "peaks_h_achieved": tuple(peaks),
        "valley_h_in_window": valley_h, "valley_ok": valley_ok,
        "pv_target": tgt["pv"], "pv_achieved": pv,
        "mean": float(g.mean()),
        "weekend_factor": WEEKEND_FACTOR[profile],
    }


def all_reports() -> list[dict]:
    return [profile_report(p) for p in _PROFILE_PARAMS]


if __name__ == "__main__":
    for r in all_reports():
        print(r)
