"""
simulation/arrival.py — Population arrival-intensity model (Phase-1, real 24 h axis).

    lambda(u) = base_rate * day_scale[day(u)] * weekend_factor(day(u))
                * shape(minute_of_day(u))  +  burst(u)
    arrivals_u ~ Poisson(lambda(u))

* shape() is the per-minute site profile from arrival_profiles.build_shape(),
  mean-normalised to 1 and calibrated to Xu et al. (IMC 2015) peak/valley times
  and peak-valley ratios. One period == one literal day (86,400 x 1 s slots).
* day_scale ~ U(lo, hi) is genuine day-to-day variability.
* weekend_factor applies the Xu weekday:weekend amount ratio on calendar
  weekends (T5b site runs); T5a controlled runs pass weekend_factor == 1.
* Bursts are per-day random flash crowds (bursts_per_day per RECORDED day,
  uniform start time, N(mean,std) duration, amplitude burst_amp_rel*base_rate),
  drawn from THIS model's rng stream — none scheduled during warm-up.

Time convention: the engine passes ABSOLUTE slot u in [0, warmup+horizon);
recorded t = u - warmup. Day boundaries align because warmup is whole days.
"""
from __future__ import annotations

import numpy as np

from ..config.schema import ArrivalCfg
from .arrival_profiles import build_shape, WEEKEND_FACTOR

DAY_SLOTS = 86_400


class ArrivalModel:
    def __init__(self, cfg: ArrivalCfg, total_slots_abs: int, warmup_slots: int,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.base = float(cfg.base_rate)
        n_days = total_slots_abs // DAY_SLOTS + cfg.day_scale_buffer_days + 1
        lo, hi = cfg.day_scale_range
        self._day_scale = rng.uniform(lo, hi, size=n_days)

        # per-minute shape LUT (config override wins; else built from profile)
        if cfg.shape_minutes:
            self._shape = np.asarray(cfg.shape_minutes, dtype=float)
        else:
            self._shape = build_shape(cfg.profile)
        self._mpd = len(self._shape)                     # minutes per day (1440)

        # calendar / weekend factor per absolute day
        cal = cfg.weekday_calendar
        # weekend_factor semantics: 1.0 = disabled (T5a controlled runs);
        # <= 0.0 = sentinel "use the profile's Xu-derived table" (T5b);
        # any other value = explicit override.
        if cfg.weekend_factor == 1.0:
            wf = 1.0
        elif cfg.weekend_factor <= 0.0:
            wf = WEEKEND_FACTOR.get(cfg.profile, 1.0)
        else:
            wf = cfg.weekend_factor
        self._wfac = np.ones(n_days)
        if cal and wf != 1.0:
            for d in range(min(n_days, len(cal))):
                if cal[d] in (5, 6):                      # Sat / Sun
                    self._wfac[d] = wf

        # per-day random flash-crowd bursts (recorded days only)
        self._bursts: list[tuple[int, int, float]] = []
        first_rec_day = warmup_slots // DAY_SLOTS
        last_day = total_slots_abs // DAY_SLOTS
        amp = cfg.burst_amp_rel * self.base
        dmean, dstd = cfg.burst_duration_s
        for d in range(first_rec_day, last_day):
            for _ in range(cfg.bursts_per_day):
                start = d * DAY_SLOTS + int(rng.integers(0, DAY_SLOTS))
                dur = max(10, int(rng.normal(dmean, dstd)))
                self._bursts.append((start, dur, amp))
        # legacy explicit schedule still honoured if provided (absolute slots)
        for (s, d, a) in cfg.burst_events:
            self._bursts.append((int(s), int(d), float(a)))

    # ------------------------------------------------------------------
    def shape_at(self, u: int) -> float:
        minute = (u % DAY_SLOTS) * self._mpd // DAY_SLOTS
        return float(self._shape[minute])

    def intensity(self, u: int) -> float:
        day = (u // DAY_SLOTS) % len(self._day_scale)
        m = self.base * self._day_scale[day] * self._wfac[day] * self.shape_at(u)
        burst = sum(a for s, d, a in self._bursts if s <= u < s + d)
        return float(max(0.0, m + burst))
