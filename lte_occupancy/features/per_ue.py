"""
features/per_ue.py — Per-UE feature vector for the Per-UE estimators.

The optional cell-context block feeds cell-level signals into the per-UE model.
It now uses the OBSERVABLE ``surge_flag`` (see features/cell.observable_surge)
instead of the former oracle ``is_burst`` that read the burst schedule (fix 10.2).
The vector length is unchanged (surge_flag simply takes is_burst's slot).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def per_ue_features_extended(tr, t: int, cell_context: Optional[dict] = None) -> list:
    elapsed = t - tr.last_seen
    tk = max(1, t - tr.first_seen)
    base = [
        elapsed,
        1.0 if tr.is_connected else 0.0,
        tr.n_connections,
        np.mean(tr.session_lengths) if tr.session_lengths else 0.0,
        tr.n_connections / tk * 1000,
        tr.n_paging_resp / max(1, tr.n_paging_sent),
        tr.total_active_time / tk,
        tk,
        np.std(tr.session_lengths) if len(tr.session_lengths) > 1 else 0.0,
    ]
    if cell_context is not None:
        base += [
            float(cell_context.get("n_connected_t", 0)),
            float(cell_context.get("surge_flag", 0)),   # observable (was oracle is_burst)
            float(cell_context.get("t_sin", 0)),
            float(cell_context.get("t_cos", 0)),
        ]
    return base
