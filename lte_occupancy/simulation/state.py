"""simulation/state.py — UE state machine primitives."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class State(IntEnum):
    OUT = 0
    IDLE = 1
    CONNECTED = 2


@dataclass
class Person:
    pid: int
    usage_intensity: str
    dwell_type: str
    device_bg_class: str
    enter_time: int
    exit_time: int
    state: State = State.OUT
    session_remaining: int = 0
    rsrp: float = -80.0
    distance: float = 0.0
    is_resident: bool = False
    inactivity_remaining: int = 0
    in_inactivity_wait: bool = False
    next_bg_reconnect: int = 0
    is_bg_session: bool = False
    active_service: Optional[str] = None
    service_total_remaining: int = 0
    service_burst_remaining: int = 0
    service_idle_remaining: int = 0
    in_service_idle: bool = False
    has_attached: bool = False
    awaiting_paging_response: bool = False
    last_service: Optional[str] = None
    # S-TMSI fields (set by observation.mode_b when realloc is active)
    _next_realloc: Optional[int] = None
    _s_tmsi_epoch: int = 0
