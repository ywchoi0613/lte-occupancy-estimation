"""
event_order.py — canonical ordering for per-UE event streams.

Episode-aware Mode-B observation (observation/mode_b.py) can place a (clamped)
release and the next session's setup in the SAME slot:

    t=15  disconnect   (previous session's delayed release, clamped to next connect)
    t=15  connect      (new session's RRC setup)

For the per-UE state machine to read these correctly, the disconnect MUST be
processed before the connect (close the old session, then open the new one).
Plain ``sorted(events)`` orders by the event STRING, which puts "connect" before
"disconnect" ('c' < 'd') and silently reintroduces the "delayed release closes a
new session" bug. Every place that orders events therefore uses EVENT_PRIORITY —
in the observer, in the estimator, and again after JSON round-trips or S-TMSI
fragmentation.
"""
from __future__ import annotations

# Lower number = processed first within the same slot.
EVENT_PRIORITY = {
    "disconnect": 0,
    "connect": 1,
}
_DEFAULT_PRIORITY = 2  # gt_enter / gt_exit / s_tmsi_realloc: after connect, order-neutral


def event_priority(event_type: str) -> int:
    return EVENT_PRIORITY.get(event_type, _DEFAULT_PRIORITY)


def event_key(t_et):
    """Sort key for a ``(t, event_type)`` tuple: (time, priority)."""
    return (t_et[0], event_priority(t_et[1]))


def event_key_with_pid(t_pid_et):
    """Sort key for a ``(t, pid, event_type)`` tuple: (time, pid, priority).
    Tracks are independent, so pid only makes the order deterministic; the
    priority tie-break is what keeps same-slot disconnect-before-connect."""
    return (t_pid_et[0], t_pid_et[1], event_priority(t_pid_et[2]))
