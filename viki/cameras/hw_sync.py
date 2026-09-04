"""Fail-closed hardware-sync policy for multi-Kinect rigs.

This module is deliberately SDK-free so the wiring/configuration rules can be
validated in unit tests and by the server before any USB handle is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


WIRED_STANDALONE = 0
WIRED_MASTER = 1
WIRED_SUBORDINATE = 2


class HardwareSyncError(RuntimeError):
    """The connected Kinect rig cannot safely run in hardware-sync mode."""


@dataclass(frozen=True)
class SyncRole:
    mode: int
    delay_us: int = 0

    @property
    def name(self) -> str:
        return {
            WIRED_STANDALONE: "standalone",
            WIRED_MASTER: "master",
            WIRED_SUBORDINATE: "subordinate",
        }[self.mode]


def build_sync_plan(
    detected_ids: Iterable[str], spec: Mapping[str, object] | None,
) -> dict[str, SyncRole]:
    """Return the only permitted role for each connected Kinect.

    A single connected Kinect may run standalone.  With two or more devices,
    the configuration must name exactly one detected master and every other
    detected device as a subordinate.  Missing, duplicate, stale, or partial
    assignments are rejected instead of degrading to software sync.
    """
    detected = tuple(sorted({str(device_id) for device_id in detected_ids}))
    if len(detected) < 2:
        return {device_id: SyncRole(WIRED_STANDALONE) for device_id in detected}

    cfg = dict(spec or {})
    master = str(cfg.get("master") or "")
    raw_subordinates = cfg.get("subordinates") or []
    if not isinstance(raw_subordinates, (list, tuple)):
        raise HardwareSyncError(
            "KINECT_SYNC.subordinates must be a list of Kinect device IDs"
        )
    subordinates = tuple(str(device_id) for device_id in raw_subordinates)
    if not master:
        raise HardwareSyncError(
            f"{len(detected)} Kinects detected but KINECT_SYNC.master is not set"
        )
    if len(set(subordinates)) != len(subordinates):
        raise HardwareSyncError("KINECT_SYNC.subordinates contains duplicate IDs")
    if master in subordinates:
        raise HardwareSyncError(
            f"KINECT_SYNC assigns {master} as both master and subordinate"
        )

    assigned = {master, *subordinates}
    connected = set(detected)
    missing = sorted(connected - assigned)
    stale = sorted(assigned - connected)
    if missing or stale:
        details = []
        if missing:
            details.append(f"unassigned connected devices: {', '.join(missing)}")
        if stale:
            details.append(f"configured but not detected: {', '.join(stale)}")
        raise HardwareSyncError(
            "KINECT_SYNC must cover the connected rig exactly (" + "; ".join(details) + ")"
        )

    try:
        delay_us = int(cfg.get("subordinate_delay_us", 0))
    except (TypeError, ValueError) as exc:
        raise HardwareSyncError("KINECT_SYNC.subordinate_delay_us must be an integer") from exc
    if delay_us < 0:
        raise HardwareSyncError("KINECT_SYNC.subordinate_delay_us must be >= 0")

    plan = {master: SyncRole(WIRED_MASTER, 0)}
    plan.update(
        {device_id: SyncRole(WIRED_SUBORDINATE, delay_us)
         for device_id in subordinates}
    )
    return plan


def require_role_jack(
    device_id: str,
    role: SyncRole,
    *,
    sync_in_connected: bool,
    sync_out_connected: bool,
) -> None:
    """Reject a configured role when its required physical jack is open."""
    if role.mode == WIRED_MASTER and not sync_out_connected:
        raise HardwareSyncError(
            f"{device_id} is the Kinect HW_SYNC master but SYNC OUT is not connected"
        )
    if role.mode == WIRED_SUBORDINATE and not sync_in_connected:
        raise HardwareSyncError(
            f"{device_id} is a Kinect HW_SYNC subordinate but SYNC IN is not connected"
        )
