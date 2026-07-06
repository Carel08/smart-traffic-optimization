"""
Event logging utilities for the traffic simulation.

The event log is mainly for demo and interpretability:
- scenario events
- accident events
- pedestrian queue warnings
- emergency dispatch
- signal preemption events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class EventLogEntry:
    time_step: int
    event_type: str
    message: str
    severity: str = "info"


class EventLogger:
    def __init__(self):
        self.events: List[EventLogEntry] = []
        self._seen_keys: set[str] = set()

    def log(
        self,
        time_step: int,
        event_type: str,
        message: str,
        severity: str = "info",
        dedupe_key: str | None = None,
    ) -> None:
        """
        Add event to log.

        dedupe_key prevents the same event from being logged repeatedly.
        """
        if dedupe_key is not None:
            if dedupe_key in self._seen_keys:
                return
            self._seen_keys.add(dedupe_key)

        self.events.append(
            EventLogEntry(
                time_step=time_step,
                event_type=event_type,
                message=message,
                severity=severity,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([event.__dict__ for event in self.events])