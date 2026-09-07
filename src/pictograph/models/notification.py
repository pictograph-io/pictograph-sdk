"""Notification model - the organization event feed.

Response-side schema (``extra="ignore"``) so backend column additions never
break parsing. Notifications are emitted on job-lifecycle events (training
complete, export ready, batch auto-annotate done) - an agent can poll them to
learn what finished without tracking every run id itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Notification(BaseModel):
    """A single row from the ``notifications`` table."""

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str
    user_id: str | None = None
    type: str = Field(description="Event type slug (e.g. 'training_complete', 'export_ready').")
    title: str
    message: str | None = None
    metadata: dict[str, Any] | None = None
    read: bool = False
    created_at: datetime
