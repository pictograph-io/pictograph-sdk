"""Annotation-comment Pydantic model.

An inline comment / issue on a SPECIFIC annotation within an image - the
programmatic surface of the app's collaboration/QA feature. Managed via
:class:`pictograph.resources.annotation_comments.AnnotationComments`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AnnotationComment(BaseModel):
    """One comment on an annotation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    annotation_id: str
    body: str
    resolved: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    user_id: str | None = None
    author_name: str | None = None
    author_username: str | None = None
    author_avatar_url: str | None = None
    is_mine: bool = False
