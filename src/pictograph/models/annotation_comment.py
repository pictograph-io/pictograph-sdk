"""Annotation-comment Pydantic model.

An inline comment / issue on a SPECIFIC annotation within an image - the
programmatic surface of the app's collaboration/QA feature. Managed via
:class:`pictograph.resources.annotation_comments.AnnotationComments`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AnnotationCommentReaction(BaseModel):
    """A grouped reaction tally on a comment (one entry per reaction key)."""

    model_config = ConfigDict(extra="ignore")

    reaction: str
    count: int = 0
    #: Whether the calling user has added this reaction.
    reacted: bool = False


class AnnotationComment(BaseModel):
    """One comment on an annotation.

    The primary annotator<->assigner communications channel: comments reply-thread
    (``parent_comment_id``), carry reactions, and can be resolved/reopened. A
    soft-deleted comment keeps its place in the thread with ``is_deleted=True`` and
    an empty ``body``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    annotation_id: str
    body: str
    resolved: bool = False
    #: The parent comment this one replies to (``None`` for a top-level comment).
    parent_comment_id: str | None = None
    is_edited: bool = False
    is_deleted: bool = False
    resolved_by: str | None = None
    resolved_by_name: str | None = None
    resolved_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    user_id: str | None = None
    author_name: str | None = None
    author_username: str | None = None
    author_avatar_url: str | None = None
    is_mine: bool = False
    reactions: list[AnnotationCommentReaction] = Field(default_factory=list)
