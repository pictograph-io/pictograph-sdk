"""Task models - annotation tasks and per-annotator contribution tracking.

Returned by the :class:`pictograph.resources.tasks.Tasks` resource. A **task**
assigns a frozen set of a dataset's images to one or more org members for
annotation or review. The contribution breakdown lets you audit or bill against
the annotation work: who annotated how many images, their measured active
editing time, and the annotations they authored.

Response-side schema (``extra="ignore"``) so backend column additions don't
break callers.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Task(BaseModel):
    """One annotation task in the organization (list item)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str = Field(
        validation_alias=AliasChoices("dataset_id", "project_id"),
        description="The dataset the task's images belong to.",
    )
    dataset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset", "project"),
        description="Dataset name (may be null if renamed/removed).",
    )
    title: str
    kind: str = Field(description="'annotate' or 'review'.")
    status: str = Field(description="'open' or 'done'.")
    created_at: datetime
    image_count: int = Field(description="Number of images frozen into the task.")
    assignee_count: int = Field(description="Number of org members the task is assigned to.")


class TaskContribution(BaseModel):
    """One annotator's contribution to a task.

    ``active_seconds`` and ``images_worked`` are exact (from per-image active-time
    tracking). ``images_completed`` and ``annotations_added`` are LAST-WRITER
    attribution - they credit the most recent saver of each image, since there is
    no per-shape authorship record.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str
    full_name: str
    email: str | None = None
    avatar_url: str | None = None
    is_assignee: bool = Field(
        description="Whether this person is a task assignee (vs a non-assignee contributor)."
    )
    active_seconds: int = Field(
        description="Measured active editing time on the task's images (seconds)."
    )
    images_worked: int = Field(
        description="Distinct task images this person annotated or authored."
    )
    images_completed: int = Field(
        description="Task images they last-saved that are now at 'complete'."
    )
    annotations_added: int = Field(description="Annotations on images they most recently authored.")

    @property
    def active_minutes(self) -> float:
        """``active_seconds`` as minutes (float), for display."""
        return self.active_seconds / 60.0


class TaskContributions(BaseModel):
    """Per-annotator contribution breakdown for a task, with rollup totals."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    contributors: list[TaskContribution]
    contributor_count: int
    total_images: int = Field(description="Total images frozen into the task.")
    images_complete: int = Field(description="Task images at status 'complete' (any annotator).")
    total_active_seconds: int = Field(
        description="Sum of every annotator's active editing time (seconds)."
    )

    @property
    def total_active_minutes(self) -> float:
        """``total_active_seconds`` as minutes (float), for display."""
        return self.total_active_seconds / 60.0
