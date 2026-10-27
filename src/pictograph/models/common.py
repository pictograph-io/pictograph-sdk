"""Common geometric primitives shared across annotation types.

These types are immutable (``frozen=True``) so they can safely live inside
caches, sets, and as dict keys. Mutating an annotation's geometry should be
done by constructing a fresh ``Point`` / ``BoundingBox`` and assigning it.

Coordinates are absolute pixel coordinates - the SDK does not normalise. The
canonical Pictograph JSON format the backend stores uses the same shape::

    {"x": 100.0, "y": 200.0}  # Point
    {"x": 100.0, "y": 200.0, "w": 50.0, "h": 80.0}  # BoundingBox

Float coordinates are accepted (sub-pixel annotations are valid for some CV
workflows). Integer literals are coerced to float automatically by Pydantic.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field


def _validate_non_blank(value: str) -> str:
    """Reject empty / whitespace-only strings.

    Pydantic's ``min_length=1`` does not catch ``"   "`` because it operates
    on the raw string length. We need to also strip-and-check.
    """
    if not value.strip():
        raise ValueError("must not be blank or whitespace-only")
    return value


NonBlankStr = Annotated[str, AfterValidator(_validate_non_blank)]
"""A string that is at least one non-whitespace character.

Use for class labels, IDs, names - anywhere an empty value is a bug rather
than a meaningful "no value" signal.
"""


class Point(BaseModel):
    """A 2D point in absolute pixel coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float


class BoundingBox(BaseModel):
    """A bounding box: top-left corner + size, absolute pixels.

    ``w`` and ``h`` must be strictly positive - a zero-area box is not a
    valid annotation. Negative ``x`` / ``y`` are accepted (some workflows use
    them for off-canvas reference points).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float
    w: float = Field(gt=0, description="Width in pixels (strictly positive).")
    h: float = Field(gt=0, description="Height in pixels (strictly positive).")


class BulkDeleteResult(BaseModel):
    """Result of a server-side bulk delete (one chunked, org-scoped call).

    Resource-agnostic so every bulk-delete surface can share it. The op is
    idempotent: ``succeeded`` is the subset of requested ids that existed in
    your organization and were removed, while ``not_found`` are ids that did
    not resolve (already gone, or never in your org). Re-running a completed
    bulk delete therefore succeeds with every id reported in ``not_found``.

    The house rules make ``succeeded`` the canonical key across every bulk
    op (delete *and* state-change), aligning this with :class:`BulkActionResult`.
    Backends that still emit the legacy ``deleted`` key parse transparently via
    an alias, and :attr:`deleted` remains a read-only shim for pre-2.0 callers.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    succeeded: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("succeeded", "deleted"),
        description="Ids that existed and were deleted.",
    )
    not_found: list[str] = Field(
        default_factory=list,
        description="Requested ids that did not resolve (idempotent re-runs).",
    )
    count: int = Field(default=0, description="Number of rows actually deleted.")

    @property
    def deleted(self) -> list[str]:
        """Deprecated alias for :attr:`succeeded` (the pre-2.0 field name)."""
        return self.succeeded


class BulkActionResult(BaseModel):
    """Result of a server-side bulk state-change (e.g. pause/resume).

    Like :class:`BulkDeleteResult` but for ops that transition rather than
    remove. ``succeeded`` is the subset of requested ids that this call actually
    transitioned; ``not_found`` are ids that did not resolve in your org OR were
    not in an actionable state (e.g. already paused / not paused / insufficient
    credits to resume), so a re-run is idempotent.
    """

    model_config = ConfigDict(extra="ignore")

    succeeded: list[str] = Field(
        default_factory=list, description="Ids this call actually transitioned."
    )
    not_found: list[str] = Field(
        default_factory=list,
        description="Ids that did not resolve or were not in an actionable state.",
    )
    count: int = Field(default=0, description="Number of ids actually transitioned.")
