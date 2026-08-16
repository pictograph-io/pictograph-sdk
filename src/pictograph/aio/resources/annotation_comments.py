"""Async Annotation-comments resource.

Async twin of :class:`pictograph.resources.annotation_comments.AnnotationComments`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.aio.resources import _resolve
from pictograph.models.annotation_comment import AnnotationComment
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Sequence

_API_PATH = "/api/v1/developer/annotation-comments"


class AsyncAnnotationComments(AsyncResource):
    """List / create / resolve / delete comments on annotations (async)."""

    async def list(self, dataset_name: str, image: str) -> Sequence[AnnotationComment]:
        """Every comment on the given image's annotations (oldest first).

        Addressed by ``(dataset name, filename)`` like :meth:`Annotations.get`.
        """
        image_id = await _resolve.image_id(self._transport, dataset_name, image)
        response = await self._transport.request("GET", _API_PATH, params={"image_id": image_id})
        items = response.get("comments", []) if isinstance(response, dict) else []
        return self._parse_list(AnnotationComment, items)

    async def create(
        self, dataset_name: str, image: str, *, annotation_id: str, body: str
    ) -> AnnotationComment:
        """Comment on an annotation. ``@username`` mentions notify org members.

        The image is addressed by ``(dataset name, filename)``. ``annotation_id``
        stays an id: an annotation has a CLASS name, not a unique one - a hundred
        boxes on an image can all be called "car" - so there is no name that
        identifies one.
        """
        image_id = await _resolve.image_id(self._transport, dataset_name, image)
        payload: dict[str, Any] = {
            "image_id": image_id,
            "annotation_id": annotation_id,
            "body": body,
        }
        response = await self._transport.request("POST", _API_PATH, json=payload)
        return self._parse(AnnotationComment, response["comment"])

    async def update(
        self, comment_id: str, *, body: str | None = None, resolved: bool | None = None
    ) -> AnnotationComment:
        """Edit the body (author only) and/or resolve/reopen the comment."""
        payload: dict[str, Any] = {}
        if body is not None:
            payload["body"] = body
        if resolved is not None:
            payload["resolved"] = resolved
        response = await self._transport.request("PATCH", f"{_API_PATH}/{comment_id}", json=payload)
        return self._parse(AnnotationComment, response["comment"])

    async def resolve(self, comment_id: str, *, resolved: bool = True) -> AnnotationComment:
        """Convenience: mark a comment resolved (or reopen with ``resolved=False``)."""
        return await self.update(comment_id, resolved=resolved)

    async def delete(self, comment_id: str) -> None:
        """Delete a comment (the author, or an org admin/owner)."""
        await self._transport.request("DELETE", f"{_API_PATH}/{comment_id}")
