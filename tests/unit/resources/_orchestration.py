"""Test harness for the resource methods that reach across resources.

A handful of methods act on more than one noun - ``Images.upload_from_directory``
creates the dataset before uploading, ``Training.from_dataset`` makes the export
before training, ``Annotations.import_coco`` resolves image ids by file name.
They do it the way :meth:`Datasets.as_pytorch` already did: a function-level
import of the sibling class, constructed off the SAME transport.

This helper swaps those sibling constructors for mocks hanging off one
``client``-shaped object, and delegates the resource's OWN cross-called methods
to the same object. A test then reads exactly as it did when these were
standalone orchestrator functions - ``client.images.upload`` /
``client.datasets.get`` - while the code under test is the real method on the
real resource.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

ResourceT = TypeVar("ResourceT")

#: Sibling resource classes the orchestration methods import at call time, mapped
#: to the attribute a test reaches them through.
_SIBLINGS: tuple[tuple[str, str], ...] = (
    ("pictograph.resources.datasets.Datasets", "datasets"),
    ("pictograph.resources.images.Images", "images"),
    ("pictograph.resources.annotations.Annotations", "annotations"),
    ("pictograph.resources.exports.Exports", "exports"),
    ("pictograph.resources.models.Models", "models"),
)

_ASYNC_SIBLINGS: tuple[tuple[str, str], ...] = (
    ("pictograph.aio.resources.datasets.AsyncDatasets", "datasets"),
    ("pictograph.aio.resources.images.AsyncImages", "images"),
    ("pictograph.aio.resources.annotations.AsyncAnnotations", "annotations"),
)


@contextlib.contextmanager
def sibling_resources(client: MagicMock, *, is_async: bool = False) -> Iterator[None]:
    """Route every sibling-resource construction to ``client.<attr>`` for the block."""
    with contextlib.ExitStack() as stack:
        for target, attr in _ASYNC_SIBLINGS if is_async else _SIBLINGS:
            stack.enter_context(patch(target, new=_constant(getattr(client, attr))))
        yield


def build(
    resource_cls: type[ResourceT],
    client: MagicMock,
    *,
    own: str,
    delegate: Sequence[str] = (),
) -> ResourceT:
    """Instantiate ``resource_cls`` on a mock transport, delegating ``delegate``.

    Args:
        resource_cls: The resource under test (e.g. ``Images``).
        client: The client-shaped MagicMock the test configures.
        own: Which ``client`` attribute this resource IS (e.g. ``"images"``).
        delegate: Method names the orchestration calls on ITSELF (``self.upload``,
            ``self.iter``, ...). Each is replaced by ``client.<own>.<name>`` so a
            test stubs it in one place. The method under test is never delegated -
            it is the real thing.
    """
    resource = resource_cls(MagicMock())  # type: ignore[call-arg]
    for name in delegate:
        setattr(resource, name, getattr(getattr(client, own), name))
    return resource


def _constant(value: Any) -> Any:
    """A stand-in constructor that ignores the transport and yields ``value``."""

    def _factory(_transport: Any) -> Any:
        return value

    return _factory
