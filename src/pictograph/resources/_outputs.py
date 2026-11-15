"""A run's outputs as a request sends them (B607): a file run's (``client.workflows.run``) and a stream's
(``open_stream`` on a workflow's endpoint) - one rule, so the two never send them differently."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from pictograph.models.workflow import WorkflowRunOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

#: What a run writes beside its log: :class:`~pictograph.models.workflow.WorkflowRunOutputs` or the same as a mapping.
RunOutputs = Union[WorkflowRunOutputs, "Mapping[str, Any]"]


def outputs_body(outputs: RunOutputs | None) -> dict[str, Any] | None:
    """A run's outputs as its request sends them (validated here: a mistyped field raises before any request), or None
    when none is given."""
    if outputs is None:
        return None
    model = (
        outputs
        if isinstance(outputs, WorkflowRunOutputs)
        else WorkflowRunOutputs.model_validate(dict(outputs))
    )
    return model.model_dump(exclude_none=True)
