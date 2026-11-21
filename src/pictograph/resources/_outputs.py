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


#: What a model deployment's stream writes (2026-11-15): its Passive capture alone - a deployment has no events to
#: capture or send. The API holds a deployment's stream to the same list.
DEPLOYMENT_OUTPUTS = ("passive_capture",)
#: Each output's name as the app's Run view and the API's refusals say it.
OUTPUT_NAMES = {
    "annotated_video": "Write annotated video",
    "annotations": "Write annotations",
    "passive_capture": "Passive capture",
    "event_capture": "Event capture",
    "webhook": "Send to a webhook",
}


def deployment_outputs_body(outputs: RunOutputs | None) -> dict[str, Any] | None:
    """A model deployment's stream's outputs as its request sends them: validated as a run's are, then held to what a
    deployment writes (Passive capture alone) - anything else is a ``ValueError`` before any request, in the words the
    API refuses it with."""
    sent = outputs_body(outputs)
    if not sent:
        return sent
    for key, value in sent.items():
        if key not in DEPLOYMENT_OUTPUTS and value not in (None, False):
            raise ValueError(
                f"A deployment's stream writes Passive capture - {OUTPUT_NAMES.get(key, key)} is a workflow's."
            )
    return {k: v for k, v in sent.items() if k in DEPLOYMENT_OUTPUTS} or None
