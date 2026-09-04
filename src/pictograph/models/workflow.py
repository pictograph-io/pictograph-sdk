"""Workflow Pydantic models - node-graph pipelines and their runs.

"Workflow" means exactly one thing in this SDK: the graph-editor feature modelled
here - a saved DAG (source → model → filter → track → step → sink) run over an
image, video, or dataset, accessed via ``client.workflows``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkflowStatus = Literal["draft", "ready", "archived"]
WorkflowRunStatus = Literal["queued", "processing", "completed", "error", "cancelled"]


class Workflow(BaseModel):
    """A saved node-graph workflow."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    name: str
    description: str | None = None
    graph: dict[str, Any] = Field(default_factory=dict)
    template_key: str | None = None
    status: WorkflowStatus = "draft"
    last_run_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkflowRun(BaseModel):
    """One execution of a workflow over a source."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    workflow_id: str
    status: WorkflowRunStatus
    progress: float = 0.0
    frames_total: int | None = None
    frames_done: int = 0
    sample_fps: float | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    # B473 - per condition node {passed, failed, combinator, rules}: how an if/else
    # condition split the run. Empty when the graph has no condition node.
    condition_results: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Un-charged pre-run estimate (µUSD); the settled charge is ``final_micro_usd``
    # once the run completes (charge-on-success from measured GPU time).
    deposit_micro_usd: int = 0
    final_micro_usd: int | None = None
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowRunCreated(BaseModel):
    """Run response - the new run id + ``deposit_micro_usd``, which is the
    un-charged pre-run ESTIMATE. Workflows bill ONCE, on success, from measured GPU
    time; a failed or cancelled run is free. The field name is kept for wire-compat."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    run_id: str
    # Un-charged pre-run estimate (µUSD). The settled charge lands on the run's
    # ``final_micro_usd`` once it completes; see :class:`WorkflowRun`.
    deposit_micro_usd: int = 0
