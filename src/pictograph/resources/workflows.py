"""Workflows resource - the node-graph feature, run over an image / video / dataset.

Build a graph in the app, then drive runs headlessly here: ``create`` → ``run`` →
poll ``get_run`` until ``status == "completed"`` and read its ``artifacts``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.workflow import (
    Workflow,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowStatus,
)
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_API = "/api/v1/developer/workflows"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_RUN_TIMEOUT = 3600.0  # 1 hour - matches the workflow runner's own timeout


class Workflows(Resource):
    """Manage node-graph workflows and their runs."""

    def list(self) -> Sequence[Workflow]:
        """List every workflow in your organization."""
        response = self._transport.request("GET", f"{_API}/")
        return self._parse_list(Workflow, response.get("workflows", []))

    def get(self, workflow: str) -> Workflow:
        """Fetch a single workflow by NAME (includes a ``validation`` list).

        A UUID is accepted too.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("GET", f"{_API}/{workflow_id}")
        return self._parse(Workflow, response["workflow"])

    def create(
        self,
        name: str,
        graph: dict[str, Any],
        *,
        readme: str | None = None,
        description: str | None = None,
        template_key: str | None = None,
    ) -> Workflow:
        """Create a workflow from a graph (``{version, nodes, edges}``)."""
        body: dict[str, Any] = {"name": name, "graph": graph}
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if template_key is not None:
            body["template_key"] = template_key
        response = self._transport.request("POST", f"{_API}/", json=body)
        return self._parse(Workflow, response["workflow"])

    def update(
        self,
        workflow: str,
        *,
        name: str | None = None,
        readme: str | None = None,
        description: str | None = None,
        graph: dict[str, Any] | None = None,
        status: WorkflowStatus | None = None,
    ) -> Workflow:
        """Update a workflow's name / readme / graph / status.

        ``readme`` is the card the UI renders; ``description`` is deprecated in
        its favour and kept so existing callers keep working.
        """
        body: dict[str, Any] = {}
        if readme is not None:
            body["readme"] = readme
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if graph is not None:
            body["graph"] = graph
        if status is not None:
            body["status"] = status
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("PATCH", f"{_API}/{workflow_id}", json=body)
        return self._parse(Workflow, response["workflow"])

    def delete(self, workflow: str) -> None:
        """Delete a workflow, by NAME, and its run history."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        self._transport.request("DELETE", f"{_API}/{workflow_id}")

    def bulk_delete(self, workflows: Sequence[str]) -> BulkDeleteResult:
        """Delete many workflows in one atomic, org-scoped server-side call.

        Issues a single request the backend resolves with chunked,
        organization-scoped deletes, so it never fans out N calls. Requires the
        ``member``, ``admin``, or ``owner`` role (same as :meth:`delete`).

        Args:
            workflows: Names to delete (UUIDs are accepted too).
                Duplicates are ignored; ids that don't resolve in your
                organization are reported in
                :attr:`~pictograph.models.common.BulkDeleteResult.not_found`
                rather than raising, so a re-run still succeeds.

        Returns:
            A :class:`~pictograph.models.common.BulkDeleteResult`.

        Raises:
            ForbiddenError: Your API key role cannot manage workflows.
            ValidationError: ``workflows`` is empty.
        """
        response = self._transport.request(
            "POST",
            f"{_API}/bulk-delete",
            json={"workflow_ids": _resolve.workflow_ids(self._transport, workflows)},
        )
        return self._parse(BulkDeleteResult, response)

    def run(self, workflow: str) -> WorkflowRunCreated:
        """Validate + start a run. Raises ValidationError (400) if the graph is invalid,
        PaymentRequiredError (402) if there's not enough compute credit."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("POST", f"{_API}/{workflow_id}/run")
        return self._parse(WorkflowRunCreated, response)

    def get_run(self, run_id: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = self._transport.request("GET", f"{_API}/runs/{run_id}")
        return self._parse(WorkflowRun, response["run"])

    def cancel_run(self, run_id: str) -> None:
        """Cancel an in-flight run (stops the GPU job). The run is FREE - workflows
        bill on success only, so a cancelled run is never charged (no refund needed)."""
        self._transport.request("POST", f"{_API}/runs/{run_id}/cancel")

    def bulk_cancel_runs(self, run_ids: Sequence[str]) -> BulkActionResult:
        """Cancel many in-flight workflow runs in one org-scoped server-side call.

        One request the backend resolves per-item (each run stops its GPU job; a
        cancelled run is free, like :meth:`cancel_run`) instead of fanning out N
        calls. Idempotent: duplicate ids are collapsed, and any id that doesn't
        resolve in your org OR is already terminal is reported in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires the same role as :meth:`cancel_run` (member+).
        """
        response = self._transport.request(
            "POST", f"{_API}/runs/bulk-cancel", json={"run_ids": list(run_ids)}
        )
        return self._parse(BulkActionResult, response)

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> WorkflowRun:
        """Poll a run until it reaches a terminal status, then return it.

        Returns the run on ``completed``. Raises :class:`ApiError` if the run ends
        ``error`` or ``cancelled``, and :class:`PollTimeoutError` if ``timeout``
        elapses first (the run keeps going server-side - re-poll via :meth:`get_run`).

        Args:
            run_id: The run UUID from :meth:`run`.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 3600 = 1h, the runner's cap).
            sleep: Injectable sleep (tests pass a no-op); defaults to ``time.sleep``.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            run = self.get_run(run_id)
            if run.status == "completed":
                return run
            if run.status in ("error", "cancelled"):
                raise ApiError(
                    f"Workflow run {run_id} ended with status '{run.status}': "
                    f"{run.error or 'no error message provided'}",
                    response=run.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run_id} did not complete within {timeout:.0f}s "
                    f"(last status: {run.status}). Fetch later via "
                    f"client.workflows.get_run(...)."
                )
            sleep_fn(poll_interval)
