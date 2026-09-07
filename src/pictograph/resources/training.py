"""Training resource - create, list, get, cancel, wait_for_completion.

Training is asynchronous: :meth:`Training.create` queues a GPU job and
returns immediately. The default ``wait=True`` polls until the run reaches
a terminal status (``completed`` / ``failed`` / ``cancelled``). Pass
``wait=False`` to fire-and-forget and poll later via
:meth:`Training.wait_for_completion`.

Credit handling is fully server-side and **charge-on-success**: the backend
gates the start on a minimum spendable balance ($2.00) and then charges ONCE,
after the run succeeds, for the actual GPU minutes used - failed / OOM /
cancelled runs are never charged (no pre-charge, no refund dance). Too low a
balance surfaces as :class:`PaymentRequiredError` (HTTP 402) before the job
is submitted.

Runs are addressed by their UUID or by name; because run names are not unique,
a by-name lookup resolves to the **most recent** run of that name.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from pictograph._http.pagination import OffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkActionResult
from pictograph.models.training import GpuType, PipelineType, TrainingRun, TrainingStatus
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_API_PATH = "/api/v1/developer/training/"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_TIMEOUT = 7200.0  # 2h - the training service's own hard cap
_TERMINAL_STATUSES: frozenset[TrainingStatus] = frozenset({"completed", "failed", "cancelled"})


def _single_path(name: str | None, run_id: str | None, suffix: str = "") -> str:
    """Resolve the by-name vs by-uuid path form. Exactly one of name/run_id.

    Mirrors ``models._single_path`` so the addressing contract is uniform.
    """
    if (name is None) == (run_id is None):
        raise ValueError("Pass exactly one of `name` (positional) or `run_id=`.")
    # ONE segment for both forms - `/training/{run}` takes a name or a UUID.
    # The `/by-name/` prefix was removed from the API and 404s.
    base = f"{_API_PATH}{quote(name, safe='')}" if name is not None else f"{_API_PATH}{run_id}"
    return f"{base}{suffix}"


def _load_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    """Normalize ``create``'s config input to the dict the wire expects.

    A ``str`` / ``Path`` is a filesystem path to a JSON file (the
    ``config.json`` round-trip) - read, parsed, and passed through the
    SAME request field as a dict; there is deliberately no divergent code
    path. Shared with the async twin.
    """
    if isinstance(config, (str, Path)):
        parsed = json.loads(Path(config).expanduser().read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Config file {config} must contain a JSON object")
        return cast("dict[str, Any]", parsed)
    return config or {}


class Training(Resource):
    """Operations on training runs."""

    def create(
        self,
        dataset_name: str,
        export_name: str,
        *,
        pipeline_type: PipelineType,
        name: str,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        gpu_count: int = 1,
        version_of_model_id: str | None = None,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> TrainingRun:
        """Spawn a new training run.

        The backend looks up dataset and export by name within your
        organization, validates the export is ``completed``, gates on a
        minimum spendable balance, and submits the GPU job (charged once on
        success for actual GPU minutes). Returns the :class:`TrainingRun` -
        pending if ``wait=False``, terminal status otherwise.

        Args:
            dataset_name: Dataset name within your organization.
            export_name: Completed export's name (within the dataset).
            pipeline_type: ``"yolox"``, ``"sm_pytorch"``,
                ``"classification"``, ``"rfdetr_detection"``, or
                ``"rfdetr_segmentation"``, or ``"rfdetr_keypoint"``.
            name: Human-readable label for this run (1-100 chars).
            config: Pipeline-specific hyperparameters: ``epochs``,
                ``batch_size``, ``learning_rate``, ``model_type``, etc.
                Empty dict uses pipeline defaults. Also accepts a path
                (``str`` or :class:`~pathlib.Path`) to a JSON file - most
                usefully a ``config.json`` downloaded via
                :meth:`Models.download_file`: the file is read, parsed, and
                sent through the same field, closing the reproducibility
                round-trip (the backend unwraps the ``_pictograph`` envelope
                and 400s on a pipeline mismatch).

                Include ``class_overrides`` - a ``{source: target}`` map - to
                remap classes at TRAIN time. ``config={"class_overrides":
                {"bus": "truck"}}`` trains every ``bus`` annotation AS ``truck``,
                collapsing the model's class set (e.g. ``bus``/``car``/``truck``/
                ``person`` -> ``truck``/``car``/``person``); your stored
                annotations are never modified. Every target must be a selected
                class; the effective (collapsed) list is what the model records
                in ``class_mapping``. Omit it (or ``{}``) for no remap.
            gpu_type: ``"a10g"`` (default), ``"a100"``, ``"h100"``, or
                ``"auto"`` - the backend picks the cheapest tier whose VRAM
                fits the config's predicted peak, instrumenting the
                decision into ``config["gpu_autoselect"]``.
            gpu_count: GPUs in the training container (1-4, RF-DETR
                pipelines only). ``>1`` trains with DDP and bills
                ``gpu_count x`` the per-second GPU rate.
            version_of_model_id: Append the result to this EXISTING model as
                a new version instead of minting a new model row. The
                target must be in your org and share the pipeline's task
                type; the new version goes live unless the model is pinned
                (see :meth:`Models.set_current_version`).
            wait: When ``True`` (default), poll until the run reaches a
                terminal status. When ``False``, return after the spawn.
            poll_interval: Seconds between status checks when waiting.
                Defaults to 5s - training is slow, frequent polling wastes
                requests against your rate limit.
            timeout: Maximum seconds to wait. Defaults to 2 hours, matching
                the training service's own hard timeout.

        Returns:
            The :class:`TrainingRun` (pending if ``wait=False``, terminal
            otherwise).

        Raises:
            NotFoundError: ``dataset_name`` or ``export_name`` doesn't exist.
            ValidationError: Export is not in ``completed`` status, GPU
                type invalid, or pipeline type invalid.
            PaymentRequiredError: The org is below the minimum spendable
                balance to start training. The 402 body carries ``required``
                and ``remaining`` (micro-USD) for agent decision-making.
            ApiError: The training run was spawned but reached ``failed``
                status. ``error_message`` is in the exception's response.
            PollTimeoutError: ``timeout`` elapsed before the run reached a
                terminal status. The run is still executing server-side -
                fetch its status later via :meth:`get`.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "export_name": export_name,
            "pipeline_type": pipeline_type,
            "name": name,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "config": _load_config(config),
        }
        if version_of_model_id is not None:
            body["version_of_model_id"] = version_of_model_id
        response = self._transport.request("POST", _API_PATH, json=body)
        run = self._parse(TrainingRun, response["data"])
        if not wait:
            return run
        return self.wait_for_completion(run.id, poll_interval=poll_interval, timeout=timeout)

    def list(
        self,
        *,
        dataset_name: str | None = None,
        status: TrainingStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TrainingRun]:
        """Single-page list of training runs in your organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        if status is not None:
            params["status"] = status
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(TrainingRun, response.get("data", []))

    def iter(
        self,
        *,
        dataset_name: str | None = None,
        status: TrainingStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> OffsetPager[TrainingRun]:
        """Auto-paging iterator across every training run in your org."""
        base: dict[str, Any] = {}
        if dataset_name is not None:
            base["dataset_name"] = dataset_name
        if status is not None:
            base["status"] = status

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                self._transport.request("GET", _API_PATH, params=params),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(TrainingRun, raw),
        )

    def get(self, name: str | None = None, *, run_id: str | None = None) -> TrainingRun:
        """Fetch a training run's status, metrics, and progress.

        Address it by name (positional) or ``run_id=`` UUID. Because run names
        are not unique, a by-name lookup returns the **most recent** run of
        that name.
        """
        response = self._transport.request("GET", _single_path(name, run_id))
        return self._parse(TrainingRun, response["data"])

    def cancel(self, name: str | None = None, *, run_id: str | None = None) -> TrainingRun:
        """Cancel a running training job (by name or ``run_id=`` UUID).

        The backend stops the GPU job in-flight - it does not merely mark the
        row - and atomically marks the run ``cancelled``. Under charge-on-success a run
        cancelled before completion is never charged; a legacy pre-charged run is
        refunded once. By-name cancels the most recent run of that name. Requires
        a member+ API key.
        """
        response = self._transport.request("POST", _single_path(name, run_id, "/cancel"))
        return self._parse(TrainingRun, response["data"])

    def bulk_cancel(self, run_ids: Sequence[str]) -> BulkActionResult:
        """Cancel many training runs in one org-scoped server-side call.

        One request the backend resolves in org-scoped chunks instead of fanning
        out N :meth:`cancel` calls. Each run is stopped in-flight; a
        cancelled-before-completion run is never charged (a legacy pre-charged
        run is refunded once). Idempotent: duplicate ids are collapsed, and any
        id that doesn't resolve in your org OR is already terminal is reported in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires the same role as :meth:`cancel` (member+).
        """
        response = self._transport.request(
            "POST", f"{_API_PATH}bulk-cancel", json={"run_ids": list(run_ids)}
        )
        return self._parse(BulkActionResult, response.get("data", response))

    def wait_for_completion(
        self,
        name: str | None = None,
        *,
        run_id: str | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> TrainingRun:
        """Poll a training run until it reaches a terminal status.

        Args:
            name: The run's name (positional). A UUID works here too.
            run_id: The run's UUID - the keyword alternative to ``name``.
            poll_interval: Seconds between checks (default 5s).
            timeout: Maximum seconds to wait (default 7200 = 2h).
            sleep: Override the sleep function (testing hook).

        Returns:
            The :class:`TrainingRun` in ``completed`` status.

        Raises:
            ApiError: The run reached ``failed`` or ``cancelled`` status.
                ``error_message`` is in the exception's response.
            PollTimeoutError: ``timeout`` elapsed before completion. The
                run is still executing server-side.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            run = self.get(name, run_id=run_id)
            if run.status == "completed":
                return run
            if run.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Training run '{run.name}' ended with status "
                    f"'{run.status}': {run.error_message or 'no error message provided'}",
                    response=run.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Training run '{run.name}' did not complete within "
                    f"{timeout:.0f}s (last status: {run.status}). The run is "
                    f"still executing - fetch later via client.training.get(...)."
                )
            sleep_fn(poll_interval)

    # ───────────── train straight off a dataset ─────────────
