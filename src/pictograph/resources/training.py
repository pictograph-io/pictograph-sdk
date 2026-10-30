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
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.common import BulkActionResult
from pictograph.models.training import (
    GpuType,
    PipelineType,
    TrainingQuote,
    TrainingRun,
    TrainingStatus,
)
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_API_PATH = "/api/v1/developer/training/"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_TIMEOUT = 7200.0  # 2h - the training service's own hard cap
_TERMINAL_STATUSES: frozenset[TrainingStatus] = frozenset({"completed", "failed", "cancelled"})


def _single_path(run: str, suffix: str = "") -> str:
    """The path of one run: ``/{run}`` takes its name or its UUID in the same
    segment (the API resolves either), so one argument addresses it (REFERENCE_LANGUAGE.md
    R1: never two ways to say one thing). Shared with the async twin."""
    return f"{_API_PATH}{quote(run, safe='')}{suffix}"


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


def _plan_body(
    dataset: str,
    export: str,
    pipeline_type: str,
    config: dict[str, Any] | str | Path | None,
    gpu_type: str,
    version_of: str | None,
) -> dict[str, Any]:
    """The fields that decide what a run trains on - one body for ``create`` and
    ``quote``, so a quote is asked about exactly the run ``create`` would start."""
    body: dict[str, Any] = {
        "dataset": dataset,
        "export": export,
        "pipeline_type": pipeline_type,
        "gpu_type": gpu_type,
        "config": _load_config(config),
    }
    if version_of is not None:
        body["version_of"] = version_of
    return body


class Training(Resource):
    """Operations on training runs."""

    def create(
        self,
        dataset: str,
        export: str,
        *,
        pipeline_type: PipelineType,
        name: str,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        version_of: str | None = None,
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
            dataset: Dataset name within your organization.
            export: Completed export's name (within the dataset).
            pipeline_type: ``"yolox"``, ``"sm_pytorch"``,
                ``"classification"``, ``"rfdetr_detection"``, or
                ``"rfdetr_segmentation"``, or ``"rfdetr_keypoint"``.
            name: Human-readable label for this run (1-100 chars).
            config: Pipeline-specific hyperparameters: ``epochs``,
                ``batch_size``, ``learning_rate``, ``model_size`` (the RF-DETR / YOLOX
                variant), ``image_width`` / ``image_height`` (the input size), etc. A key
                left out takes the default the web app uses: RF-DETR detection and
                segmentation ``medium``, keypoint ``preview``, YOLOX ``m``; YOLOX trains at
                640 px, semantic segmentation 512 px, classification at its backbone's
                pretrained size (224; EfficientNet B1-B4 240 / 260 / 300 / 380), RF-DETR at
                its variant's (the run records the size it trains at). Also accepts a path
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
                decision into ``config["gpu_autoselect"]``. Training runs on
                ONE GPU, always.
            version_of: Append the result to this EXISTING model as
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
            NotFoundError: ``dataset`` or ``export`` doesn't exist.
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
        body = _plan_body(dataset, export, pipeline_type, config, gpu_type, version_of)
        body["name"] = name
        response = self._transport.request("POST", _API_PATH, json=body)
        run = self._parse(TrainingRun, response["data"])
        if not wait:
            return run
        return self.wait_for_completion(run.id, poll_interval=poll_interval, timeout=timeout)

    def quote(
        self,
        dataset: str,
        export: str,
        *,
        pipeline_type: PipelineType,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        version_of: str | None = None,
    ) -> TrainingQuote:
        """Check and price a training run without starting it.

        Takes :meth:`create`'s arguments (no ``name``) and runs every check ``create``
        runs on them, so it raises what ``create`` would raise (a missing dataset or
        export, an export not completed or empty, a config the pipeline cannot train,
        no class it could train) - and a quote that comes back describes a run
        ``create`` would accept, apart from what ``blocking`` names: too little
        balance to start, an API key below the member role, or a key whose creator
        was removed. Nothing is written and nothing is charged; any role may ask.

        Args:
            dataset: Dataset name within your organization.
            export: Completed export's name (within the dataset).
            pipeline_type: The training pipeline, as for :meth:`create`.
            config: Hyperparameters, a dict or a path to a JSON file (``config.json``
                round-trips), as for :meth:`create`.
            gpu_type: ``"a10g"`` (default), ``"a100"``, ``"h100"`` or ``"auto"``. The
                quote always reports what ``"auto"`` would pick
                (``recommended_gpu_type``) and whether the config's predicted peak
                memory fits each tier (``tiers``).
            version_of: As for :meth:`create`; checked the same way.

        Returns:
            :class:`TrainingQuote`: the tier the run would use, the per-class training
            support, the estimate the app's training dialog shows (``estimate`` - an
            estimate only: a run is billed once, when it succeeds, for the GPU time it
            used), the balance, ``can_start`` and ``blocking``, and the config the run
            would be stored with.
        """
        body = _plan_body(dataset, export, pipeline_type, config, gpu_type, version_of)
        response = self._transport.request("POST", f"{_API_PATH}quote", json=body)
        return self._parse(TrainingQuote, response["data"])

    def list(
        self,
        *,
        dataset: str | None = None,
        status: TrainingStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TrainingRun]:
        """Single-page list of training runs in your organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset is not None:
            params["dataset"] = dataset
        if status is not None:
            params["status"] = status
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(TrainingRun, response.get("data", []))

    def iter(
        self,
        *,
        dataset: str | None = None,
        status: TrainingStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> OffsetPager[TrainingRun]:
        """Auto-paging iterator across every training run in your org."""
        base: dict[str, Any] = {}
        if dataset is not None:
            base["dataset"] = dataset
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

    def get(
        self,
        run: str,
    ) -> TrainingRun:
        """Fetch a training run's status, metrics, and progress.

        Address it by name (positional) or ``run`` UUID. Because run names
        are not unique, a by-name lookup returns the **most recent** run of
        that name.
        """
        response = self._transport.request("GET", _single_path(run))
        return self._parse(TrainingRun, response["data"])

    def cancel(
        self,
        run: str,
    ) -> TrainingRun:
        """Cancel a running training job.

        The backend stops the GPU job in-flight - it does not merely mark the
        row - and atomically marks the run ``cancelled``. Under charge-on-success a run
        cancelled before completion is never charged; a legacy pre-charged run is
        refunded once. By-name cancels the most recent run of that name. Requires
        a member+ API key.
        """
        response = self._transport.request("POST", _single_path(run, "/cancel"))
        return self._parse(TrainingRun, response["data"])

    def bulk_cancel(self, runs: Sequence[str]) -> BulkActionResult:
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
            "POST", f"{_API_PATH}bulk-cancel", json={"runs": list(runs)}
        )
        return self._parse(BulkActionResult, response.get("data", response))

    def wait_for_completion(
        self,
        run: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> TrainingRun:
        """Poll a training current until it reaches a terminal status.

        Args:
            run: The run's name (its UUID also works).
            poll_interval: Seconds between checks (default 5s).
            timeout: Maximum seconds to wait (default 7200 = 2h).
            sleep: Override the sleep function (testing hook).

        Returns:
            The :class:`TrainingRun` in ``completed`` status.

        Raises:
            ApiError: The current reached ``failed`` or ``cancelled`` status.
                ``error_message`` is in the exception's response.
            PollTimeoutError: ``timeout`` elapsed before completion. The
                current is still executing server-side.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self.get(run)
            if current.status == "completed":
                return current
            if current.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Training run '{current.name}' ended with status "
                    f"'{current.status}': {current.error_message or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Training run '{current.name}' did not complete within "
                    f"{timeout:.0f}s (last status: {current.status}). The run is "
                    f"still executing - fetch later via client.training.get(...)."
                )
            sleep_fn(poll_interval)

    # ───────────── train straight off a dataset ─────────────
