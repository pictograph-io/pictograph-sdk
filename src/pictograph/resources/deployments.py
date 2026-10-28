"""Deployments resource - manage org-level model inference deployments.

A deployment serves a trained model on a GPU (T4, L4 or A100) on PRT, Pictograph's realtime
engine: a TensorRT engine of the model, built for the deployment's GPU at its precision.
:meth:`Deployments.create` answers at once, ``provisioning``; the deployment turns ``active``
when its server is up - about a minute when the model's engine is already built for that GPU
and precision, a few minutes for a model's first deployment there, which builds it first
(``Deployment.engine_build`` shows the build). :meth:`Deployments.wait_until_active` waits for
that; then call the endpoint directly with
:class:`pictograph.resources._deployment_client.DeploymentClient` (see
:meth:`Deployments.connect`). Billing is by uptime and by the data its requests send through
its endpoint (per GiB), not per request; pause a deployment to stop billing, resume to bring
it back. A deployment has no request ceiling unless you set one (``rate_limit_rpm``).
:meth:`Deployments.update` changes its autoscaling, name, README and ceiling after it was
created (a live deployment takes a scaling change at once, with no redeploy).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pictograph._http.pagination import OffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.deployment import (
    CreatedDeployment,
    Deployment,
    DeploymentComputeOptions,
    DeploymentGpuType,
    DeploymentPrecision,
    DeploymentQuote,
    DeploymentStatus,
)
from pictograph.resources import _resolve
from pictograph.resources._base import Resource
from pictograph.resources._deployment_client import DEFAULT_TIMEOUT_S, DeploymentClient
from pictograph.resources.jobs import check_wait_args

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pictograph.inference.results import (
        AnyResult,
        ClassificationResult,
        DetectionResult,
        InstanceSegmentationResult,
        KeypointResult,
        SemanticSegmentationResult,
        TaskName,
    )

_API_PATH = "/api/v1/developer/deployments/"

#: How long :meth:`Deployments.wait_until_active` waits by default: a stored engine starts in
#: seconds, and a model's first engine on a GPU builds in a few minutes at most.
_DEFAULT_WAIT_TIMEOUT = 900.0
_DEFAULT_POLL_INTERVAL = 3.0


def not_coming_up(deployment: Deployment) -> ApiError | None:
    """The error :meth:`Deployments.wait_until_active` raises when waiting cannot make
    ``deployment`` active - it failed, or it was paused or terminated - else None.

    The deployment itself is the error's ``response`` (its ``model_dump``), so a caller reads
    its ``id``, ``status`` and ``error_message`` without another request.
    """
    status = deployment.status
    if status == "failed":
        return ApiError(
            f"Deployment '{deployment.name}' failed: "
            f"{deployment.error_message or 'no error message provided'}",
            response=deployment.model_dump(mode="json"),
            code=job_end_code("failed"),
            fix=(
                "Fix what the error names, then delete the deployment and create it again "
                "(a failed deployment cannot be resumed)."
            ),
        )
    if status in ("paused", "terminated"):
        fix = (
            "Resume it (client.deployments.resume(...)), then wait again."
            if status == "paused"
            else "Create a new deployment of the model."
        )
        return ApiError(
            f"Deployment '{deployment.name}' is {status}: waiting will not make it active.",
            response=deployment.model_dump(mode="json"),
            code=job_end_code("cancelled"),
            fix=fix,
        )
    return None


def still_provisioning(deployment: Deployment, timeout: float) -> PollTimeoutError:
    """The error :meth:`Deployments.wait_until_active` raises when its timeout runs out first."""
    build = deployment.engine_build
    building = (
        ", building its TensorRT engine" if build is not None and build.state == "building" else ""
    )
    return PollTimeoutError(
        f"Deployment '{deployment.name}' did not turn active within {timeout:.0f}s (last "
        f"status: {deployment.status}{building}). It is still provisioning - wait again with "
        "client.deployments.wait_until_active(...)."
    )


def create_body(
    model_id: str,
    *,
    name: str | None,
    region: str,
    gpu_type: str,
    precision: str,
    min_containers: int,
    max_containers: int,
    scaledown_window: int,
    rate_limit_rpm: int | None,
    inference_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """What :meth:`Deployments.create` sends (and its async twin): the optional values only when
    given."""
    body: dict[str, Any] = {
        "model_id": model_id,
        "region": region,
        "gpu_type": gpu_type,
        "precision": precision,
        "min_containers": min_containers,
        "max_containers": max_containers,
        "scaledown_window": scaledown_window,
        "inference_config": inference_config or {},
    }
    if name is not None:
        body["name"] = name
    if rate_limit_rpm is not None:
        body["rate_limit_rpm"] = rate_limit_rpm
    return body


def update_changes(
    min_containers: int | None,
    max_containers: int | None,
    scaledown_window: int | None,
    name: str | None = None,
    readme: str | None = None,
    rate_limit_rpm: int | None = None,
) -> dict[str, int | str]:
    """What :meth:`Deployments.update` sends: only the values given (``0`` and ``""`` are
    values, not "not given": an empty ``readme`` clears it, a ``rate_limit_rpm`` of ``0`` removes
    the ceiling). Raises ``ValueError`` when none is, before any request."""
    given: dict[str, int | str | None] = {
        "name": name,
        "readme": readme,
        "min_containers": min_containers,
        "max_containers": max_containers,
        "scaledown_window": scaledown_window,
        "rate_limit_rpm": rate_limit_rpm,
    }
    changes = {key: value for key, value in given.items() if value is not None}
    if not changes:
        raise ValueError(
            "Nothing to change: pass name, readme, min_containers, max_containers, "
            "scaledown_window or rate_limit_rpm."
        )
    return changes


class Deployments(Resource):
    """Operations on model deployments in your organization."""

    def list(
        self,
        *,
        model: str | None = None,
        name: str | None = None,
        status: DeploymentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Deployment]:
        """Single-page list of deployments in your organization.

        Args:
            model: Only deployments of this model, by NAME (an id also works).
            name: Exact deployment name.
            status: Only deployments in this state.
            limit: Page size.
            offset: Pagination offset.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if model is not None:
            params["model_id"] = _resolve.model_id(self._transport, model)
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Deployment, response.get("deployments", []))

    def iter(
        self,
        *,
        model: str | None = None,
        name: str | None = None,
        status: DeploymentStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> OffsetPager[Deployment]:
        """Auto-paging iterator across every deployment in your organization.

        Filters mirror :meth:`list`.
        """
        base: dict[str, Any] = {}
        if model is not None:
            base["model_id"] = _resolve.model_id(self._transport, model)
        if name is not None:
            base["name"] = name
        if status is not None:
            base["status"] = status

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]", self._transport.request("GET", _API_PATH, params=params)
            )

        return OffsetPager(
            fetch,
            items_key="deployments",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Deployment, raw),
        )

    def get(self, deployment: str) -> Deployment:
        """Fetch a single deployment by NAME (an id is accepted too)."""
        deployment_id = _resolve.deployment_id(self._transport, deployment)
        response = self._transport.request("GET", f"{_API_PATH}{deployment_id}")
        return self._parse(Deployment, response["deployment"])

    def compute_options(self, *, region: str = "auto") -> DeploymentComputeOptions:
        """What a deployment may run with: ``regions``, the regions it can run in (``region`` is
        the one these prices are for); ``options``, the GPU tiers (T4, L4, A100), each with its
        container (GPU memory, CPU cores, memory) and one container's price per minute and per
        hour there; and ``precisions``, each precision its TensorRT engine is built at (``fp32``,
        the default, first; then ``fp16``), in the server's words.

        Args:
            region: A region's id, or ``"auto"`` - the one nearest where this call is made.
        """
        response = self._transport.request(
            "GET", f"{_API_PATH}compute-options", params={"region": region}
        )
        return self._parse(DeploymentComputeOptions, response)

    def quote(
        self,
        *,
        gpu_type: DeploymentGpuType = "t4",
        min_containers: int = 0,
        max_containers: int = 1,
        region: str = "auto",
    ) -> DeploymentQuote:
        """What a deployment costs while it is up (marked-up micro-USD), BEFORE creating it.

        The arguments are :meth:`create`'s, so you can quote and then create with the same
        values. ``min_containers=0`` quotes scale-to-zero (billed only while a container is
        up); ``>=1`` keeps that many warm, billed continuously.
        ``cost_per_hour_micro_usd`` is the warm cost (one container's when it scales to zero)
        and ``max_cost_per_hour_micro_usd`` the cost with every container up to
        ``max_containers`` running. ``transfer_per_gib_micro_usd`` is the price of a GiB of its
        requests' data (each image sent through the endpoint and its answer), billed on top of
        the uptime.

        Args:
            gpu_type: ``"t4"`` (16 GB), ``"l4"`` (24 GB) or ``"a100"`` (40 GB).
            min_containers: Containers kept warm (``0`` scales to zero).
            max_containers: Autoscale ceiling.
            region: Where it would run: a region's id, or ``"auto"`` - the one nearest
                where this call is made. A region's price is its own.
        """
        params: dict[str, Any] = {
            "gpu_type": gpu_type,
            "min_containers": min_containers,
            "max_containers": max_containers,
            "region": region,
        }
        response = self._transport.request("GET", f"{_API_PATH}quote", params=params)
        return self._parse(DeploymentQuote, response["quote"])

    def create(
        self,
        model: str,
        *,
        name: str | None = None,
        region: str = "auto",
        gpu_type: DeploymentGpuType = "t4",
        precision: DeploymentPrecision = "fp32",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
        rate_limit_rpm: int | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> CreatedDeployment:
        """Deploy a trained model to a live inference endpoint, on a GPU.

        The deployment runs on PRT, Pictograph's realtime engine: a TensorRT engine of the
        model, built for ``gpu_type`` at ``precision``. It answers at once, ``provisioning``,
        and turns ``active`` when its server is up - about a minute when that engine is already
        built; a model's first deployment on a GPU at a precision builds it first, a few
        minutes (``deployment.engine_build`` reports the build). Wait with
        :meth:`wait_until_active`, then call the endpoint via :meth:`connect`.

        Returns the deployment plus its plaintext bearer token - store it securely; the API
        never returns it again (a member can reveal it from the deployment's Analytics in the
        app).

        Args:
            model: Trained model by name (status ``ready``); an id also works.
            name: Deployment name, unique among your organization's live deployments; the
                model's name when omitted.
            region: Where it runs - ``"us-east"`` or ``"us-west"``
                (``compute_options().regions``), or ``"auto"``: the region nearest where this
                call is made. Pick the region nearest the callers and cameras; it is set here
                and never changes.
            gpu_type: ``"t4"`` (16 GB), ``"l4"`` (24 GB) or ``"a100"`` (40 GB). There is no
                CPU tier.
            precision: ``"fp32"`` serves the model as trained; ``"fp16"`` is faster,
                accuracy-equivalent, not output-identical.
            min_containers: Containers kept warm (``0`` scales to zero; at most 5).
            max_containers: Autoscale ceiling (at most 10).
            scaledown_window: Idle seconds before a container scales down.
            rate_limit_rpm: Its request ceiling: requests a minute past which its endpoint
                answers 429 (``RateLimitError``, with ``Retry-After``). None, the default, is no
                ceiling: what it costs is bounded by ``max_containers`` and your budget.
            inference_config: Defaults for every request, such as ``{"confidence": 0.4}``.
        """
        body = create_body(
            _resolve.model_id(self._transport, model),
            name=name,
            region=region,
            gpu_type=gpu_type,
            precision=precision,
            min_containers=min_containers,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
            rate_limit_rpm=rate_limit_rpm,
            inference_config=inference_config,
        )
        response = self._transport.request("POST", _API_PATH, json=body)
        return self._parse(CreatedDeployment, response)

    def wait_until_active(
        self,
        deployment: str | Deployment,
        *,
        timeout: float = _DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        sleep: Callable[[float], None] | None = None,
    ) -> Deployment:
        """Poll a deployment until it is ``active``: its server is up and its endpoint answers.

        A deployment is ``provisioning`` after :meth:`create` and :meth:`resume`. It turns
        ``active`` in about a minute when its model's TensorRT engine is already built for its GPU
        and precision; a model's first deployment there builds the engine first, a few minutes
        (``engine_build`` reports the build, with an estimate when there is history).

        Args:
            deployment: Deployment name (an id works too), or the :class:`Deployment` that
                :meth:`create` or :meth:`resume` returned.
            timeout: Seconds to wait before raising ``PollTimeoutError``.
            poll_interval: Seconds between status checks.
            sleep: Override the sleep function (testing hook).

        Returns:
            The :class:`Deployment`, ``active``.

        Raises:
            ApiError: the deployment failed (``code="job_failed"``; its ``error_message`` is in
                the message), or it was paused or terminated while waiting
                (``code="job_cancelled"``). The deployment is the error's ``response``.
            PollTimeoutError: ``timeout`` elapsed first. The deployment keeps provisioning.
        """
        check_wait_args(poll_interval, timeout)
        deployment_id = (
            deployment.id
            if isinstance(deployment, Deployment)
            else _resolve.deployment_id(self._transport, deployment)
        )
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            response = self._transport.request("GET", f"{_API_PATH}{deployment_id}")
            current = self._parse(Deployment, response["deployment"])
            if current.status == "active":
                return current
            ended = not_coming_up(current)
            if ended is not None:
                raise ended
            if time.monotonic() >= deadline:
                raise still_provisioning(current, timeout)
            sleep_fn(poll_interval)

    def pause(self, deployment: str) -> Deployment:
        """Pause a deployment by NAME (stops compute + billing). An id works too."""
        deployment_id = _resolve.deployment_id(self._transport, deployment)
        response = self._transport.request("POST", f"{_API_PATH}{deployment_id}/pause")
        return self._parse(Deployment, response["deployment"])

    def resume(self, deployment: str) -> Deployment:
        """Resume a paused deployment by NAME (an id works too).

        It provisions again, on its model's current version: ``provisioning`` until its server
        is up, then ``active`` - wait with :meth:`wait_until_active`.
        """
        deployment_id = _resolve.deployment_id(self._transport, deployment)
        response = self._transport.request("POST", f"{_API_PATH}{deployment_id}/resume")
        return self._parse(Deployment, response["deployment"])

    def update(
        self,
        deployment: str,
        *,
        min_containers: int | None = None,
        max_containers: int | None = None,
        scaledown_window: int | None = None,
        name: str | None = None,
        readme: str | None = None,
        rate_limit_rpm: int | None = None,
    ) -> Deployment:
        """Change a deployment after it was created, by NAME (an id works too): its name, its
        README, its autoscaling, its request ceiling.

        Only what you pass changes. An ``active`` deployment takes a scaling change at once,
        with no redeploy and no restart; a ``provisioning`` one takes it when it is up, a
        ``paused`` one when it is resumed. A rename keeps its endpoint and its token; a ceiling
        reaches its endpoint within half a minute. Its region, GPU and precision cannot change
        this way: deploy again.

        Args:
            deployment: Deployment name (an id works too).
            min_containers: Containers kept running, and billed, whether or not requests come
                (at most 5). ``0`` lets it scale to zero after ``scaledown_window`` idle
                seconds: nothing is billed while it sleeps, and the next request wakes it.
            max_containers: How many containers it scales up to under load (at most 10, and
                never fewer than ``min_containers``).
            scaledown_window: Idle seconds before a container scales down (2 to 3600).
            name: Its new name (unique among the organization's live deployments; at most 100
                characters, no ``/``).
            readme: Its README, Markdown (at most 20000 characters); ``""`` clears it.
            rate_limit_rpm: Requests a minute past which its endpoint answers 429
                (``RateLimitError``); ``0`` removes the ceiling (none is the default).

        Returns:
            The :class:`Deployment`, as it is now.

        Raises:
            ValueError: nothing was given to change; no request is sent.
            ValidationError: a value out of its bounds, or a ``max_containers`` below the
                ``min_containers`` (the one you pass, or the deployment's own).
            PaymentRequiredError: ``min_containers`` was raised (more containers kept running,
                billed continuously) and the organization has no spendable compute credit.
            ConflictError: another live deployment of the organization has that ``name``; the
                deployment is ``failed`` or ``terminated``, so its scaling cannot change; or it
                changed while this was saved (try again).
            ServerError: the live deployment's server refused the change, so nothing was
                changed (the SDK retries that ``502`` first).
        """
        body = update_changes(
            min_containers, max_containers, scaledown_window, name, readme, rate_limit_rpm
        )
        deployment_id = _resolve.deployment_id(self._transport, deployment)
        response = self._transport.request("PATCH", f"{_API_PATH}{deployment_id}", json=body)
        return self._parse(Deployment, response["deployment"])

    def delete(self, deployment: str) -> None:
        """Terminate a deployment by NAME and tear down its serving endpoint."""
        deployment_id = _resolve.deployment_id(self._transport, deployment)
        self._transport.request("DELETE", f"{_API_PATH}{deployment_id}")

    def bulk_pause(self, deployments: Sequence[str]) -> BulkActionResult:
        """Pause many deployments in one org-scoped server-side call.

        One request the backend resolves per item (each deployment's server stopped,
        exactly like :meth:`pause`), instead of fanning out N :meth:`pause` calls.
        Idempotent: duplicate ids are collapsed, and any id that doesn't resolve
        in your org or isn't in a pausable state is reported in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires the same role as :meth:`pause` (member+).
        """
        response = self._transport.request(
            "POST",
            f"{_API_PATH}bulk-pause",
            json={"deployment_ids": _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    def bulk_resume(self, deployments: Sequence[str]) -> BulkActionResult:
        """Resume many paused deployments in one org-scoped server-side call.

        Provisions each deployment again, exactly like :meth:`resume` (each is
        ``provisioning`` until its server is up). An id that can't resume - not paused,
        no compute credit, its engine unavailable - is reported in ``not_found``
        instead of aborting the batch. Requires the same role as :meth:`resume`
        (member+). Deployments run on every plan, Free included.
        """
        response = self._transport.request(
            "POST",
            f"{_API_PATH}bulk-resume",
            json={"deployment_ids": _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    def bulk_delete(self, deployments: Sequence[str]) -> BulkDeleteResult:
        """Delete (terminate) many deployments in one org-scoped server-side call.

        Stops each deployment's server (its last minutes are billed as they are
        metered), drops its mirrored workflows key and marks it terminated - exactly
        like :meth:`delete`, per item. Idempotent: an already-terminated or missing id
        is reported in ``not_found``. Requires the same role as :meth:`delete`.
        """
        response = self._transport.request(
            "POST",
            f"{_API_PATH}bulk-delete",
            json={"deployment_ids": _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkDeleteResult, response)

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["object_detection"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[DetectionResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["instance_segmentation"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[InstanceSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["semantic_segmentation"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[SemanticSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["keypoint_detection"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[KeypointResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["classification"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[ClassificationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[AnyResult]: ...

    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: TaskName | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[Any]:
        """Build a direct :class:`DeploymentClient` for an active deployment.

        Pass ``task=`` (the deployed model's ``model_type``) to narrow what
        :meth:`DeploymentClient.infer` returns to that task's result class,
        exactly as ``task=`` does on :func:`pictograph.get_model`. Without it the
        client is un-narrowed and ``infer`` returns
        :data:`~pictograph.inference.results.AnyResult`; either way the runtime
        object is the correct result class for whatever the endpoint serves.

        A deployment has its ``endpoint_url`` from the moment it is created, but the
        endpoint answers only once the deployment is ``active`` (a ``409`` before that):
        wait with :meth:`wait_until_active` first. ``timeout`` is per request; the
        default outlasts the gateway's hold on the first request to a scaled-to-zero
        deployment while its container starts.
        """
        if not deployment.endpoint_url:
            raise ValueError(
                "Deployment has no endpoint_url to connect to; fetch it again with "
                "client.deployments.get(...)."
            )
        return DeploymentClient(
            deployment.endpoint_url,
            api_key,
            task=cast("Any", task),
            timeout=timeout,
        )
