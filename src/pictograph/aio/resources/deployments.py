"""Async Deployments resource - manage org-level model inference deployments.

Async twin of :class:`pictograph.resources.deployments.Deployments`; see that module for
how a deployment provisions (``provisioning``, then ``active``) on PRT. :meth:`connect`
still returns the standalone (sync) :class:`DeploymentClient` - calling a deployed
model's ``/predict`` endpoint is a separate transport with its own ergonomics.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.aio.resources import _resolve
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
from pictograph.resources._base import AsyncResource
from pictograph.resources._deployment_client import DEFAULT_TIMEOUT_S, DeploymentClient
from pictograph.resources.deployments import (
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_WAIT_TIMEOUT,
    create_body,
    not_coming_up,
    still_provisioning,
    update_changes,
)
from pictograph.resources.jobs import check_wait_args

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

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


class AsyncDeployments(AsyncResource):
    """Operations on model deployments in your organization (async)."""

    async def list(
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
            params["model"] = await _resolve.model_id(self._transport, model)
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Deployment, response.get("deployments", []))

    def iter(
        self,
        *,
        model: str | None = None,
        name: str | None = None,
        status: DeploymentStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[Deployment]:
        """Auto-paging async iterator across every deployment in your organization.

        Filters mirror :meth:`list`. ``model`` is resolved on the first page, for
        the same reason :meth:`AsyncImages.iter` resolves there.
        """
        base: dict[str, Any] = {}
        if name is not None:
            base["name"] = name
        if status is not None:
            base["status"] = status
        resolved_model: str | None = None

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            nonlocal resolved_model
            if model is not None and resolved_model is None:
                resolved_model = await _resolve.model_id(self._transport, model)
            if resolved_model is not None:
                base["model"] = resolved_model
            params = {**base, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="deployments",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Deployment, raw),
        )

    async def get(self, deployment: str) -> Deployment:
        """Fetch a single deployment by NAME (an id is accepted too)."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("GET", f"{_API_PATH}{deployment_id}")
        return self._parse(Deployment, response["deployment"])

    async def compute_options(self, *, region: str = "auto") -> DeploymentComputeOptions:
        """What a deployment may run with: ``regions`` (``region`` is the one these prices are
        for), ``options``, the GPU tiers (T4, L4, A100) with their container and price there,
        and ``precisions``, each precision its engine is built at (the default first).
        ``region``: a region's id, or ``"auto"`` - the one nearest where this call is made."""
        response = await self._transport.request(
            "GET", f"{_API_PATH}compute-options", params={"region": region}
        )
        return self._parse(DeploymentComputeOptions, response)

    async def quote(
        self,
        *,
        gpu_type: DeploymentGpuType = "t4",
        min_containers: int = 0,
        max_containers: int = 1,
        region: str = "auto",
    ) -> DeploymentQuote:
        """What a deployment costs while it is up (µUSD), BEFORE creating it.

        The arguments are :meth:`create`'s. ``min_containers=0`` quotes scale-to-zero;
        ``>=1`` quotes a warm deployment; ``max_cost_per_hour_micro_usd`` is every
        container up to ``max_containers`` running. ``region``: where it would run (a
        region's id, or ``"auto"``); a region's price is its own.
        """
        params: dict[str, Any] = {
            "gpu_type": gpu_type,
            "min_containers": min_containers,
            "max_containers": max_containers,
            "region": region,
        }
        response = await self._transport.request("GET", f"{_API_PATH}quote", params=params)
        return self._parse(DeploymentQuote, response["quote"])

    async def create(
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

        It answers at once, ``provisioning``, and turns ``active`` when its server is up:
        about a minute when the model's TensorRT engine is already built for ``gpu_type`` at
        ``precision``, a few minutes for a model's first deployment there
        (``deployment.engine_build`` reports the build). Wait with
        :meth:`wait_until_active`.

        ``region`` is where it runs - ``"us-east"`` or ``"us-west"``, or ``"auto"``: the region
        nearest where this call is made - set here and never changed. ``rate_limit_rpm`` is its
        request ceiling (requests a minute; none unless given).

        Returns the deployment plus its plaintext bearer token - store it securely; the API
        never returns it again (a member can reveal it from the deployment's Analytics in the
        app).
        """
        body = create_body(
            await _resolve.model_id(self._transport, model),
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
        response = await self._transport.request("POST", _API_PATH, json=body)
        return self._parse(CreatedDeployment, response)

    async def wait_until_active(
        self,
        deployment: str | Deployment,
        *,
        timeout: float = _DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> Deployment:
        """Poll a deployment until it is ``active`` (async).

        Mirrors :meth:`pictograph.resources.deployments.Deployments.wait_until_active`.

        Raises:
            ApiError: the deployment failed (``code="job_failed"``), or it was paused or
                terminated while waiting (``code="job_cancelled"``).
            PollTimeoutError: ``timeout`` elapsed first. The deployment keeps provisioning.
        """
        check_wait_args(poll_interval, timeout)
        deployment_id = (
            deployment.id
            if isinstance(deployment, Deployment)
            else await _resolve.deployment_id(self._transport, deployment)
        )
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            response = await self._transport.request("GET", f"{_API_PATH}{deployment_id}")
            current = self._parse(Deployment, response["deployment"])
            if current.status == "active":
                return current
            ended = not_coming_up(current)
            if ended is not None:
                raise ended
            if time.monotonic() >= deadline:
                raise still_provisioning(current, timeout)
            await sleep_fn(poll_interval)

    async def pause(self, deployment: str) -> Deployment:
        """Pause a deployment (stops compute + billing)."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("POST", f"{_API_PATH}{deployment_id}/pause")
        return self._parse(Deployment, response["deployment"])

    async def resume(self, deployment: str) -> Deployment:
        """Resume a paused deployment: it provisions again (``provisioning``, then
        ``active`` - wait with :meth:`wait_until_active`)."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("POST", f"{_API_PATH}{deployment_id}/resume")
        return self._parse(Deployment, response["deployment"])

    async def update(
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
        """Change a deployment after it was created: its name, its README, its autoscaling, its
        request ceiling (only what you pass changes; a ``rate_limit_rpm`` of ``0`` removes it).

        An ``active`` deployment takes a scaling change at once, with no redeploy; a
        ``provisioning`` one when it is up, a ``paused`` one when it is resumed. Mirrors
        :meth:`pictograph.resources.deployments.Deployments.update`, which raises the same
        ``ValueError`` (before any request) when nothing is given.
        """
        body = update_changes(
            min_containers, max_containers, scaledown_window, name, readme, rate_limit_rpm
        )
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("PATCH", f"{_API_PATH}{deployment_id}", json=body)
        return self._parse(Deployment, response["deployment"])

    async def delete(self, deployment: str) -> None:
        """Terminate a deployment and tear down its serving endpoint."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        await self._transport.request("DELETE", f"{_API_PATH}{deployment_id}")

    async def bulk_pause(self, deployments: Sequence[str]) -> BulkActionResult:
        """Pause many deployments in one org-scoped server-side call (idempotent)."""
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}bulk-pause",
            json={"deployments": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    async def bulk_resume(self, deployments: Sequence[str]) -> BulkActionResult:
        """Resume many paused deployments in one org-scoped server-side call.

        Provisions each again, like :meth:`resume`. An id that can't resume lands in
        ``not_found`` instead of aborting the batch. Deployments run on every plan, Free
        included.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}bulk-resume",
            json={"deployments": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    async def bulk_delete(self, deployments: Sequence[str]) -> BulkDeleteResult:
        """Delete (terminate) many deployments in one org-scoped server-side call."""
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}bulk-delete",
            json={"deployments": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkDeleteResult, response)

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: Literal["object_detection"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[DetectionResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: Literal["instance_segmentation"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[InstanceSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: Literal["semantic_segmentation"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[SemanticSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: Literal["keypoint_detection"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[KeypointResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: Literal["classification"],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[ClassificationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[AnyResult]: ...

    @staticmethod
    def connect(
        deployment: Deployment,
        *,
        api_key: str,
        task: TaskName | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> DeploymentClient[Any]:
        """Build a direct (sync) :class:`DeploymentClient` for an active deployment.

        Pass ``task=`` (the deployed model's ``model_type``) to narrow what
        :meth:`DeploymentClient.infer` returns to that task's result class,
        exactly as ``task=`` does on :func:`pictograph.get_model`. Without it the
        client is un-narrowed and ``infer`` returns
        :data:`~pictograph.inference.results.AnyResult`; either way the runtime
        object is the correct result class for whatever the endpoint serves.

        The endpoint answers only once the deployment is ``active`` (a ``409`` before
        that): wait with :meth:`wait_until_active` first.
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
