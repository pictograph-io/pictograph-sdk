"""Async Deployments resource - manage org-level model inference deployments.

Async twin of :class:`pictograph.resources.deployments.Deployments`. :meth:`connect`
still returns the standalone (sync) :class:`DeploymentClient` - calling a deployed
model's ``/predict`` endpoint is a separate transport with its own ergonomics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.aio.resources import _resolve
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.deployment import (
    ComputeType,
    CreatedDeployment,
    Deployment,
    DeploymentComputeOption,
    DeploymentGpuType,
    DeploymentQuote,
    DeploymentStatus,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources._deployment_client import DeploymentClient

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
            params["model_id"] = await _resolve.model_id(self._transport, model)
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
                base["model_id"] = resolved_model
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

    async def compute_options(self) -> Sequence[DeploymentComputeOption]:
        """List selectable compute tiers (CPU / GPU types) with per-minute rate."""
        response = await self._transport.request("GET", f"{_API_PATH}compute-options")
        return self._parse_list(DeploymentComputeOption, response.get("options", []))

    async def quote(
        self,
        *,
        compute_type: ComputeType = "gpu",
        gpu_type: DeploymentGpuType | None = None,
        min_containers: int = 0,
    ) -> DeploymentQuote:
        """Cost quote (marked-up µUSD) for a deployment BEFORE creating it.

        ``min_containers=0`` quotes scale-to-zero; ``>=1`` quotes a warm deployment.
        """
        params: dict[str, Any] = {"compute_type": compute_type, "min_containers": min_containers}
        if gpu_type is not None:
            params["gpu_type"] = gpu_type
        response = await self._transport.request("GET", f"{_API_PATH}quote", params=params)
        return self._parse(DeploymentQuote, response["quote"])

    async def create(
        self,
        model: str,
        *,
        name: str | None = None,
        compute_type: ComputeType = "gpu",
        gpu_type: DeploymentGpuType | None = "t4",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
        inference_config: dict[str, Any] | None = None,
    ) -> CreatedDeployment:
        """Deploy a trained model to a live inference endpoint.

        Returns the deployment plus a one-time plaintext bearer token - store it
        securely; it is never retrievable again. The deployment starts in
        ``provisioning``; poll :meth:`get` until ``status == "active"``.
        """
        body: dict[str, Any] = {
            "model_id": await _resolve.model_id(self._transport, model),
            "compute_type": compute_type,
            "gpu_type": gpu_type if compute_type == "gpu" else None,
            "min_containers": min_containers,
            "max_containers": max_containers,
            "scaledown_window": scaledown_window,
            "inference_config": inference_config or {},
        }
        if name is not None:
            body["name"] = name
        response = await self._transport.request("POST", _API_PATH, json=body)
        return self._parse(CreatedDeployment, response)

    async def pause(self, deployment: str) -> Deployment:
        """Pause a deployment (stops compute + billing)."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("POST", f"{_API_PATH}{deployment_id}/pause")
        return self._parse(Deployment, response["deployment"])

    async def resume(self, deployment: str) -> Deployment:
        """Resume a paused deployment (re-provisions the endpoint)."""
        deployment_id = await _resolve.deployment_id(self._transport, deployment)
        response = await self._transport.request("POST", f"{_API_PATH}{deployment_id}/resume")
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
            json={"deployment_ids": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    async def bulk_resume(self, deployments: Sequence[str]) -> BulkActionResult:
        """Resume many paused deployments in one org-scoped server-side call.

        Requires the ``model_deployment`` feature (paid tier). An id that can't
        resume lands in ``not_found`` instead of aborting the batch.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}bulk-resume",
            json={"deployment_ids": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkActionResult, response)

    async def bulk_delete(self, deployments: Sequence[str]) -> BulkDeleteResult:
        """Delete (terminate) many deployments in one org-scoped server-side call."""
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}bulk-delete",
            json={"deployment_ids": await _resolve.deployment_ids(self._transport, deployments)},
        )
        return self._parse(BulkDeleteResult, response)

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["object_detection"],
        timeout: float = 60.0,
    ) -> DeploymentClient[DetectionResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["instance_segmentation"],
        timeout: float = 60.0,
    ) -> DeploymentClient[InstanceSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["semantic_segmentation"],
        timeout: float = 60.0,
    ) -> DeploymentClient[SemanticSegmentationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["keypoint_detection"],
        timeout: float = 60.0,
    ) -> DeploymentClient[KeypointResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: Literal["classification"],
        timeout: float = 60.0,
    ) -> DeploymentClient[ClassificationResult]: ...

    @overload
    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: None = None,
        timeout: float = 60.0,
    ) -> DeploymentClient[AnyResult]: ...

    @staticmethod
    def connect(
        deployment: Deployment,
        api_key: str,
        *,
        task: TaskName | None = None,
        timeout: float = 60.0,
    ) -> DeploymentClient[Any]:
        """Build a direct (sync) :class:`DeploymentClient` for an active deployment.

        Pass ``task=`` (the deployed model's ``model_type``) to narrow what
        :meth:`DeploymentClient.infer` returns to that task's result class,
        exactly as ``task=`` does on :func:`pictograph.get_model`. Without it the
        client is un-narrowed and ``infer`` returns
        :data:`~pictograph.inference.results.AnyResult`; either way the runtime
        object is the correct result class for whatever the endpoint serves.
        """
        if not deployment.endpoint_url:
            raise ValueError("Deployment has no endpoint_url yet (still provisioning?)")
        return DeploymentClient(
            deployment.endpoint_url,
            api_key,
            task=cast("Any", task),
            timeout=timeout,
        )
