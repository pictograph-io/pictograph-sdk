"""Async Models resource - read, download, delete trained models.

Async twin of :class:`pictograph.resources.models.Models`. Local inference
(``Models.load`` / :func:`pictograph.get_model`) stays synchronous - ONNX
inference is CPU-bound, so there is no async ``load`` here; use the sync client
for local prediction.
"""

from __future__ import annotations

import contextlib
import uuid as _uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote, unquote

from pictograph._http.pagination import AsyncOffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.aio._download import stream_url_to_file
from pictograph.aio.resources import _resolve
from pictograph.exceptions import ApiError
from pictograph.models.common import BulkDeleteResult
from pictograph.models.model import (
    Model,
    ModelFileManifest,
    ModelPredictResult,
    ModelStatus,
    ModelType,
    ModelVersionsPayload,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources.models import _resolve_manifest_version, _single_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_API_PATH = "/api/v1/developer/models/"


class AsyncModels(AsyncResource):
    """Operations on trained CV models in your organization (async)."""

    # ───────────── list / iter ─────────────

    async def list(
        self,
        *,
        name: str | None = None,
        dataset_name: str | None = None,
        status: ModelStatus | None = None,
        model_type: ModelType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Model]:
        """Single-page list of models in your organization.

        Args:
            name: Restrict to models with this exact name (org-unique).
            dataset_name: Restrict to models trained on this dataset.
            status: Restrict to a lifecycle status.
            model_type: Restrict to one of the four model categories.
            limit: Page size (backend cap: 100).
            offset: Page offset for paginating manually.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if name is not None:
            params["name"] = name
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        if status is not None:
            params["status"] = status
        if model_type is not None:
            params["model_type"] = model_type
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Model, response.get("data", []))

    def iter(
        self,
        *,
        dataset_name: str | None = None,
        status: ModelStatus | None = None,
        model_type: ModelType | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[Model]:
        """Auto-paging async iterator across every model in your organization."""
        base: dict[str, Any] = {}
        if dataset_name is not None:
            base["dataset_name"] = dataset_name
        if status is not None:
            base["status"] = status
        if model_type is not None:
            base["model_type"] = model_type

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Model, raw),
        )

    # ───────────── get / update (by name / by id) ─────────────

    async def get(self, name: str | None = None, *, model_id: str | None = None) -> Model:
        """Fetch a single model by name (or ``model_id=`` UUID)."""
        response = await self._transport.request("GET", _single_path(name, model_id))
        return self._parse(Model, response["data"])

    async def update(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
        new_name: str | None = None,
        description: str | None = None,
        readme: str | None = None,
        visibility: Literal["private", "public"] | None = None,
        license_id: str | None = None,
        license_custom_text: str | None = None,
    ) -> Model:
        """Update a model's editable fields (only the ones you pass change).

        Member+ API key; ``visibility`` changes require admin+. Pass ``new_name``
        to rename (kept distinct from the ``name`` path argument).
        """
        body: dict[str, Any] = {}
        for key, value in (
            ("name", new_name),
            ("description", description),
            ("readme", readme),
            ("visibility", visibility),
            ("license_id", license_id),
            ("license_custom_text", license_custom_text),
        ):
            if value is not None:
                body[key] = value
        if not body:
            raise ValueError("update() requires at least one field to change")
        response = await self._transport.request("PATCH", _single_path(name, model_id), json=body)
        return self._parse(Model, response["data"])

    async def get_by_name(self, model: str) -> Model:
        """Fetch a model by its name (org-unique) OR its id - whichever you have.

        Which form it is, is decided by SHAPE. This used to try the name first
        and fall back on ``NotFoundError``, which cost a whole extra round-trip
        for every id and logged a spurious 404 server-side on each one.

        Raises:
            NotFoundError: No model with that name or id in your organization.
        """
        if _resolve.looks_like_id(model):
            return await self.get(model_id=model)
        return await self.get(model)

    async def predict(
        self,
        name: str,
        *,
        image: str | Path | bytes,
        confidence: float = 0.5,
        top_k: int = 3,
    ) -> ModelPredictResult:
        """Run ONE image through the model on Pictograph's GPU service.

        Async twin of :meth:`pictograph.resources.models.Models.predict` -
        remote test inference (no local ONNX runtime; spends org compute
        credits, charged on success only; member+).
        """
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser()
            data = path.read_bytes()
            filename = path.name
        else:
            data = image
            filename = "upload.jpg"
        found = await self.get_by_name(name)
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}{found.id}/predict",
            params={"confidence_threshold": confidence, "top_k": top_k},
            files={"file": (filename, data, "application/octet-stream")},
        )
        return self._parse(ModelPredictResult, response["data"])

    async def fork(self, organization: str, model: str) -> Model:
        """Import (fork) a public model into your organization.

        Creates a private copy referencing the source model's weights (no byte
        copy). Addressed by the QUALIFIED PAIR ``organization/model`` - the slug
        the model's public page uses - because a bare name is unique only within
        an organization and this call deliberately reaches across them.
        The copy's name is suffixed (``"Name (2)"``) on collision.

        Raises:
            NotFoundError: The source model does not exist or is not public.
            ValidationError: The source model is not ``ready`` or has no weights.
            ForbiddenError: Your API key role cannot create models (member+).
        """
        path = f"{_API_PATH}{quote(organization, safe='')}/{quote(model, safe='')}/fork"
        response = await self._transport.request("POST", path)
        return self._parse(Model, response["data"])

    async def download(
        self,
        name: str | None = None,
        *,
        output_path: str | Path,
        model_id: str | None = None,
        format: Literal["onnx", "pytorch", "safetensors", "pte", "engine"] = "onnx",
        precision: Literal["fp32", "fp16"] | None = None,
        target: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download the model's weights file, addressed by name or ``model_id=``.

        Streamed with an atomic ``.part`` rename.

        Args:
            name: The model to download, addressed by name (positional).
            output_path: Local path for the weights file. Parent dirs created.
            model_id: The model to download, addressed by UUID (keyword).
            format: ``"onnx"`` (default), ``"pytorch"``, ``"safetensors"``,
                ``"pte"`` or ``"engine"``. ``"pytorch"`` raises
                :class:`~pictograph.exceptions.ConflictError` for models trained
                before dual-format export.
            precision: ``"fp32"`` or ``"fp16"``. ``None`` takes the server default.
            target: For ``"engine"``, which built binding to fetch. Omitting it on
                a model with several is a ``ConflictError`` that LISTS the
                available targets - so this argument is what makes that error
                recoverable.
            chunk_size: Streaming chunk size (default 8 MB).
            progress: Optional ``(bytes_so_far, total_bytes)`` callback.

        Returns:
            The output path.

        Raises:
            NotFoundError: No such model, or its weights file is missing.
            ValidationError: Model status is not ``ready``.
            ConflictError: ``format="pytorch"`` for a pre-dual-format model, or an
                ambiguous / not-yet-built ``"engine"`` / ``"pte"`` request.
            ApiError: The transfer from storage failed.
        """
        params: dict[str, str] = {"format": format}
        if precision is not None:
            params["precision"] = precision
        if target is not None:
            params["target"] = target
        url_response = await self._transport.request(
            "GET",
            _single_path(name, model_id, "/download"),
            params=params,
        )
        out = await stream_url_to_file(
            url_response["data"]["download_url"],
            output_path,
            timeout=self._transport._config.timeout,
            chunk_size=chunk_size,
            progress=progress,
            error_prefix="Model download",
        )
        # Mirror the sync client's safetensors validation (the async
        # streamer has already renamed, so a failed check deletes the file).
        if format == "safetensors":
            from pictograph.resources.models import _validate_safetensors_header

            try:
                _validate_safetensors_header(out)
            except Exception as exc:
                with contextlib.suppress(OSError):
                    Path(out).unlink()
                raise ApiError(f"Downloaded safetensors file failed validation: {exc}") from exc
        return out

    async def files(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
    ) -> ModelFileManifest:
        """The model's version + file manifest - see the sync twin."""
        response = await self._transport.request("GET", _single_path(name, model_id, "/files"))
        return self._parse(ModelFileManifest, response["data"])

    async def versions(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
    ) -> ModelVersionsPayload:
        """The model's version list + promote state - see the sync twin."""
        response = await self._transport.request("GET", _single_path(name, model_id, "/versions"))
        return self._parse(ModelVersionsPayload, response["data"])

    async def set_current_version(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
        version_id: str | None,
    ) -> ModelVersionsPayload:
        """Promote / roll back the model's current version - see the
        sync twin. ``version_id=None`` clears the pin (follow latest)."""
        response = await self._transport.request(
            "PATCH",
            _single_path(name, model_id, "/current-version"),
            json={"version_id": version_id},
        )
        return self._parse(ModelVersionsPayload, response["data"])

    async def download_file(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
        file_name: str,
        version: str | int | None = None,
        output_path: str | Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download one manifest artifact by ``name`` - see the sync twin.

        ``version`` accepts a label, number, or a ``version_id`` from a prior
        :meth:`files` call (used as-is); ``None`` resolves to the model's
        current version. Generated LICENSE.md / README.md arrive as a
        ``data:`` URL and are written directly; stored artifacts stream with
        the atomic ``.part`` rename.
        """
        version_id: str
        if isinstance(version, str):
            try:
                version_id = str(_uuid.UUID(version))
            except ValueError:
                version_id = _resolve_manifest_version(
                    await self.files(name, model_id=model_id), version
                )
        else:
            version_id = _resolve_manifest_version(
                await self.files(name, model_id=model_id), version
            )

        url_response = await self._transport.request(
            "GET",
            _single_path(name, model_id, "/files/download"),
            params={"version_id": version_id, "name": file_name},
        )
        download_url: str = url_response["data"]["download_url"]

        if download_url.startswith("data:"):
            out = Path(output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            _header, _, payload = download_url.partition(",")
            try:
                tmp.write_bytes(unquote(payload).encode("utf-8"))
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            tmp.replace(out)
            return out

        return await stream_url_to_file(
            download_url,
            output_path,
            timeout=self._transport._config.timeout,
            chunk_size=chunk_size,
            progress=progress,
            error_prefix="Model file download",
        )

    async def delete(self, name: str | None = None, *, model_id: str | None = None) -> None:
        """Delete a model by name (or ``model_id=`` UUID). Requires admin+."""
        await self._transport.request("DELETE", _single_path(name, model_id))

    async def bulk_delete(self, model_ids: Sequence[str]) -> BulkDeleteResult:
        """Delete many models in one atomic, org-scoped, server-side call.

        A single chunked, org-scoped delete (no N-call fan-out, idempotent).
        Requires ``admin``/``owner``. Ids that don't resolve in your org land in
        :attr:`~pictograph.models.common.BulkDeleteResult.not_found`.

        Raises:
            ForbiddenError: Your API key role cannot delete models.
            ValidationError: ``model_ids`` is empty.
        """
        response = await self._transport.request(
            "POST", f"{_API_PATH}bulk-delete", json={"model_ids": list(model_ids)}
        )
        return self._parse(BulkDeleteResult, response.get("data", response))
