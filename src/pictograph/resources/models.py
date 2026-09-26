"""Models resource - read, update, download, fork, delete trained models.

Models are produced by the training pipeline; they are never created directly
via this resource. Use :meth:`Training.create` to train one.

Like datasets, models are org-unique by ``name`` and the SDK prefers name-based
addressing: every single-model method takes the ``name`` positionally OR a
``model_id=`` keyword - exactly one - and both forms hit the same backend
serializer, so the returned shape is identical.

Download streams the weights (ONNX by default, or native PyTorch via
``format="pytorch"``) through a backend-issued signed storage URL (60-min TTL),
with the same atomic ``.part`` file rename as ``client.images.download`` so a
failed transfer leaves no partial file at the destination.
"""

from __future__ import annotations

import contextlib
import uuid as _uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote, unquote

import httpx

from pictograph._http.pagination import OffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
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
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pictograph.inference import AnyModel, Device, TaskName, WeightFormat

_API_PATH = "/api/v1/developer/models/"


def _single_path(name: str | None, model_id: str | None, suffix: str = "") -> str:
    """Resolve the by-name vs by-uuid path form. Exactly one of name/model_id.

    Shared by every single-model method (and the async twin) so the addressing
    contract can't drift per-method - mirrors ``datasets._single_path``.
    """
    if (name is None) == (model_id is None):
        raise ValueError("Pass exactly one of `name` (positional) or `model_id=`.")
    # ONE segment for both forms - `/models/{model}` takes a name or a UUID.
    # The `/by-name/` prefix was removed from the API and 404s.
    base = f"{_API_PATH}{quote(name, safe='')}" if name is not None else f"{_API_PATH}{model_id}"
    return f"{base}{suffix}"


def _resolve_manifest_version(manifest: ModelFileManifest, version: str | int | None) -> str:
    """Map a caller's ``version`` (label, number, id, or None=current) to a version_id.

    Shared with the async twin so the resolution contract can't drift.
    """
    if version is None:
        for v in manifest.versions:
            if v.is_latest:
                return v.version_id
        if manifest.versions:
            return manifest.versions[0].version_id
        raise ValueError("Model has no versions in its file manifest")
    want = str(version)
    for v in manifest.versions:
        if want in (v.version_id, v.version_label or "", str(v.version_number)):
            return v.version_id
    known = ", ".join(v.version_label or str(v.version_number) for v in manifest.versions)
    raise ValueError(f"No version {version!r} in manifest (known: {known})")


def _validate_safetensors_header(path: Path) -> None:
    """Parse the safetensors header (8-byte LE length + JSON) and require at
    least one tensor key - the stdlib half of ``safetensors.safe_open``."""
    import json
    import struct

    with Path(path).open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("file too short for a safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > 100_000_000:
            raise ValueError(f"implausible header length {header_len}")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    keys = [k for k in header if k != "__metadata__"]
    if not keys:
        raise ValueError("no tensor keys in header")


class Models(Resource):
    """Operations on trained CV models in your organization."""

    # ───────────── list / iter ─────────────

    def list(
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
            name: Restrict to models with this exact name (org-unique). Prefer
                :meth:`get` to fetch a single model by name.
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
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Model, response.get("data", []))

    def iter(
        self,
        *,
        dataset_name: str | None = None,
        status: ModelStatus | None = None,
        model_type: ModelType | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> OffsetPager[Model]:
        """Auto-paging iterator across every model in your organization.

        Stops on the server-computed ``pagination.has_more`` flag.
        """
        base: dict[str, Any] = {}
        if dataset_name is not None:
            base["dataset_name"] = dataset_name
        if status is not None:
            base["status"] = status
        if model_type is not None:
            base["model_type"] = model_type

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
            parse_item=lambda raw: self._parse(Model, raw),
        )

    # ───────────── get (by name / by id) ─────────────

    def get(self, name: str | None = None, *, model_id: str | None = None) -> Model:
        """Fetch a single model by name (or ``model_id=`` UUID).

        Args:
            name: Model name. Case-sensitive, unique within the org.
            model_id: Model UUID - the keyword alternative to ``name``.
        """
        response = self._transport.request("GET", _single_path(name, model_id))
        return self._parse(Model, response["data"])

    def get_by_name(self, model: str) -> Model:
        """Fetch a model by its name (org-unique) OR its id - whichever you have.

        Handy when you have a user-supplied string that could be either (e.g. to
        run a model locally with :mod:`pictograph.inference`).

        Which form it is, is decided by SHAPE. This used to try the name first
        and fall back on ``NotFoundError``, which cost a whole extra round-trip
        for every id and logged a spurious 404 server-side on each one.

        Raises:
            NotFoundError: No model with that name or id in your organization.
        """
        if _resolve.looks_like_id(model):
            return self.get(model_id=model)
        return self.get(model)

    # ───────────── update (by name / by id) ─────────────

    def update(
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

        Rename it, edit its description/readme, set its license, or flip its
        visibility. A member+ API key is required; changing ``visibility``
        (publishing to Explore) requires admin+. A ``new_name`` that collides
        with another model in your org is rejected (400).

        Args:
            name: The model to update, addressed by name (positional).
            model_id: The model to update, addressed by UUID (keyword).
            new_name: A new name for the model (the field a rename sets - kept
                distinct from the ``name`` path argument).
            description: New description.
            readme: New markdown model card.
            visibility: ``"private"`` or ``"public"`` (admin+).
            license_id: A ``licenses`` catalog id, or ``"custom"``.
            license_custom_text: License body when ``license_id == "custom"``.
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
        response = self._transport.request("PATCH", _single_path(name, model_id), json=body)
        return self._parse(Model, response["data"])

    def load(
        self,
        name: str,
        *,
        task: TaskName | None = None,
        format: WeightFormat = "onnx",
        precision: Literal["fp32", "fp16"] | None = None,
        target: str | None = None,
        confidence: float = 0.5,
        device: Device = "auto",
        cache_dir: str | Path | None = None,
    ) -> AnyModel:
        """Load a name for LOCAL inference, using this client's auth.

        Synchronous-only: there is no ``AsyncClient`` twin. It builds a local
        inference engine from downloaded weights, and the resulting
        ``model.predict()`` is itself synchronous in-process torch/ORT.

        The client-bound twin of the top-level :func:`pictograph.get_model` - same
        arguments, same behaviour, same task classes; see it for the full reference,
        for the ``format=`` table and for how ``task=`` narrows the return type.

        Requires the inference extra::

            pip install "pictograph[inference]"                 # onnx / pytorch / safetensors
            pip install "pictograph[inference,executorch]"      # + pytorch_engine
            pip install "pictograph[inference,tensorrt]"        # + tensorrt_engine

        Raises:
            ConflictError: The name does not publish ``format``. The message names
                the formats it does publish - a format is never substituted.
            ImportError: The runtime's package (or a name family's framework) is
                not installed. The message carries the exact ``pip install``.
        """
        from pictograph.inference import _load_by_name

        return _load_by_name(  # type: ignore[no-any-return]
            name,
            models=self,
            task=task,
            format=format,
            precision=precision,
            target=target,
            confidence=confidence,
            device=device,
            cache_dir=cache_dir,
        )

    def predict(
        self,
        name: str,
        *,
        image: str | Path | bytes,
        confidence: float = 0.5,
        top_k: int = 3,
    ) -> ModelPredictResult:
        """Run ONE image through the model on Pictograph's GPU service.

        Remote test inference - no local ONNX runtime needed (for local
        inference use :meth:`load`). Spends your organization's compute
        credits (priced as a single trained-model inference pass; charged on
        success only). member+ role required.

        Args:
            name: The model's name, unique within your organization.
            image: Path to an image file, or raw image bytes.
            confidence: Minimum score for returned predictions (0.05-0.95).
            top_k: For classification models, how many predictions to return.

        Returns:
            :class:`~pictograph.models.model.ModelPredictResult` -
            ``annotations`` (detection/segmentation) or ``tags``
            (classification), plus ``inference_seconds``.
        """
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser()
            data = path.read_bytes()
            filename = path.name
        else:
            data = image
            filename = "upload.jpg"
        found = self.get_by_name(name)
        response = self._transport.request(
            "POST",
            f"{_API_PATH}{found.id}/predict",
            params={"confidence_threshold": confidence, "top_k": top_k},
            files={"file": (filename, data, "application/octet-stream")},
        )
        return self._parse(ModelPredictResult, response["data"])

    def fork(self, organization: str, model: str) -> Model:
        """Import (fork) a public model into your organization.

        The model analog of forking a public dataset: a private copy of a
        public model in ANY organization is created in yours. The fork
        references the source model's weights (no byte copy), so it is
        downloadable immediately and fast even for large models.

        Addressed by the QUALIFIED PAIR ``organization/model`` - the same
        owner-plus-name slug the model's public page uses. A bare name will not
        do, because names are unique only within an organization, and this is
        the one model call that deliberately reaches across them.

        The copy's name is suffixed (``"Name (2)"``) if a model of that name
        already exists in your organization.

        Args:
            organization: Slug of the organization that owns the source model.
            model: Slug or name of the source **public** model.

        Returns:
            The newly created :class:`~pictograph.models.model.Model` in your
            organization (``visibility="private"``, ``status="ready"``,
            ``forked_from_model_id`` set to the source).

        Raises:
            NotFoundError: The source model does not exist or is not public.
            ValidationError: The source model is not ``ready`` or has no
                weights to import.
            ForbiddenError: Your API key role cannot create models
                (requires member, admin, or owner).
        """
        path = f"{_API_PATH}{quote(organization, safe='')}/{quote(model, safe='')}/fork"
        response = self._transport.request("POST", path)
        return self._parse(Model, response["data"])

    # ───────────── download (by name / by id) ─────────────

    def download(
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

         ``format`` here is the DOWNLOAD ROUTE's vocabulary, which names the FILE,
        and is deliberately not the loader's
        :data:`~pictograph.inference.runtime.WeightFormat`. This method is a raw byte
        fetch that mirrors ``GET /models/{id}/download?format=`` one-for-one; to RUN
        a model, use :meth:`load` / :func:`pictograph.get_model` and their
        ``format=``. Three of the five spell the same; two differ:

        ==================  ===================================
        ``download``        ``load`` / ``get_model``
        ==================  ===================================
        ``onnx``            ``onnx``
        ``pytorch``         ``pytorch``
        ``safetensors``     ``safetensors``
        ``pte``             ``pytorch_engine``
        ``engine``          ``tensorrt_engine``
        ==================  ===================================

        Fetches a signed URL from the backend, then streams the bytes over a
        SECOND request to a different host - the storage service - through a
        separate, scoped ``httpx.Client`` that sends NO SDK credentials (the
        authorization is the signature inside the URL). Bytes land in a
        sibling ``.part`` file and are renamed atomically on success -
        failures leave nothing at the destination.

        Args:
            name: The model to download, addressed by name (positional).
            output_path: Local path for the weights file. Parent dirs created.
            model_id: The model to download, addressed by UUID (keyword).
            format: Weights format to download. ``"onnx"`` (default) returns
                the exported ONNX graph; ``"pytorch"`` returns the native
                PyTorch ``.pth`` checkpoint; ``"safetensors"`` returns the
                raw trained tensors (published only for runs whose
                inference-parity gate passed); ``"pte"`` returns an ExecuTorch
                program and ``"engine"`` a TensorRT plan. ``"pytorch"`` and
                ``"safetensors"`` raise
                :class:`~pictograph.exceptions.ConflictError` (HTTP 409) for
                models trained before the respective export shipped.
            precision: ``"fp32"`` (the server's default) or ``"fp16"``. Only
                meaningful for ``"pte"`` and ``"engine"``, which exist per
                precision - the other formats have exactly one file each.
            target: Which binding to fetch, for the formats that have more than
                one. ``"engine"`` is 1:N - one plan per GPU architecture - so
                ``target="sm75"`` picks the T4 plan; omitting it on a model with
                several returns a 409 that LISTS the targets that exist. For
                ``"pte"`` it is the lowering backend (``"xnnpack"``, the default).
            chunk_size: Streaming chunk size (default 8 MB).
            progress: Optional ``(bytes_so_far, total_bytes)`` callback.
                ``total_bytes`` is ``0`` if the storage response has no
                ``Content-Length``.

        Returns:
            The output path.

        Raises:
            NotFoundError: No such model, or its weights file is missing.
            ValidationError: Model status is not ``ready``.
            ConflictError: ``format="pytorch"`` requested for a model trained
                before dual-format export, or an ambiguous / not-yet-built
                ``"engine"`` / ``"pte"`` request. The message states the actual
                reason and, when ambiguous, the targets available.
            ApiError: The transfer from storage failed.
        """
        params: dict[str, str] = {"format": format}
        if precision is not None:
            params["precision"] = precision
        if target is not None:
            params["target"] = target
        url_response = self._transport.request(
            "GET",
            _single_path(name, model_id, "/download"),
            params=params,
        )
        download_url: str = url_response["data"]["download_url"]

        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")

        try:
            with (
                httpx.Client(
                    http2=True,
                    timeout=httpx.Timeout(self._transport._config.timeout, read=600.0),
                ) as gcs,
                gcs.stream("GET", download_url) as response,
            ):
                if response.status_code >= 300:
                    response.read()
                    raise ApiError(
                        f"Model download failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                        response=response.text,
                    )
                total = int(response.headers.get("Content-Length", 0))
                sent = 0
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        fh.write(chunk)
                        sent += len(chunk)
                        if progress is not None:
                            progress(sent, total)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

        # A safetensors fetch is VALIDATED before it lands: the format's
        # own header (8-byte little-endian length + JSON) must parse and carry
        # at least one tensor key. Dependency-free on purpose - the SDK stays
        # torch-less; `safetensors.safe_open` reads the same header.
        if format == "safetensors":
            try:
                _validate_safetensors_header(tmp)
            except Exception as exc:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise ApiError(f"Downloaded safetensors file failed validation: {exc}") from exc

        tmp.replace(out)
        return out

    # ──────────────── files manifest + per-file download ────────────────

    def versions(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
    ) -> ModelVersionsPayload:
        """The model's version list + promote state.

        Every entry carries ``version_number`` / ``version_label`` / ``status``
        / ``architecture`` / ``metrics`` / ``precision`` plus ``is_current`` -
        the RESOLVED effective version (the owner-promoted pin first, else the
        newest ready). ``pinned_version_id`` is non-null iff a version was
        explicitly promoted via :meth:`set_current_version`.
        """
        response = self._transport.request("GET", _single_path(name, model_id, "/versions"))
        return self._parse(ModelVersionsPayload, response["data"])

    def set_current_version(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
        version_id: str | None,
    ) -> ModelVersionsPayload:
        """Promote / roll back: pin the model to one of its READY versions.

        Admin+ API key. The pin decides what the model serves EVERYWHERE -
        downloads, deployments provisioning, auto-annotate selection - and it
        SURVIVES later retrains (that is what makes rollback real). Pass
        ``version_id=None`` to clear the pin so the model follows its newest
        ready version again.

        Raises:
            NotFoundError: The version does not belong to this model (or the
                model does not exist in your org).
            ValidationError: The version is not ``ready`` - promoting a failed
                version would break every downstream resolver at once.
        """
        response = self._transport.request(
            "PATCH",
            _single_path(name, model_id, "/current-version"),
            json={"version_id": version_id},
        )
        return self._parse(ModelVersionsPayload, response["data"])

    def files(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
    ) -> ModelFileManifest:
        """The model's version + file manifest, addressed by name or ``model_id=``.

        Every version's downloadable artifacts in one list: weights
        (``onnx`` / ``pytorch``), the immutable ``config.json``
        reproducibility artifact (``kind="config"``), and the request-time
        generated ``LICENSE.md`` / ``README.md``. Each file row carries the
        ``version_id`` it belongs to; feed a row's ``name`` (plus an optional
        version) to :meth:`download_file`.

        Returns:
            :class:`~pictograph.models.model.ModelFileManifest` -
            ``versions`` is never empty; a pre-versioning model reports one
            synthetic version whose id equals the model id.
        """
        response = self._transport.request("GET", _single_path(name, model_id, "/files"))
        return self._parse(ModelFileManifest, response["data"])

    def download_file(
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
        """Download ONE manifest artifact by its ``name`` (see :meth:`files`).

        Works for every manifest kind: stored artifacts (weights,
        ``config.json``) stream from a backend-signed storage URL with the
        same atomic ``.part`` rename as :meth:`download`; the request-time
        generated ``LICENSE.md`` / ``README.md`` arrive as a ``data:`` URL
        whose payload is written directly (nothing to stream).

        Args:
            name: The model, addressed by name (positional).
            model_id: The model, addressed by UUID (keyword).
            file_name: The manifest row's ``name`` (e.g.
                ``"config.json"``).
            version: Which version to take the file from - a version label
                (``"2.0.0"``), a version number (``2``), or a ``version_id``
                from a prior :meth:`files` call (used as-is, no extra
                request). ``None`` (default) resolves to the model's current
                version.
            output_path: Local destination. Parent dirs created.
            chunk_size: Streaming chunk size (default 8 MB).
            progress: Optional ``(bytes_so_far, total_bytes)`` callback for
                streamed artifacts.

        Returns:
            The output path.

        Raises:
            NotFoundError: No such model, or ``file_name`` is not an artifact
                of the resolved version.
            ValueError: ``version`` does not match any manifest version.
            ApiError: The transfer from storage failed.
        """
        version_id: str
        if isinstance(version, str):
            try:
                version_id = str(_uuid.UUID(version))
            except ValueError:
                version_id = _resolve_manifest_version(self.files(name, model_id=model_id), version)
        else:
            version_id = _resolve_manifest_version(self.files(name, model_id=model_id), version)

        url_response = self._transport.request(
            "GET",
            _single_path(name, model_id, "/files/download"),
            params={"version_id": version_id, "name": file_name},
        )
        download_url: str = url_response["data"]["download_url"]

        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")

        if download_url.startswith("data:"):
            # Request-time generated content - the URL payload IS the file.
            _header, _, payload = download_url.partition(",")
            try:
                tmp.write_bytes(unquote(payload).encode("utf-8"))
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            tmp.replace(out)
            return out

        try:
            with (
                httpx.Client(
                    http2=True,
                    timeout=httpx.Timeout(self._transport._config.timeout, read=600.0),
                ) as gcs,
                gcs.stream("GET", download_url) as response,
            ):
                if response.status_code >= 300:
                    response.read()
                    raise ApiError(
                        f"Model file download failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                        response=response.text,
                    )
                total = int(response.headers.get("Content-Length", 0))
                sent = 0
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        fh.write(chunk)
                        sent += len(chunk)
                        if progress is not None:
                            progress(sent, total)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

        tmp.replace(out)
        return out

    # ───────────── delete (by name / by id) ─────────────

    def delete(self, name: str | None = None, *, model_id: str | None = None) -> None:
        """Delete a model by name (or ``model_id=`` UUID). Requires admin+.

        Removes the database row immediately. Object-storage cleanup runs as a
        background task - the file may linger briefly after this call returns.
        """
        self._transport.request("DELETE", _single_path(name, model_id))

    def bulk_delete(self, model_ids: Sequence[str]) -> BulkDeleteResult:
        """Delete many models in one atomic, org-scoped, server-side call.

        Unlike calling :meth:`delete` in a loop, this issues a single request
        the backend resolves with chunked, organization-scoped deletes, so it
        never fans out N calls or materializes a giant id list on the wire
        (one moving piece, idempotent). Requires the ``admin`` or ``owner``
        role, same as :meth:`delete`.

        Args:
            model_ids: UUIDs of the models to delete. Duplicates are ignored.
                Ids that do not resolve in your organization are reported in
                :attr:`~pictograph.models.common.BulkDeleteResult.not_found`
                rather than raising, so a re-run of a completed delete still
                succeeds.

        Returns:
            A :class:`~pictograph.models.common.BulkDeleteResult` with the ids
            actually deleted (``succeeded``), the ids not found, and the
            deleted ``count``.

        Raises:
            ForbiddenError: Your API key role cannot delete models.
            ValidationError: ``model_ids`` is empty.
        """
        response = self._transport.request(
            "POST", f"{_API_PATH}bulk-delete", json={"model_ids": list(model_ids)}
        )
        return self._parse(BulkDeleteResult, response.get("data", response))
