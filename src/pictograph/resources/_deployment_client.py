"""DeploymentClient - call a deployed model's endpoint directly.

Talks straight to the inference gateway (NOT through the Pictograph backend) for
minimum latency, authenticating with the per-deployment bearer token, and returns
**the same typed result classes a local model returns**::

    from pictograph import DeploymentClient, DetectionResult

    client = DeploymentClient(
        endpoint="https://workflows.pictograph.io/<slug>/predict",
        api_key="pk_deploy_...",
        task="object_detection",
    )
    result: DetectionResult = client.infer("photo.jpg")  # path | URL | bytes

    for p in result.predictions:  # list[BBoxAnnotation] - narrowed
        print(p.name, round(p.confidence, 2), p.bounding_box)

── Why Remote and Edge return the same thing ──
They are the same payload. A deployment's server-side container runs the shared
single-image dispatch function; the SDK's local engines call the byte-identical
vendored twin of that same function; and the gateway passes the dict through
verbatim as its JSON body. So both sides go through the ONE shared
converter (:func:`pictograph.inference.results.build_result`) and a difference
between them would be a bug rather than an expected variation.

``task=`` is what narrows the return type, exactly as it does on
:func:`pictograph.get_model`. It is also CHECKED: the endpoint reports its own
``model_type`` on every response, and a mismatch raises rather than parsing a
classifier's output into a detector's result. Omit it and ``infer`` returns
:data:`~pictograph.inference.results.AnyResult`, which you narrow yourself.

── What a remote result cannot tell you ──
:attr:`~pictograph.InferenceResult.backend` is ``"onnxruntime"`` (a deployment
always serves an ONNX graph). :attr:`~pictograph.InferenceResult.device` is the
hardware the gateway reports, or ``"remote"`` when it reports none - never a
fabricated ``"cpu"``. :attr:`~pictograph.InferenceResult.providers` is empty and
:attr:`~pictograph.InferenceResult.inference_ms` is ``None``: the wire carries no
forward-pass timing, and substituting the client's round-trip (which includes
network and any scale-to-zero cold start) would be a different measurement
wearing the same name. Use :meth:`infer_raw` if you want the untouched JSON.

Errors are mapped to the SDK's typed exception hierarchy (``RateLimitError``,
``AuthError``, ``ForbiddenError``, ``ServerError``, …) exactly like the main
:class:`pictograph.Client`, so a caller can ``except pictograph.exceptions.
PictographError`` uniformly. Inference is side-effect-free, so a transient
gateway ``429`` (the per-deployment rate limit), scale-to-zero cold-start ``5xx``,
or network blip is retried with the same exponential-backoff / ``Retry-After``
policy the main transport uses. Set ``max_retries=0`` for a latency-strict caller
that wants a single shot.
"""

from __future__ import annotations

import base64
import json as _json
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast, overload

import httpx

from pictograph._http.retry import RetryPolicy
from pictograph.exceptions import (
    NetworkError,
    RequestTimeoutError,
    ServerError,
    from_response,
)
from pictograph.inference.results import (
    TASK_RESULT_TYPES,
    AnyResult,
    ClassificationResult,
    DetectionResult,
    InferenceResult,
    InstanceSegmentationResult,
    KeypointResult,
    SemanticSegmentationResult,
    TaskName,
    build_result,
)

R_co = TypeVar("R_co", bound=InferenceResult, covariant=True)

#: What ``device`` says when the gateway did not report the deployment's
#: hardware. Deliberately not ``"cpu"``: a GPU deployment reported as CPU is a
#: silent falsehood, and this field is read as provenance.
UNREPORTED_DEVICE = "remote"

#: Response header the inference gateway uses to report the deployment's device.
_DEVICE_HEADER = "X-Pictograph-Device"


class DeploymentClient(Generic[R_co]):
    """Direct client for a single deployed model endpoint.

    Generic over the result type. ``DeploymentClient(..., task="object_detection")``
    is a ``DeploymentClient[DetectionResult]`` whose :meth:`infer` returns a
    :class:`~pictograph.DetectionResult`; without ``task`` it is a
    ``DeploymentClient[AnyResult]`` you narrow at the call site.
    """

    @overload
    def __init__(
        self: DeploymentClient[DetectionResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["object_detection"],
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DeploymentClient[InstanceSegmentationResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["instance_segmentation"],
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DeploymentClient[SemanticSegmentationResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["semantic_segmentation"],
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DeploymentClient[KeypointResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["keypoint_detection"],
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DeploymentClient[ClassificationResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["classification"],
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DeploymentClient[AnyResult],
        endpoint: str,
        api_key: str,
        *,
        task: None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        task: TaskName | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        if not api_key:
            raise ValueError("api_key is required")
        if task is not None and task not in TASK_RESULT_TYPES:
            raise ValueError(
                f"Unknown task {task!r}. Expected one of: {', '.join(sorted(TASK_RESULT_TYPES))}."
            )
        self._endpoint = endpoint
        self._task: TaskName | None = task
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        # Mirror pictograph.Client's transport: an injectable policy (tests pass a
        # no-op sleep), else one seeded from max_retries.
        self._retry = retry_policy or RetryPolicy(max_retries=max_retries)
        # One persistent, connection-pooled client (keep-alive) rather than a
        # fresh client per call - the low-latency serving path this advertises.
        self._http = httpx.Client(timeout=self._timeout)

    def __repr__(self) -> str:
        # Never include the bearer token in repr - it would leak through logs /
        # tracebacks (mirrors pictograph.Client.__repr__).
        task = f", task={self._task!r}" if self._task else ""
        return f"DeploymentClient(endpoint={self._endpoint!r}{task})"

    @property
    def task(self) -> TaskName | None:
        """The declared task, or ``None`` when the caller did not declare one."""
        return self._task

    def close(self) -> None:
        """Close the pooled HTTP connection. Also runs on context-manager exit
        (``with DeploymentClient(...) as dc: ...``)."""
        self._http.close()

    def __enter__(self) -> DeploymentClient[R_co]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def infer(
        self,
        image: str | bytes | Path,
        *,
        confidence: float | None = None,
        class_filter: list[str] | None = None,
        top_k: int | None = None,
    ) -> R_co:
        """Run inference on one image and return the typed result.

        Args:
            image: A local file path, an ``http(s)://`` URL, or raw image bytes.
            confidence: Override the deployment's default confidence threshold.
            class_filter: Restrict returned classes.
            top_k: For classifiers, number of predictions to return.

        Returns:
            The task's result class - :class:`~pictograph.DetectionResult`,
            :class:`~pictograph.InstanceSegmentationResult`,
            :class:`~pictograph.SemanticSegmentationResult`,
            :class:`~pictograph.KeypointResult` or
            :class:`~pictograph.ClassificationResult`. Narrowed statically when
            ``task=`` was declared at construction.

        Raises:
            RateLimitError / AuthError / ForbiddenError / NotFoundError /
            PaymentRequiredError / ValidationError / ServerError: mapped from the
            gateway's HTTP status (see :func:`pictograph.exceptions.from_response`).
            NetworkError / RequestTimeoutError: on a transport failure that
            outlived the retry budget.
            ServerError: the response carried no usable ``model_type``, or it
            contradicts the ``task`` this client was constructed with.
        """
        resp = self._request(image, confidence, class_filter, top_k)
        payload = self._parse(resp)
        device = resp.headers.get(_DEVICE_HEADER) or UNREPORTED_DEVICE
        return cast("R_co", self._to_result(payload, device))

    def infer_raw(
        self,
        image: str | bytes | Path,
        *,
        confidence: float | None = None,
        class_filter: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run inference and return the endpoint's JSON **untouched**.

        The escape hatch for a caller that wants the wire body verbatim -
        ``{"model_type": …, "predictions": [...]}``, plus ``tags`` on a
        classifier. :meth:`infer` is the typed path and what you almost always
        want; this exists so that choosing types never costs you access to a
        field the result classes do not model.
        """
        return self._parse(self._request(image, confidence, class_filter, top_k))

    # ── internals ──

    def _request(
        self,
        image: str | bytes | Path,
        confidence: float | None,
        class_filter: list[str] | None,
        top_k: int | None,
    ) -> httpx.Response:
        body = self._build_body(image, confidence, class_filter, top_k)

        def _send() -> httpx.Response:
            try:
                return self._http.post(self._endpoint, headers=self._headers, json=body)
            except httpx.TimeoutException as e:
                raise RequestTimeoutError(str(e)) from e
            except httpx.RequestError as e:
                raise NetworkError(str(e)) from e

        # A predict call has no server-side side effect, so it is safe to
        # retry on 429 (the gateway rate limit) / 5xx (cold start) / network -
        # has_idempotency_key=True opts the POST into the transient-retry set.
        return self._retry.execute(_send, method="POST", has_idempotency_key=True)

    def _to_result(self, payload: dict[str, Any], device: str) -> AnyResult:
        """Typed result from one endpoint payload, through the shared builder.

        The endpoint states its own ``model_type``; a declared ``task`` is
        verified against it rather than trusted over it, because parsing a
        classifier's ``{"class": …}`` entries as a detector's annotations would
        not fail loudly - it would quietly produce an empty result.
        """
        reported = payload.get("model_type")
        if reported is not None and reported not in TASK_RESULT_TYPES:
            raise ServerError(
                f"Deployment endpoint reported an unknown model_type {reported!r}. "
                f"Expected one of: {', '.join(sorted(TASK_RESULT_TYPES))}.",
                response=payload,
            )
        if self._task is not None and reported is not None and reported != self._task:
            raise ServerError(
                f"This deployment serves a {reported!r} model, but the client was "
                f"constructed with task={self._task!r}. Construct it with "
                f"task={reported!r} (or omit task) to read its predictions.",
                response=payload,
            )
        task = cast("TaskName | None", reported) or self._task
        if task is None:
            raise ServerError(
                "Deployment endpoint returned no model_type, so its task is "
                "unknown. Construct the client with task=… to parse it anyway, "
                "or use infer_raw() for the untouched body.",
                response=payload,
            )
        # backend: a deployment always serves an ONNX graph (its server-side
        # container runs onnxruntime and builds only ONNX wrappers), so this is
        # measured, not assumed.
        return build_result(
            payload,
            task=task,
            backend="onnxruntime",
            device=device,
            providers=(),
            inference_ms=None,
            source="This deployment",
        )

    def _build_body(
        self,
        image: str | bytes | Path,
        confidence: float | None,
        class_filter: list[str] | None,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Build the JSON request body for one image.

        Bytes and local files ride as base64 JSON (not multipart) so the
        confidence / class_filter / top_k options travel in the same request -
        the endpoint's multipart ``file`` path sets ``body = {}`` and silently
        drops every option to the deployment's baked-in defaults.
        """
        if isinstance(image, (bytes, bytearray)):
            value = base64.b64encode(bytes(image)).decode()
        elif isinstance(image, str) and image.startswith(("http://", "https://")):
            body: dict[str, Any] = {"image": {"type": "url", "value": image}}
            self._add_opts(body, confidence, class_filter, top_k)
            return body
        else:
            value = base64.b64encode(Path(image).read_bytes()).decode()
        body = {"image": {"type": "base64", "value": value}}
        self._add_opts(body, confidence, class_filter, top_k)
        return body

    @staticmethod
    def _parse(resp: httpx.Response) -> dict[str, Any]:
        """Map the final response to JSON or an SDK-typed error (mirror Transport)."""
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()  # type: ignore[no-any-return]
            except _json.JSONDecodeError as e:
                raise ServerError(
                    f"Deployment endpoint returned {resp.status_code} with non-JSON body: {e}",
                    status_code=resp.status_code,
                    response=resp.text,
                ) from e
        body: Any
        try:
            body = resp.json()
        except _json.JSONDecodeError:
            body = resp.text or None
        raise from_response(
            resp.status_code,
            body=body,
            request_id=resp.headers.get("X-Request-Id"),
            headers=resp.headers,
        )

    @staticmethod
    def _add_opts(
        body: dict[str, Any],
        confidence: float | None,
        class_filter: list[str] | None,
        top_k: int | None,
    ) -> None:
        if confidence is not None:
            body["confidence"] = confidence
        if class_filter is not None:
            body["class_filter"] = class_filter
        if top_k is not None:
            body["top_k"] = top_k
