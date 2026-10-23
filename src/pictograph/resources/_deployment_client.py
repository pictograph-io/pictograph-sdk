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
They are the same payload. A deployment runs on PRT, Pictograph's realtime engine,
which answers in exactly the shape of the shared single-image dispatch function (its
values checked against ONNX Runtime's when the deployment's engine is built); the
SDK's local engines call the vendored twin of that same function; and the gateway
passes the engine's answer through verbatim as its JSON body. So both sides go
through the ONE shared converter (:func:`pictograph.inference.results.build_result`)
and a difference in shape between them would be a bug rather than an expected
variation.

``task=`` is what narrows the return type, exactly as it does on
:func:`pictograph.get_model`. It is also CHECKED: the endpoint reports its own
``model_type`` on every response, and a mismatch raises rather than parsing a
classifier's output into a detector's result. Omit it and ``infer`` returns
:data:`~pictograph.inference.results.AnyResult`, which you narrow yourself.

── What a remote result reports ──
Every field is what the gateway reports, never a guess.
:attr:`~pictograph.InferenceResult.backend` is ``"tensorrt"`` when the gateway names
PRT as the engine (``X-Pictograph-Engine: prt``: a TensorRT engine of the model's
graph); a gateway that names no engine predates PRT and served the ONNX graph, so that
reads ``"onnxruntime"``. :attr:`~pictograph.InferenceResult.device` is the hardware the
gateway reports (``X-Pictograph-Device``, ``"cuda"``), or ``"remote"`` when it reports
none - never a fabricated ``"cpu"``. :attr:`~pictograph.InferenceResult.inference_ms`
is the engine's own forward-pass time for the request (``infer`` in its
``Server-Timing``), or ``None`` when the response carries none: substituting the
client's round-trip (which includes network and any scale-to-zero cold start) would be
a different measurement wearing the same name.
:attr:`~pictograph.InferenceResult.providers` is empty: the wire does not name the
plan's TensorRT version or GPU target. Use :meth:`infer_raw` if you want the
untouched JSON.

── Errors, retries and cold starts ──
Errors are mapped to the SDK's typed exception hierarchy (``RateLimitError``,
``AuthError``, ``ForbiddenError``, ``ConflictError``, ``ServerError``, ...) exactly
like the main :class:`pictograph.Client`, so a caller can ``except
pictograph.exceptions.PictographError`` uniformly. The gateway's own error token
(``deployment_not_active``, ``deployment_starting``, ``overloaded``,
``inference_timeout``, ...) is the error's ``code``, with a ``fix`` saying what to do.

Inference is side-effect-free, so a transient gateway ``429`` (the per-deployment
rate limit), ``5xx`` or network blip is retried with the same exponential-backoff
policy the main transport uses, and a ``Retry-After`` on a ``429`` or a ``503`` sets
the wait. A scaled-to-zero deployment's first request after idle waits while its
container starts: the gateway holds the request for up to two minutes, so the default
``timeout`` (150 s) outlasts that hold instead of abandoning a request the gateway
would have answered. A deployment that still has not started answers ``503``
``deployment_starting``, which is retried after its ``Retry-After``. Set
``max_retries=0`` for a latency-strict caller that wants a single shot.
"""

from __future__ import annotations

import base64
import json as _json
import math
from pathlib import Path
from typing import Any, Final, Generic, Literal, TypeVar, cast, get_args, overload

import httpx

from pictograph._http.retry import RetryPolicy
from pictograph.exceptions import (
    ApiError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    from_response,
)
from pictograph.inference.results import (
    TASK_RESULT_TYPES,
    AnyResult,
    BackendName,
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

#: The default per-request timeout, in seconds. The gateway holds the first request
#: to a scaled-to-zero deployment for up to 120 s while its container starts (and
#: answers ``503 deployment_starting`` if it never does); a shorter client timeout
#: would abandon a request the gateway was about to answer.
DEFAULT_TIMEOUT_S: Final = 150.0

#: Response header the inference gateway uses to report the deployment's device.
_DEVICE_HEADER = "X-Pictograph-Device"
#: Response header naming the engine that served the request (``prt``).
_ENGINE_HEADER = "X-Pictograph-Engine"
#: The engine's own timings for the request (``decode``, ``infer``, ``total``).
_TIMING_HEADER = "Server-Timing"

#: What ``backend`` reads when the gateway names no engine: the gateway learned to
#: name it with PRT, and every gateway before that served the ONNX graph on ONNX Runtime.
_UNNAMED_ENGINE_BACKEND: Final[BackendName] = "onnxruntime"
_RUNTIME_NAMES: Final[frozenset[str]] = frozenset(get_args(BackendName))

#: The retryable statuses whose ``Retry-After`` sets the wait: the per-deployment rate
#: limit (429), and a deployment that is starting or at capacity (503).
_RETRY_AFTER_STATUSES: Final = frozenset({429, 503})

#: A deployment's request ceiling when its ``inference_config`` sets no ``rate_limit_rpm``.
_DEFAULT_RATE_LIMIT_RPM: Final = 300

#: What to do about each error slug the gateway (or its engine) answers with. The slug
#: becomes the error's ``code``; the message is used only when the body carries none.
_GATEWAY_ERRORS: Final[dict[str, tuple[str, str]]] = {
    "unauthorized": (
        "The deployment refused the bearer token.",
        "Pass the deployment's own pk_deploy_ token (from create, or revealed in the "
        "app), not a pk_live_ API key.",
    ),
    "deployment_not_found": (
        "No deployment answers at this endpoint.",
        "Check the endpoint URL (client.deployments.get(name).endpoint_url); a deleted "
        "deployment's URL stops answering.",
    ),
    "not_found": (
        "The gateway has no such route.",
        "POST to the deployment's endpoint_url exactly as it is: it ends in /predict.",
    ),
    "deployment_not_active": (
        "The deployment is not active.",
        "Resume a paused deployment (client.deployments.resume(...)), or wait for a "
        "provisioning one (client.deployments.wait_until_active(...)).",
    ),
    "rate_limited": (
        "The deployment's request ceiling was reached.",
        # with_gateway_hint replaces it with one that names the Retry-After.
        f"Retry shortly. A deployment takes {_DEFAULT_RATE_LIMIT_RPM} requests a minute "
        "unless its inference_config sets rate_limit_rpm.",
    ),
    "deployment_starting": (
        "The deployment is starting: its container did not start within the gateway's two minutes.",
        "Retry shortly. A deployment that scales to zero (min_containers=0) starts a "
        "container on its first request after idle; keep one warm (min_containers=1) "
        "to avoid the wait.",
    ),
    "warming_up": (
        "The deployment's engine is still loading.",
        "Retry shortly.",
    ),
    "overloaded": (
        "The deployment is at capacity.",
        "Retry after the Retry-After seconds, or allow it more containers (max_containers).",
    ),
    "stopping": (
        "A container of the deployment is stopping.",
        "Retry: another container takes the request.",
    ),
    "inference_timeout": (
        "The model did not answer in time.",
        "Retry; if it keeps timing out, send a smaller image or deploy on a larger GPU (gpu_type).",
    ),
    "inference_failed": (
        "The model failed on this request.",
        "Retry; if it fails again on this image, contact support@pictograph.io with the "
        "request_id.",
    ),
    "deployment_unreachable": (
        "The gateway could not reach the deployment's server.",
        "Retry shortly; if it persists, contact support@pictograph.io with the request_id.",
    ),
    "gateway_unavailable": (
        "The inference gateway could not look the deployment up.",
        "Retry shortly (the SDK retries a short Retry-After itself); if it persists, contact "
        "support@pictograph.io with the request_id.",
    ),
    "deployment_unavailable": (
        "The deployment's server could not take the request.",
        "Retry shortly; if it persists, contact support@pictograph.io with the request_id.",
    ),
    "could_not_decode_image": (
        "The image could not be decoded.",
        "Send a valid image file (a JPEG or a PNG, for example).",
    ),
    "no_image_provided": (
        "No image reached the deployment.",
        "Send a file, bytes, or an http(s) URL that is publicly reachable and answers "
        "200 with at most 64 MiB.",
    ),
    "url_images_not_supported": (
        "The deployment's engine does not fetch image URLs.",
        "Send the image as a file or bytes, or its URL in a small JSON body (the gateway "
        "fetches URLs; the engine never does).",
    ),
    "invalid_request_body": (
        "The request body could not be read.",
        "Send multipart with a file part, or JSON with an image of type base64 or url.",
    ),
    "invalid_option": (
        "An option was refused.",
        "Fix the option the message names (confidence, class_filter, top_k or "
        "instance_output_map).",
    ),
    "request_too_large": (
        "The request is larger than the 64 MiB a deployment takes.",
        "Send a smaller image.",
    ),
}


def backend_of(engine: str | None) -> BackendName:
    """The runtime the gateway's engine header names.

    No header is a gateway from before PRT, which served the ONNX graph on ONNX Runtime.
    An engine named by its runtime reads as that runtime. PRT - and any engine name this
    SDK does not know yet, which only the PRT gateway sends - runs a TensorRT engine of
    the model.
    """
    if not engine or not engine.strip():
        return _UNNAMED_ENGINE_BACKEND
    name = engine.strip().lower()
    if name in _RUNTIME_NAMES:
        return cast("BackendName", name)
    return "tensorrt"


def server_timing_ms(header: str | None, metric: str) -> float | None:
    """``metric``'s ``dur`` (milliseconds) in a ``Server-Timing`` header such as
    ``decode;dur=1.204, infer;dur=13.380, total;dur=15.020``, or None when the header
    does not carry it (or carries something that is not a duration)."""
    if not header:
        return None
    for entry in header.split(","):
        name, _, params = entry.strip().partition(";")
        if name.strip().lower() != metric:
            continue
        for param in params.split(";"):
            key, _, value = param.strip().partition("=")
            if key.strip().lower() != "dur":
                continue
            try:
                ms = float(value.strip().strip('"'))
            except ValueError:
                return None
            return ms if math.isfinite(ms) and ms >= 0 else None
        return None
    return None


def with_gateway_hint(err: ApiError, body: Any) -> ApiError:
    """``err`` with the gateway's error slug as its ``code`` and a ``fix`` for it.

    The gateway answers a flat ``{"error": "<slug>", "detail"?: "..."}`` rather than the
    API's ``{"error": {"code", ...}}`` envelope, so without this the slug would reach the
    caller only as the message.
    """
    slug = body.get("error") if isinstance(body, dict) else None
    if not isinstance(slug, str) or not slug:
        return err
    if err.code is None:
        err.code = slug
    hint = _GATEWAY_ERRORS.get(slug)
    if hint is None:
        return err
    message, fix = hint
    if err.message == slug:  # the body carried no words of its own
        status = body.get("status") if isinstance(body, dict) else None
        if slug == "deployment_not_active" and isinstance(status, str) and status:
            message = f"The deployment is not active (status: {status})."
        err.message = message
    if isinstance(err, RateLimitError):
        wait = (
            f"Wait {err.retry_after:.0f}s and retry"
            if err.retry_after is not None
            else "Retry shortly"
        )
        fix = (
            f"{wait} (the SDK retries a short wait itself). A deployment takes "
            f"{_DEFAULT_RATE_LIMIT_RPM} requests a minute unless its inference_config "
            "sets rate_limit_rpm."
        )
    err.fix = fix
    return err


class DeploymentClient(Generic[R_co]):
    """Direct client for a single deployed model endpoint.

    Generic over the result type. ``DeploymentClient(..., task="object_detection")``
    is a ``DeploymentClient[DetectionResult]`` whose :meth:`infer` returns a
    :class:`~pictograph.DetectionResult`; without ``task`` it is a
    ``DeploymentClient[AnyResult]`` you narrow at the call site.

    ``timeout`` is per request: 150 s by default, longer than the gateway holds a
    scaled-to-zero deployment's first request while its container starts.
    ``max_retries`` bounds the retries of a ``429``, a ``5xx`` or a network failure.
    """

    @overload
    def __init__(
        self: DeploymentClient[DetectionResult],
        endpoint: str,
        api_key: str,
        *,
        task: Literal["object_detection"],
        timeout: float = DEFAULT_TIMEOUT_S,
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
        timeout: float = DEFAULT_TIMEOUT_S,
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
        timeout: float = DEFAULT_TIMEOUT_S,
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
        timeout: float = DEFAULT_TIMEOUT_S,
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
        timeout: float = DEFAULT_TIMEOUT_S,
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
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        task: TaskName | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
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
        # no-op sleep), else one seeded from max_retries that also honours the
        # gateway's Retry-After on a 503 (a deployment starting or at capacity).
        self._retry = retry_policy or RetryPolicy(
            max_retries=max_retries, retry_after_statuses=_RETRY_AFTER_STATUSES
        )
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
            class_filter: Restrict returned classes. ``None`` or ``[]`` returns every
                class the deployment is configured to return.
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
            ConflictError / ValidationError / ServerError: mapped from the
            gateway's HTTP status (see :func:`pictograph.exceptions.from_response`),
            with the gateway's token as ``code`` - ``ConflictError``
            ``deployment_not_active`` (409) before the deployment is active,
            ``ServerError`` ``deployment_starting`` (503) when a scaled-to-zero
            deployment's container did not start in time, ``overloaded`` (503),
            ``inference_timeout`` (504), ``deployment_unreachable`` (502).
            NetworkError / RequestTimeoutError: on a transport failure that
            outlived the retry budget.
            ServerError: the response carried no usable ``model_type``, or it
            contradicts the ``task`` this client was constructed with.
        """
        resp = self._request(image, confidence, class_filter, top_k)
        payload = self._parse(resp)
        return cast("R_co", self._to_result(payload, resp.headers))

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

    def _to_result(self, payload: dict[str, Any], headers: httpx.Headers) -> AnyResult:
        """Typed result from one endpoint payload, through the shared builder.

        The endpoint states its own ``model_type``; a declared ``task`` is
        verified against it rather than trusted over it, because parsing a
        classifier's ``{"class": …}`` entries as a detector's annotations would
        not fail loudly - it would quietly produce an empty result. The
        provenance (``backend``, ``device``, ``inference_ms``) is what the
        gateway's headers report.
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
        return build_result(
            payload,
            task=task,
            backend=backend_of(headers.get(_ENGINE_HEADER)),
            device=headers.get(_DEVICE_HEADER) or UNREPORTED_DEVICE,
            providers=(),
            inference_ms=server_timing_ms(headers.get(_TIMING_HEADER), "infer"),
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
        raise with_gateway_hint(
            from_response(
                resp.status_code,
                body=body,
                request_id=resp.headers.get("X-Request-Id"),
                headers=resp.headers,
            ),
            body,
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
