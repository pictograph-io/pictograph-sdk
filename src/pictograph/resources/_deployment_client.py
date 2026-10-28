"""DeploymentClient - call a deployed model's endpoint directly.

Talks straight to the inference gateway (NOT through the Pictograph backend) for
minimum latency, authenticating with the per-deployment bearer token, and returns
**the same typed result classes a local model returns**::

    from pictograph import DeploymentClient, DetectionResult

    client = DeploymentClient(
        endpoint="https://workflows.pictograph.io/us-east/<slug>/predict",  # its endpoint_url
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

Inference is side-effect-free, so a transient gateway ``429`` (the deployment's request
ceiling, when its owner set one), ``5xx`` or network blip is retried with the same exponential-backoff
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
import math
from pathlib import Path
from typing import Any, Final, Generic, Literal, TypeVar, cast, overload

import httpx

from pictograph._http.retry import RetryPolicy
from pictograph.exceptions import (
    NetworkError,
    RequestTimeoutError,
    ServerError,
    ValidationError,
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
from pictograph.resources._deployment_stream import (
    DeploymentStream,
    RtspBufferMode,
    RtspReconnect,
    RtspTransport,
    open_stream as _open_stream,
    predict_url,
)
from pictograph.resources._gateway import UNREPORTED_DEVICE, backend_of, parse_response

R_co = TypeVar("R_co", bound=InferenceResult, covariant=True)

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

#: The retryable statuses whose ``Retry-After`` sets the wait: the deployment's request
#: ceiling (429, only when one is set), and a deployment that is starting or at capacity (503).
_RETRY_AFTER_STATUSES: Final = frozenset({429, 503})


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
            ServerError: the response carried no usable ``model_type``.
            ValidationError: the endpoint serves another task than the ``task`` this
                client was constructed with (refused before parsing; ``fix`` names it).
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

    def open_stream(
        self,
        source: str,
        *,
        overlay: bool = True,
        transport: RtspTransport | None = None,
        buffer_mode: RtspBufferMode | None = None,
        reconnect: RtspReconnect | None = None,
    ) -> DeploymentStream[R_co]:
        """Open a live stream through the deployment: video goes in, each frame's result comes out.

        Args:
            source: Where the video comes from. A camera's URL - ``rtsp://``, ``rtsps://``, or a
                WebRTC camera's ``https://`` WHEP URL - which the deployment connects to (its
                address must be public); or ``"whip"`` to send the video yourself, H.264 from
                any WHIP client, to the stream's ``whip_url``.
            overlay: Draw the results on the live view (the stream's ``whep_url``). Change it
                while the stream runs with :meth:`DeploymentStream.set_overlay`.
            transport: An RTSP camera's: how its video comes - ``"auto"`` (the default: UDP, then
                TCP when UDP brings nothing), ``"tcp"`` (over the RTSP connection: a camera's UDP
                seldom reaches a cloud container, and a TCP-only tunnel carries nothing else) or
                ``"udp"``.
            buffer_mode: An RTSP camera's: ``"auto"`` (the default) smooths its clock onto the
                deployment's; ``"none"`` keeps the sender's own timestamps - a recording streamed,
                timed as its file is.
            reconnect: An RTSP camera's: ``"always"`` (the default) tries a camera that drops or
                ends again; ``"never"`` - a recording - ends the stream at its end, its last frame
                decoded (the stream then ends with ``ended == "source_ended"``).

        Returns:
            The open :class:`~pictograph.DeploymentStream`: read its ``frames()`` (each one's
            ``result`` is the typed result :meth:`infer` returns), and stop it with ``stop()``
            or by opening it in a ``with`` block. It is billed as the deployment's uptime while
            it runs.

        A deployment that is asleep starts a stream container first (about 20 seconds). The
        gateway holds the open for up to two minutes; a ``503`` ``deployment_starting`` after
        that is tried again, up to ``max_retries`` times.

        Raises:
            ValueError: ``source`` is neither a camera URL nor ``"whip"``, or ``transport`` /
                ``buffer_mode`` / ``reconnect`` is given for a source that is not an RTSP camera,
                or is not one of its values.
            RateLimitError: ``stream_limit`` - the deployment already runs as many streams as
                it can (four per container, up to ``max_containers``).
            ConflictError: ``deployment_not_active``, or ``streams_unavailable`` (deployed
                before streams: pause and resume it once).
            ValidationError: ``invalid_source`` or ``unsafe_source`` (a camera address that is
                not public).
            ServerError: ``deployment_starting`` once the retries are spent, or
                ``session_failed`` (the stream's engine did not start; the message says why).
            NetworkError / RequestTimeoutError: on a transport failure.
        """
        return cast(
            "DeploymentStream[R_co]",
            _open_stream(
                self._http,
                endpoint=self._endpoint,
                headers=self._headers,
                task=self._task,
                retry=self._retry,
                source=source,
                overlay=overlay,
                transport=transport,
                buffer_mode=buffer_mode,
                reconnect=reconnect,
            ),
        )

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
                return self._http.post(
                    predict_url(self._endpoint), headers=self._headers, json=body
                )
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
                fix="The endpoint serves a model type this SDK does not know yet: upgrade it "
                "(pip install -U pictograph). infer_raw() returns the untouched body meanwhile.",
            )
        if self._task is not None and reported is not None and reported != self._task:
            # The CALLER's argument is wrong, not the server: an invalid-input refusal (a
            # ServerError here told an agent to retry later, and to contact support - B569).
            raise ValidationError(
                f"This deployment serves a {reported!r} model, but the client was "
                f"constructed with task={self._task!r}. Construct it with "
                f"task={reported!r} (or omit task) to read its predictions.",
                response=payload,
                fix=f"Construct the client with task={reported!r}, or omit task: the endpoint "
                "names its own.",
            )
        task = cast("TaskName | None", reported) or self._task
        if task is None:
            raise ServerError(
                "Deployment endpoint returned no model_type, so its task is "
                "unknown. Construct the client with task=… to parse it anyway, "
                "or use infer_raw() for the untouched body.",
                response=payload,
                fix="Construct the client with task=... to parse the body as that task, or "
                "call infer_raw() for the untouched body.",
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
        return parse_response(resp)

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
