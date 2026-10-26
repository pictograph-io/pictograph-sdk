"""What a deployment's endpoint answers, read the same way for every call.

A deployment is called at its own URL, on the inference gateway in its region: its
``/predict`` endpoint (:class:`~pictograph.DeploymentClient`) and its streams
(:class:`~pictograph.DeploymentStream`). The gateway answers an error as a flat
``{"error": "<slug>", "detail"?: "..."}`` rather than the API's envelope, so this module
turns that into the SDK's typed exception, with the slug as its ``code`` and a ``fix``
saying what to do, and names the runtime that served a result.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any, Final, cast, get_args

from pictograph.exceptions import ApiError, RateLimitError, ServerError, from_response
from pictograph.inference.results import BackendName

if TYPE_CHECKING:
    import httpx

#: What ``device`` says when the gateway did not report the deployment's
#: hardware. Deliberately not ``"cpu"``: a GPU deployment reported as CPU is a
#: silent falsehood, and this field is read as provenance.
UNREPORTED_DEVICE = "remote"

#: What ``backend`` reads when the gateway names no engine: the gateway learned to
#: name it with PRT, and every gateway before that served the ONNX graph on ONNX Runtime.
_UNNAMED_ENGINE_BACKEND: Final[BackendName] = "onnxruntime"
_RUNTIME_NAMES: Final[frozenset[str]] = frozenset(get_args(BackendName))

#: How a ceiling is raised or removed - a deployment has none unless one is set.
_RAISE_THE_CEILING: Final = (
    "Its request ceiling is its own setting: raise it, or remove it with 0, through "
    "client.deployments.update(name, rate_limit_rpm=...) or its settings in the app."
)

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
    "wrong_region": (
        "The deployment runs in another region than this URL names.",
        "POST to the deployment's endpoint_url exactly as it is "
        "(client.deployments.get(name).endpoint_url): its path names the region whose gateway serves it.",
    ),
    "deployment_not_active": (
        "The deployment is not active.",
        "Resume a paused deployment (client.deployments.resume(...)), or wait for a "
        "provisioning one (client.deployments.wait_until_active(...)).",
    ),
    "rate_limited": (
        "The deployment's request ceiling was reached.",
        # with_gateway_hint replaces it with one that names the Retry-After.
        f"Retry shortly. {_RAISE_THE_CEILING}",
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

#: A stream's own error slugs, and the ones whose fix differs for a stream.
_STREAM_ERRORS: Final[dict[str, tuple[str, str]]] = {
    "not_found": (
        "The gateway has no such route.",
        "Open streams through DeploymentClient.open_stream: it calls the deployment's "
        "endpoint_url with /sessions in place of /predict.",
    ),
    "wrong_region": (
        "The deployment runs in another region than this URL names.",
        "Use the deployment's endpoint_url exactly as it is "
        "(client.deployments.get(name).endpoint_url): its path names the region whose gateway serves it.",
    ),
    "deployment_starting": (
        "The deployment's stream container did not start within the gateway's two minutes.",
        "Retry shortly: the next try finds the container on its way.",
    ),
    "invalid_source": (
        "The deployment refused the stream's source.",
        'Pass a camera URL (rtsp://, rtsps://, or a WebRTC camera\'s https:// WHEP URL), or "whip" '
        "to send the video yourself.",
    ),
    "unsafe_source": (
        "The camera's address is not public.",
        "The deployment connects to the camera itself, so the address must resolve to public "
        "addresses only.",
    ),
    "invalid_request_body": (
        "The request body could not be read.",
        'Send {"source": {"kind": "whip" | "rtsp" | "whep", "uri"?: ...}, "overlay"?: true}.',
    ),
    "invalid_option": (
        "An option was refused.",
        "overlay is true or false.",
    ),
    "request_too_large": (
        "The request is larger than the 256 KiB a stream's call takes.",
        "Send a smaller offer or body.",
    ),
    "stream_limit": (
        "The deployment is already running as many streams as it can.",
        "Stop a stream, or raise the deployment's max_containers (each container runs four).",
    ),
    "streams_unavailable": (
        "This deployment was deployed before streams.",
        "Pause and resume it once (client.deployments.pause, then resume) to add them.",
    ),
    "session_not_found": (
        "The stream does not exist.",
        "It ended, or was never opened: open a new one.",
    ),
    "session_ended": (
        "The stream ended.",
        "Open a new one.",
    ),
    "session_failed": (
        "The stream's engine did not start.",
        "Retry; the message says what the engine reported.",
    ),
    "session_timeout": (
        "The stream's engine did not answer in time.",
        "Retry.",
    ),
    "sessions_unavailable": (
        "Streams are briefly unavailable.",
        "Retry shortly.",
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


def with_gateway_hint(err: ApiError, body: Any, *, stream: bool = False) -> ApiError:
    """``err`` with the gateway's error slug as its ``code`` and a ``fix`` for it.

    The gateway answers a flat ``{"error": "<slug>", "detail"?: "..."}`` rather than the
    API's ``{"error": {"code", ...}}`` envelope, so without this the slug would reach the
    caller only as the message. ``stream=True`` reads it as a stream's call answered it.
    """
    slug = body.get("error") if isinstance(body, dict) else None
    if not isinstance(slug, str) or not slug:
        return err
    if err.code is None:
        err.code = slug
    hint = (_STREAM_ERRORS.get(slug) if stream else None) or _GATEWAY_ERRORS.get(slug)
    if hint is None:
        return err
    message, fix = hint
    if err.message == slug:  # the body carried no words of its own
        status = body.get("status") if isinstance(body, dict) else None
        if slug == "deployment_not_active" and isinstance(status, str) and status:
            message = f"The deployment is not active (status: {status})."
        err.message = message
    if isinstance(err, RateLimitError) and slug == "rate_limited":
        wait = (
            f"Wait {err.retry_after:.0f}s and retry"
            if err.retry_after is not None
            else "Retry shortly"
        )
        fix = f"{wait} (the SDK retries a short wait itself). {_RAISE_THE_CEILING}"
    err.fix = fix
    return err


def parse_response(resp: httpx.Response, *, stream: bool = False) -> dict[str, Any]:
    """The JSON of a 2xx answer, or the SDK's typed error for any other (with its fix)."""
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
        stream=stream,
    )
