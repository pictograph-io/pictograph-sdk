"""``pictograph deployments {list,get,compute-options,quote,create,wait,predict,pause,resume,delete}``.

Stand a trained model up as a live inference endpoint on a GPU, then call it::

    pictograph deployments create "Swift Falcon" --gpu t4 --wait   # the one-time token, once active
    pictograph deployments create "Swift Falcon" --gpu l4 --precision fp16
    pictograph deployments wait "Swift Falcon"                     # provisioning -> active
    pictograph deployments predict "Swift Falcon" photo.jpg --token pk_deploy_...
    pictograph deployments predict "Swift Falcon" https://example.com/x.jpg --token pk_deploy_... --confidence 0.4

A deployment runs on PRT, Pictograph's realtime engine (a TensorRT engine of the model,
for its GPU and precision). ``create`` answers at once, ``provisioning``; it turns
``active`` in seconds when the model's engine is already built, and a model's first
deployment on a GPU at a precision builds the engine first, a minute or more.

``list`` shows what each active deployment is doing right now - ``active · live (2)``,
``active · starting (1)``, ``active · waiting for a GPU 1m42s``, ``active · asleep`` - and
``get`` prints it as ``runtime``. Reading it never wakes a deployment.

``predict`` looks up the deployment's ``endpoint_url`` with your org key, then calls
that endpoint directly (minimum latency) with the per-deployment bearer token.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import typer

from pictograph import DeploymentClient
from pictograph.cli import _inputs
from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.models.deployment import DeploymentGpuType, DeploymentPrecision

if TYPE_CHECKING:
    from pictograph import Client
    from pictograph.models.deployment import CreatedDeployment, Deployment, DeploymentRuntime

app = command_group(no_args_is_help=True)


@app.command(
    "list",
    help="List model deployments in your organization, with what each active one is doing "
    "right now (live, starting, waiting for a GPU, asleep).",
)
def list_deployments(
    status: Annotated[
        str | None, typer.Option("--status", help="provisioning/active/paused/failed/terminated.")
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", min=1, help="How many to return (pages through the API as needed)."
        ),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    # Paged: one request takes at most 100 rows, and a larger --limit used to 422.
    deployments = list(
        client.deployments.iter(
            status=status,  # type: ignore[arg-type]
            page_size=min(limit, 100),
            max_total=limit,
        )
    )
    if json_output:
        print_json([d.model_dump(mode="json", exclude_none=True) for d in deployments])
        return
    rows = [
        {
            "name": d.name,
            "id": d.id,
            "status": _status(d),
            "gpu": d.gpu_type or d.compute_type,
            "precision": d.precision,
            "engine": _build_state(d),
            "endpoint_url": d.endpoint_url,
        }
        for d in deployments
    ]
    # The name is what the next command takes: never wrapped into two halves.
    print_table(rows, title=f"Deployments ({len(rows)})", no_wrap=("name",))


@app.command(
    "get", help="Fetch a single deployment by name, with its live state when it is active."
)
def get_deployment(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.deployments.get(deployment)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command(
    "compute-options",
    help="The GPU tiers a deployment runs on (container, price) and the precisions it takes.",
)
def compute_options(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    offer = client.deployments.compute_options()
    if json_output:
        print_json(offer.model_dump(mode="json"))
        return
    rows = [
        {
            "key": o.key,
            "label": o.label,
            "gpu": o.gpu_type or o.compute_type,
            "vram": f"{o.vram_gb} GB" if o.vram_gb else "",
            "cpu_cores": o.cpu_cores,
            "memory": f"{o.memory_gib:g} GiB" if o.memory_gib else "",
            "rate_per_min_micro_usd": o.rate_per_min_micro_usd,
            "cost_per_hour_micro_usd": o.cost_per_hour_micro_usd,
        }
        for o in offer.options
    ]
    print_table(rows, title=f"Compute options ({len(rows)})")
    if offer.precisions:
        print_table(
            [
                {
                    "precision": p.precision,
                    "label": p.label,
                    "default": "yes" if p.precision == _DEFAULT_PRECISION else "",
                    "description": p.description,
                }
                for p in offer.precisions
            ],
            title="Precisions (--precision on create)",
        )


_DEFAULT_PRECISION: DeploymentPrecision = "fp32"
_GPU_HELP = "t4 (default), l4 or a100. There is no CPU tier."
_PRECISION_HELP = (
    "fp32 (default) serves the model as trained; fp16 is faster, accuracy-equivalent, "
    "not output-identical."
)
_MIN_HELP = "Containers kept warm (0 = scale to zero, billed only while one is up)."
_MAX_HELP = "Autoscale ceiling."
_WAIT_HELP = (
    "Wait until it is active (a model's first deployment on a GPU at a precision builds "
    "its TensorRT engine first, a minute or more)."
)
_TIMEOUT_HELP = "Max seconds to wait (with --wait)."


def _gpu(gpu: str | None) -> DeploymentGpuType:
    """``--gpu`` checked before any request."""
    choice = (gpu or "t4").strip().lower()
    tiers = get_args(DeploymentGpuType)
    if choice not in tiers:
        raise typer.BadParameter(
            f"{gpu!r} is not a deployment GPU: use {', '.join(tiers)} "
            "(`pictograph deployments compute-options` lists them with prices). "
            "There is no CPU tier.",
            param_hint="--gpu",
        )
    return cast("DeploymentGpuType", choice)


def _precision(precision: str) -> DeploymentPrecision:
    """``--precision`` checked before any request."""
    choice = precision.strip().lower()
    allowed = get_args(DeploymentPrecision)
    if choice not in allowed:
        raise typer.BadParameter(
            f"{precision!r}: use {' or '.join(allowed)}.", param_hint="--precision"
        )
    return cast("DeploymentPrecision", choice)


def _building(deployment: Deployment) -> bool:
    """Whether the deployment waits on its model's TensorRT engine being built."""
    return deployment.engine_build is not None and deployment.engine_build.state == "building"


def _estimate_s(deployment: Deployment) -> float | None:
    """How long builds like the one it waits on took before (None: no history)."""
    build = deployment.engine_build
    estimates = [m.estimate_s for m in (build.models if build else []) if m.estimate_s is not None]
    return max(estimates) if estimates else None


def _build_state(deployment: Deployment) -> str:
    """The engine build a deployment shows, in a few words ("" when there is none)."""
    build = deployment.engine_build
    if build is None:
        return ""
    if _building(deployment):
        estimate = _estimate_s(deployment)
        return f"building (about {estimate:.0f}s)" if estimate is not None else "building"
    times = [m.build_s for m in build.models if m.build_s is not None]
    return f"built in {sum(times):.0f}s" if times else build.state


def _waited(runtime: DeploymentRuntime) -> str:
    """How long a request has waited for a container ("" when the wait has no start), on the
    server's clock when it says the time it read it."""
    if runtime.waiting_since is None:
        return ""
    now = runtime.checked_at or datetime.now(timezone.utc)
    seconds = max(0, int((now - runtime.waiting_since).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _status(deployment: Deployment) -> str:
    """The status a person reads: an ``active`` deployment with what it is doing right now -
    ``active · live (2)``, ``active · asleep``, ``active · waiting for a GPU 1m42s``."""
    runtime = deployment.runtime
    if deployment.status != "active" or runtime is None:
        return deployment.status
    if runtime.state == "live":
        starting = f", {runtime.creating} starting" if runtime.creating else ""
        live = f"live ({runtime.containers}{starting})"
    elif runtime.state == "starting":
        live = f"starting ({max(runtime.creating, 1)})"
    elif runtime.state == "waiting":
        live = " ".join(filter(None, ("waiting for a GPU", _waited(runtime))))
    else:  # asleep, unknown, or a state added later: its own word
        live = runtime.state
    return f"{deployment.status} · {live}"


def _summary(deployment: Deployment) -> dict[str, Any]:
    """What ``create`` / ``wait`` / ``resume`` print about a deployment."""
    out: dict[str, Any] = {
        "id": deployment.id,
        "name": deployment.name,
        "status": deployment.status,
        "gpu_type": deployment.gpu_type,
        "precision": deployment.precision,
        "endpoint_url": deployment.endpoint_url,
    }
    if deployment.engine_build is not None:
        out["engine_build"] = deployment.engine_build.model_dump(mode="json", exclude_none=True)
    if deployment.runtime is not None:
        out["runtime"] = deployment.runtime.model_dump(mode="json")
    if deployment.error_message:
        out["error_message"] = deployment.error_message
    return out


def _created(created: CreatedDeployment, deployment: Deployment) -> dict[str, Any]:
    """What ``create`` prints: the deployment, and its one-time token."""
    return {**_summary(deployment), "auth_token": created.auth_token}


def _wait_for(client: Client, deployment: Deployment, timeout: float) -> Deployment:
    """Wait for a provisioning deployment to turn active, saying on stderr what it waits on."""
    if _building(deployment):
        estimate = _estimate_s(deployment)
        about = f" (about {estimate:.0f}s)" if estimate is not None else ""
        err_console.print(
            f"Building the model's TensorRT engine for {deployment.gpu_type} at "
            f"{deployment.precision} first{about}, then starting {deployment.name}..."
        )
    else:
        err_console.print(f"Waiting for {deployment.name} to turn active...")
    return client.deployments.wait_until_active(deployment, timeout=timeout)


@app.command("quote", help="Cost quote (micro-USD) for a deployment before creating it.")
def quote(
    gpu_type: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    min_containers: Annotated[
        int, typer.Option("--min-containers", "--min", min=0, help=_MIN_HELP)
    ] = 0,
    max_containers: Annotated[
        int, typer.Option("--max-containers", "--max", min=1, help=_MAX_HELP)
    ] = 1,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    gpu = _gpu(gpu_type)
    client = get_client(api_key)
    q = client.deployments.quote(
        gpu_type=gpu, min_containers=min_containers, max_containers=max_containers
    )
    print_json(q.model_dump(mode="json"))


@app.command(
    "create",
    help="Deploy a trained model to a live endpoint on a GPU (prints a one-time token).",
)
def create_deployment(
    model: Annotated[
        str, typer.Argument(help="Trained model name (status must be 'ready'). A UUID also works.")
    ],
    name: Annotated[str | None, typer.Option("--name")] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    precision: Annotated[
        str, typer.Option("--precision", help=_PRECISION_HELP)
    ] = _DEFAULT_PRECISION,
    min_containers: Annotated[
        int, typer.Option("--min", "--min-containers", min=0, help=_MIN_HELP)
    ] = 0,
    max_containers: Annotated[
        int, typer.Option("--max", "--max-containers", min=1, help=_MAX_HELP)
    ] = 1,
    scaledown_window: Annotated[
        int, typer.Option("--scaledown", help="Idle seconds before scale-down.")
    ] = 60,
    wait: Annotated[bool, typer.Option("--wait", help=_WAIT_HELP)] = False,
    timeout: Annotated[float, typer.Option("--timeout", help=_TIMEOUT_HELP)] = 900.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    gpu_type = _gpu(gpu)
    chosen_precision = _precision(precision)
    client = get_client(api_key)
    created = client.deployments.create(
        model,
        name=name,
        gpu_type=gpu_type,
        precision=chosen_precision,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    err_console.print(
        "[bold yellow]Save this token now - the API won't show it again "
        "(a member can reveal it from the deployment's settings in the app).[/bold yellow]"
    )
    if not wait:
        print_json(_created(created, created.deployment))
        return
    try:
        active = _wait_for(client, created.deployment, timeout)
    except BaseException:
        # The token is shown once: a wait that fails (or is interrupted) still prints it.
        print_json(_created(created, created.deployment))
        raise
    print_json(_created(created, active))


@app.command(
    "wait",
    help="Wait until a deployment is active (it provisions after create and resume).",
)
def wait_deployment(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 900.0,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between status checks.")
    ] = 3.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    active = client.deployments.wait_until_active(
        deployment, timeout=timeout, poll_interval=poll_interval
    )
    print_json(_summary(active))


@app.command(
    "predict",
    help="Run images, a video's frames (--video) or a dataset's images (--dataset, "
    "read-only) through a deployment's endpoint. Images can be paths, globs, URLs "
    "or - (stdin).",
)
def predict(
    deployment: Annotated[
        str,
        typer.Argument(
            help="Deployment name (a UUID also works). Needs an account API key "
            "to resolve; pass --endpoint instead to call the URL directly with "
            "only --token."
        ),
    ],
    # Declared before IMAGES only because IMAGES is now optional (--video and
    # --dataset take its place) and Python wants required parameters first; the
    # command line is unchanged: DEPLOYMENT IMAGES... --token ...
    token: Annotated[
        str,
        typer.Option(
            "--token", help="The per-deployment bearer token (pk_deploy_...) from create."
        ),
    ],
    images: Annotated[list[str] | None, typer.Argument(help=_inputs.HELP)] = None,
    confidence: Annotated[
        float | None, typer.Option("--confidence", help="Override the confidence threshold.")
    ] = None,
    class_filter: Annotated[
        str | None, typer.Option("--class-filter", help="Comma-separated classes to keep.")
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Classifier: number of predictions.")
    ] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="The deployment's /predict URL. With this, --token is the ONLY "
            "credential needed - no account API key, matching DeploymentClient "
            "(--dataset still needs one: reading a dataset is an account call).",
        ),
    ] = None,
    video: Annotated[list[str] | None, typer.Option("--video", help=_inputs.VIDEO_HELP)] = None,
    sample_fps: Annotated[
        float | None,
        typer.Option("--sample-fps", min=_inputs.MIN_SAMPLE_FPS, help=_inputs.SAMPLE_FPS_HELP),
    ] = None,
    dataset: Annotated[str | None, typer.Option("--dataset", help=_inputs.DATASET_HELP)] = None,
    directory: Annotated[
        str | None, typer.Option("--directory", help=_inputs.DIRECTORY_HELP)
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help=_inputs.LIMIT_HELP)] = None,
    jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl",
            help="One JSON object per image or frame, one per line: {input, result} or "
            "{input, error}. Always used for several images, a video or a dataset.",
        ),
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    source = _inputs.parse_source(
        images,
        video=video,
        sample_fps=sample_fps,
        dataset=dataset,
        directory=directory,
        limit=limit,
    )
    # A `pk_deploy_` token is meant to stand alone: it authorizes exactly one
    # endpoint and is what the cURL and Python snippets use by itself. Resolving
    # a name, though, is an ACCOUNT operation, so that form additionally needs
    # an account key - and so does reading a dataset. Given --endpoint and no
    # dataset there is nothing to resolve, so the deployment token alone is
    # sufficient here too.
    client = get_client(api_key) if (not endpoint or source.kind == "dataset") else None
    items = _inputs.open_items(source, client, frames="jpeg")
    if endpoint:
        conn: Any = DeploymentClient(endpoint=endpoint, api_key=token)
    else:
        account = client if client is not None else get_client(api_key)
        conn = account.deployments.connect(account.deployments.get(deployment), token)
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] if class_filter else None

    # The endpoint's own JSON, not the typed result's dump: someone reaching for
    # the CLI on a deployment is inspecting what the endpoint actually returns.
    # The typed shape is what the Python API gives you (`DeploymentClient.infer`).
    def run(image: Any) -> Any:
        return conn.infer_raw(image, confidence=confidence, class_filter=classes, top_k=top_k)

    if jsonl or not source.single_image:
        _inputs.emit(items, run)
        return
    (item,) = list(items)
    print_json(run(item.load()))


@app.command("pause", help="Pause one or more deployments (stops compute + billing).")
def pause_deployment(
    deployments: Annotated[
        list[str], typer.Argument(help="One or more deployment names (UUIDs also work).")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if len(deployments) == 1:
        deployment = client.deployments.pause(deployments[0])
        print_json({"id": deployment.id, "status": deployment.status})
    else:
        # One server-side bulk call - never N fan-out requests.
        print_json(client.deployments.bulk_pause(deployments).model_dump(mode="json"))


@app.command(
    "resume", help="Resume one or more paused deployments (each provisions again, then is active)."
)
def resume_deployment(
    deployments: Annotated[
        list[str], typer.Argument(help="One or more deployment names (UUIDs also work).")
    ],
    wait: Annotated[
        bool, typer.Option("--wait", help="Wait until each resumed deployment is active.")
    ] = False,
    timeout: Annotated[float, typer.Option("--timeout", help=_TIMEOUT_HELP)] = 900.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if len(deployments) == 1:
        deployment = client.deployments.resume(deployments[0])
        if wait:
            deployment = _wait_for(client, deployment, timeout)
        print_json({"id": deployment.id, "status": deployment.status})
        return
    result = client.deployments.bulk_resume(deployments)
    if wait:
        # The bulk call answers ids; each one it resumed is waited for in turn.
        for deployment_id in result.succeeded:
            _wait_for(client, client.deployments.get(deployment_id), timeout)
    print_json(result.model_dump(mode="json"))


@app.command(
    "delete", help="Terminate one or more deployments and tear down their serving endpoints."
)
def delete_deployment(
    deployments: Annotated[
        list[str], typer.Argument(help="One or more deployment names (UUIDs also work).")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Terminate {len(deployments)} deployment(s)?"):
        raise typer.Abort()
    client = get_client(api_key)
    if len(deployments) == 1:
        client.deployments.delete(deployments[0])
        print_json({"id": deployments[0], "terminated": True})
    else:
        print_json(client.deployments.bulk_delete(deployments).model_dump(mode="json"))
