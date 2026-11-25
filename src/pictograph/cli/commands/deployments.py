"""``pictograph deployments {list,get,edge-session,compute-options,quote,create,wait,predict,stream,pause,resume,update,delete}``.

Stand a trained model up as a live inference endpoint on a GPU, then call it::

    pictograph deployments create "Swift Falcon" --gpu t4 --wait   # the one-time token, once active
    pictograph deployments create "Swift Falcon" --gpu l4 --precision fp16
    pictograph deployments create "Swift Falcon" --environment edge --wait   # in the browser that opens it, FP16
    pictograph deployments edge-session "Swift Falcon"             # what that browser runs it with
    pictograph deployments wait "Swift Falcon"                     # provisioning -> active
    pictograph deployments predict "Swift Falcon" photo.jpg --token pk_deploy_...
    pictograph deployments predict "Swift Falcon" https://example.com/x.jpg --token pk_deploy_... --confidence 0.4
    pictograph deployments stream "Swift Falcon" rtsp://camera.example.com/live --token pk_deploy_...
    pictograph deployments update "Swift Falcon" --min-containers 1 --max-containers 3
    pictograph deployments update "Swift Falcon" --name dock-gate --readme "Gate cameras, 6am-10pm."
    pictograph deployments update "Swift Falcon" --rate-limit 6000   # 0 removes the ceiling

A deployment serves a TensorRT engine of the model, built for its GPU and precision. ``create`` answers at once, ``provisioning``; it turns
``active`` in about a minute when the model's engine is already built, and a model's first
deployment on a GPU at a precision builds the engine first, a few minutes.

``update`` changes it after it was created - its ``--name`` and ``--readme``, its autoscaling
(``--min-containers``, ``--max-containers``, ``--scaledown-window``), its request ceiling
(``--rate-limit``), only what is given. A live deployment takes a scaling change at once, with no
redeploy; a paused one when it is resumed. A deployment has no request ceiling unless one is set.

It is billed for its uptime and for its requests' data transfer (each image sent through its
endpoint and its answer), per GiB: ``quote`` and ``compute-options`` show both prices.

``list`` shows what each active deployment is doing right now - ``active · live (2)``,
``active · starting (1)``, ``active · waiting for a GPU 1m42s``, ``active · asleep`` - and
``get`` prints it as ``runtime``. Reading it never wakes a deployment. ``active · live (1,
elsewhere in the US)``: its region had no GPU free for 30 seconds, and the rest of the US
served it.

``predict`` looks up the deployment's ``endpoint_url`` with your org key, then calls
that endpoint directly (minimum latency) with the per-deployment bearer token.

``stream`` streams live video through the deployment - a camera it connects to, or ``whip``
to send the video yourself - and prints each frame's result as it is inferred, one JSON line
per frame, until Ctrl-C, ``--frames`` or ``--seconds``; the stream stops when the command does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import typer

from pictograph import DeploymentClient
from pictograph.cli import _inputs, _live_log
from pictograph.cli._client import get_client
from pictograph.cli._draw import DRAW_HELP, LOOK_HELP, Drawer
from pictograph.cli._endpoint import (
    MaxChange,
    MaxContainers,
    MinChange,
    MinContainers,
    Region,
    ScaledownChange,
    ScaledownWindow,
    build_state,
    live_status,
    run_stream,
    say_token_once,
    say_waiting,
)
from pictograph.cli._errors import command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.models.deployment import (
    DeploymentEnvironment,
    DeploymentGpuType,
    DeploymentPrecision,
)

if TYPE_CHECKING:
    from pictograph import Client
    from pictograph.models.deployment import CreatedDeployment, Deployment

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
            "status": live_status(d.status, d.runtime),
            "region": d.region or "",
            "gpu": d.gpu_type or d.compute_type,
            "precision": d.precision,
            "engine": build_state(d.engine_build),
            "endpoint_url": d.endpoint_url,
        }
        for d in deployments
    ]
    # The name is what the next command takes: never wrapped into two halves.
    print_table(rows, title=f"Deployments ({len(rows)})", no_wrap=("name",))


@app.command(
    "logs",
    help="The deployment's Live log: every session's and device's frames, events, captures, stats and streams' "
    "status. --live prints each record as it arrives, until Ctrl-C; without it, the history, oldest first. A table "
    "by default; --json prints one record a line for jq.",
)
def deployment_logs(
    deployment: Annotated[str, typer.Argument(help="Deployment name.")],
    live: Annotated[bool, _live_log.live_option()] = False,
    since: Annotated[str | None, typer.Option("--since", help=_live_log.SINCE_HELP)] = None,
    until: Annotated[str | None, typer.Option("--until", help=_live_log.UNTIL_HELP)] = None,
    kind: Annotated[list[str] | None, typer.Option("--kind", help=_live_log.KIND_HELP)] = None,
    class_name: Annotated[
        list[str] | None, typer.Option("--class", help=_live_log.CLASS_HELP)
    ] = None,
    min_confidence: Annotated[
        float | None,
        typer.Option("--min-confidence", min=0.0, max=1.0, help=_live_log.CONFIDENCE_HELP),
    ] = None,
    session: Annotated[
        str | None, typer.Option("--session", help="One stream's records: the stream's id.")
    ] = None,
    every: Annotated[int | None, typer.Option("--every", min=1, help=_live_log.EVERY_HELP)] = None,
    cursor: Annotated[str | None, typer.Option("--cursor", help=_live_log.CURSOR_HELP)] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", min=0, help=_live_log.LIMIT_HELP)
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help=_live_log.JSON_HELP)] = False,
    download: Annotated[
        Path | None, typer.Option("--download", help=_live_log.DOWNLOAD_HELP, file_okay=False)
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    stream = _live_log.records(
        client.deployments.logs,
        deployment,
        live=live,
        since=since,
        until=until,
        kinds=kind,
        classes=class_name,
        min_confidence=min_confidence,
        session=session,
        every=every,
        cursor=cursor,
    )
    _live_log.show(
        stream,
        title=f"Live log of {deployment}",
        json_output=json_output,
        limit=limit,
        download=download,
        live=live,
    )


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
    "edge-session",
    help="What a browser runs an edge deployment with: its ONNX graph (a URL good for 60 "
    "minutes), its config, inference defaults, look and log.",
)
def edge_session(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    session = client.deployments.edge_session(deployment)
    print_json(session.model_dump(mode="json", exclude_none=True))


@app.command(
    "compute-options",
    help="The regions a deployment runs in, its GPU tiers (container, price there) and the precisions it takes.",
)
def compute_options(
    region: Annotated[
        str,
        typer.Option(
            "--region", help="Price for this region (see the list); auto = the one nearest you."
        ),
    ] = "auto",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    offer = client.deployments.compute_options(region=region)
    if json_output:
        print_json(offer.model_dump(mode="json"))
        return
    if offer.regions:
        print_table(
            [
                {
                    "region": r.id,
                    "label": r.label,
                    "location": r.location,
                    "priced": "yes" if r.id == offer.region else "",
                }
                for r in offer.regions
            ],
            title="Regions (--region on create)",
        )
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
    print_table(
        rows, title=f"Compute options ({len(rows)}) in {offer.region or 'the default region'}"
    )
    if offer.transfer_per_gib_micro_usd is not None:
        err_console.print(
            f"Plus data transfer: {offer.transfer_per_gib_micro_usd} micro-USD a GiB (each image "
            "sent through the endpoint and its answer), the same on every GPU and in every region."
        )
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
    "fp32 serves the model as trained; fp16 is faster, accuracy-equivalent, not output-identical. "
    "Default: fp32 in the cloud, fp16 on the edge (its FP16 graph built first when the model has "
    "none, never billed)."
)
_RATE_LIMIT_HELP = (
    "Its request ceiling: requests a minute past which its endpoint answers 429. None by default."
)
_WAIT_HELP = (
    "Wait until it is active (a model's first deployment on a GPU at a precision builds "
    "its TensorRT engine first, a few minutes; on the edge, a model's first at FP16 builds its "
    "FP16 graph first)."
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


def _precision(precision: str | None) -> DeploymentPrecision | None:
    """``--precision`` checked before any request (None: the environment's default)."""
    if precision is None:
        return None
    choice = precision.strip().lower()
    allowed = get_args(DeploymentPrecision)
    if choice not in allowed:
        raise typer.BadParameter(
            f"{precision!r}: use {' or '.join(allowed)}.", param_hint="--precision"
        )
    return cast("DeploymentPrecision", choice)


def _summary(deployment: Deployment) -> dict[str, Any]:
    """What ``create`` / ``wait`` / ``resume`` print about a deployment."""
    out: dict[str, Any] = {
        "id": deployment.id,
        "name": deployment.name,
        "status": deployment.status,
        "environment": deployment.environment,
        "region": deployment.region,
        "gpu_type": deployment.gpu_type,
        "precision": deployment.precision,
        "endpoint_url": deployment.endpoint_url,
    }
    if deployment.rate_limit_rpm is not None:  # none unless set: said only when there is one
        out["rate_limit_rpm"] = deployment.rate_limit_rpm
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


def _scaled(deployment: Deployment) -> dict[str, Any]:
    """What ``update`` prints: the deployment, its README, the autoscaling and the request
    ceiling (null: none) it has now."""
    return {
        **_summary(deployment),
        "readme": deployment.readme,
        "min_containers": deployment.min_containers,
        "max_containers": deployment.max_containers,
        "scaledown_window": deployment.scaledown_window,
        "rate_limit_rpm": deployment.rate_limit_rpm,
    }


def _wait_for(client: Client, deployment: Deployment, timeout: float) -> Deployment:
    """Wait for a provisioning deployment to turn active, saying on stderr what it waits on."""
    building = (
        f"the model's {(deployment.precision or 'fp16').upper()} graph"
        if deployment.environment == "edge"
        else f"the model's TensorRT engine for {deployment.gpu_type} at {deployment.precision}"
    )
    say_waiting(deployment.name, deployment.engine_build, building=building, until="turn active")
    return client.deployments.wait_until_active(deployment, timeout=timeout)


@app.command("quote", help="Cost quote (micro-USD) for a deployment before creating it.")
def quote(
    gpu_type: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    min_containers: MinContainers = 0,
    max_containers: MaxContainers = 1,
    region: Region = "auto",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    gpu = _gpu(gpu_type)
    client = get_client(api_key)
    q = client.deployments.quote(
        gpu_type=gpu, min_containers=min_containers, max_containers=max_containers, region=region
    )
    print_json(q.model_dump(mode="json"))


@app.command(
    "create",
    help="Deploy a trained model to a live endpoint on a GPU (prints a one-time token), or to "
    "the edge (--environment edge: it runs in the browser of whoever opens it).",
)
def create_deployment(
    model: Annotated[
        str, typer.Argument(help="Trained model name (status must be 'ready'). A UUID also works.")
    ],
    name: Annotated[str | None, typer.Option("--name")] = None,
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            help=(
                "cloud (a GPU in its region, billed by the minute) or edge (the browser of whoever "
                "opens it, on that device's GPU: no region, GPU or scaling, nothing billed)."
            ),
        ),
    ] = "cloud",
    region: Region = "auto",
    gpu: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    precision: Annotated[str | None, typer.Option("--precision", help=_PRECISION_HELP)] = None,
    min_containers: MinContainers = 0,
    max_containers: MaxContainers = 1,
    scaledown_window: ScaledownWindow = 60,
    rate_limit: Annotated[
        int | None, typer.Option("--rate-limit", min=1, help=_RATE_LIMIT_HELP)
    ] = None,
    wait: Annotated[bool, typer.Option("--wait", help=_WAIT_HELP)] = False,
    timeout: Annotated[float, typer.Option("--timeout", help=_TIMEOUT_HELP)] = 900.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    env = environment.strip().lower()
    if env not in ("cloud", "edge"):
        raise typer.BadParameter(f"{environment!r}: use cloud or edge.", param_hint="--environment")
    if env == "edge" and gpu is not None:
        raise typer.BadParameter(
            "an edge deployment runs on the viewer's own GPU", param_hint="--gpu"
        )
    gpu_type = _gpu(gpu)
    chosen_precision = _precision(precision)
    client = get_client(api_key)
    created = client.deployments.create(
        model,
        name=name,
        environment=cast("DeploymentEnvironment", env),
        region=region,
        gpu_type=gpu_type,
        precision=chosen_precision,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
        rate_limit_rpm=rate_limit,
    )
    if created.auth_token is None:  # an edge deployment: nothing calls it, so no token
        if wait and created.deployment.status == "provisioning":  # its FP16 graph is being built
            print_json(_created(created, _wait_for(client, created.deployment, timeout)))
            return
        print_json(_created(created, created.deployment))
        return
    say_token_once("the deployment's Analytics")
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
    draw: Annotated[str | None, typer.Option("--draw", help=DRAW_HELP)] = None,
    look: Annotated[str | None, typer.Option("--look", help=LOOK_HELP)] = None,
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
    # Checked before anything runs: a target that cannot hold what is drawn, a look that is no look.
    drawer = Drawer.open(draw, source, look)
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
        conn = account.deployments.connect(account.deployments.get(deployment), api_key=token)
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] if class_filter else None

    # The endpoint's own JSON, not the typed result's dump: someone reaching for
    # the CLI on a deployment is inspecting what the endpoint actually returns.
    # The typed shape is what the Python API gives you (`DeploymentClient.infer`).
    def run(image: Any) -> Any:
        return conn.infer_raw(image, confidence=confidence, class_filter=classes, top_k=top_k)

    if jsonl or not source.single_image:
        try:
            _inputs.emit(
                items,
                run,
                after=(lambda item, image, answer: {"drawn": drawer.draw(item, image, answer)})
                if drawer is not None
                else None,
            )
        finally:
            if drawer is not None:
                drawer.close()
        return
    (item,) = list(items)
    loaded = item.load()
    answer = run(loaded)
    if drawer is not None:
        err_console.print(f"Drawn: {drawer.draw(item, loaded, answer)}")
    print_json(answer)


@app.command(
    "stream",
    help="Stream live video through a deployment and print each frame's result as it is "
    "inferred, one JSON line per frame. SOURCE is a camera the deployment connects to "
    "(rtsp://, rtsps://, or a WebRTC camera's https:// WHEP URL), or whip to send the video "
    "yourself. Ctrl-C stops it; the stream is billed while it runs.",
)
def stream(
    deployment: Annotated[
        str,
        typer.Argument(
            help="Deployment name (a UUID also works). Needs an account API key to resolve; "
            "pass --endpoint instead to use only --token."
        ),
    ],
    source: Annotated[
        str,
        typer.Argument(help="rtsp://..., rtsps://..., an https:// WHEP URL, or whip."),
    ],
    token: Annotated[
        str,
        typer.Option(
            "--token", help="The per-deployment bearer token (pk_deploy_...) from create."
        ),
    ],
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="The deployment's /predict URL. With this, --token is the only credential needed.",
        ),
    ] = None,
    frames: Annotated[
        int | None, typer.Option("--frames", min=1, help="Stop after this many frames.")
    ] = None,
    seconds: Annotated[
        float | None, typer.Option("--seconds", min=1, help="Stop after this many seconds.")
    ] = None,
    overlay: Annotated[
        bool,
        typer.Option("--overlay/--no-overlay", help="Draw the results on the live view."),
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Send the live view's pictures. --no-stream: none are drawn or encoded; each "
            "frame's result still comes.",
        ),
    ] = True,
    confidence: Annotated[
        float | None,
        typer.Option(
            "--confidence",
            min=0.0,
            max=1.0,
            help="The model's confidence as the stream opens, over 0 and at most 1 (absent: the "
            "deployment's own).",
        ),
    ] = None,
    all_messages: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Print every message the stream sends (its numbers each second, its events), "
            "not only its frames.",
        ),
    ] = False,
    idle_timeout: Annotated[
        float | None,
        typer.Option(
            "--idle-timeout",
            min=0.1,
            help="Fail after this many seconds without a frame (exit 9, stream_stalled): its camera "
            "stalled or its connection dropped. The stream stops with it.",
        ),
    ] = None,
    transport: Annotated[
        str | None,
        typer.Option(
            "--transport",
            help="An rtsp:// camera's: tcp (the default: over the RTSP connection - a camera's "
            "UDP seldom reaches a cloud container), auto (UDP first, then TCP) or udp.",
        ),
    ] = None,
    buffer_mode: Annotated[
        str | None,
        typer.Option(
            "--buffer-mode",
            help="An rtsp:// camera's: auto smooths its clock onto the deployment's; none keeps "
            "the sender's own timestamps (a recording streamed).",
        ),
    ] = None,
    reconnect: Annotated[
        str | None,
        typer.Option(
            "--reconnect",
            help="An rtsp:// camera's: always (a camera that drops or ends is tried again) or "
            "never (a recording: the stream ends at its end, its last frame decoded).",
        ),
    ] = None,
    outputs: Annotated[
        str | None,
        typer.Option(
            "--outputs",
            help="What the stream writes beside its log (JSON, or @path to a JSON file): its Passive "
            "capture alone - its frames and the model's predictions every every_s seconds, into a dataset "
            'folder, e.g. \'{"passive_capture": {"dataset": "Yard", "every_s": 10}}\'.',
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    stream_outputs = deployment_outputs_option(outputs)
    if endpoint:
        conn: Any = DeploymentClient(endpoint=endpoint, api_key=token)
    else:
        account = get_client(api_key)
        conn = account.deployments.connect(account.deployments.get(deployment), api_key=token)
    run_stream(
        conn,
        source,
        overlay=overlay,
        stream=stream,
        frames=frames,
        seconds=seconds,
        all_messages=all_messages,
        idle_timeout=idle_timeout,
        transport=transport,
        buffer_mode=buffer_mode,
        reconnect=reconnect,
        outputs=stream_outputs,
        confidence=confidence,
    )


def deployment_outputs_option(text: str | None) -> dict[str, Any] | None:
    """``--outputs`` for a deployment's stream: read and checked as a run's are (``workflows run --outputs``), then held
    to what a deployment writes - its Passive capture alone; anything else is a usage error before any request."""
    from pictograph.cli.commands.workflows import outputs_option
    from pictograph.resources._outputs import deployment_outputs_body

    given = outputs_option(text)
    try:
        return deployment_outputs_body(given)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--outputs") from None


@app.command(
    "edge-stream",
    help="Open a stream of an edge deployment, to run it yourself in the edge engine: its spec, its "
    "model's ONNX graph and config, its outputs, its own token, where its events go and its log.",
)
def edge_stream(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    outputs: Annotated[
        str | None,
        typer.Option(
            "--outputs",
            help="What the stream writes beside its log (JSON, or @path): its Passive capture alone. "
            "Writing takes a member or above.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    stream_outputs = deployment_outputs_option(outputs)
    client = get_client(api_key)
    session = client.deployments.edge_stream(deployment, outputs=stream_outputs)
    print_json(session.model_dump(mode="json", exclude_none=True))


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
        print_json(_summary(deployment))
        return
    result = client.deployments.bulk_resume(deployments)
    if wait:
        # The bulk call answers ids; each one it resumed is waited for in turn.
        for deployment_id in result.succeeded:
            _wait_for(client, client.deployments.get(deployment_id), timeout)
    print_json(result.model_dump(mode="json"))


@app.command(
    "update",
    help="Change a deployment: its name, its README, its autoscaling (min / max containers, its "
    "idle window), its request ceiling. A live deployment takes a scaling change at once, with no "
    "redeploy; a paused one when it is resumed.",
)
def update_deployment(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    min_containers: MinChange = None,
    max_containers: MaxChange = None,
    scaledown_window: ScaledownChange = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Its new name (unique among your live deployments). Its endpoint and token stay.",
        ),
    ] = None,
    readme: Annotated[
        str | None, typer.Option("--readme", help='Its README, Markdown. "" clears it.')
    ] = None,
    rate_limit: Annotated[
        int | None,
        typer.Option("--rate-limit", min=0, help=f"{_RATE_LIMIT_HELP} 0 removes it."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    given = (min_containers, max_containers, scaledown_window, name, readme, rate_limit)
    if all(v is None for v in given):
        raise typer.BadParameter(
            "nothing to change: give --name, --readme, --min-containers, --max-containers, "
            "--scaledown-window or --rate-limit.",
            param_hint="--name",
        )
    client = get_client(api_key)
    updated = client.deployments.update(
        deployment,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
        name=name,
        readme=readme,
        rate_limit_rpm=rate_limit,
    )
    # A live deployment took its scaling at once. One that is not running takes it when it next is
    # (a name or a README is the row's at once, whatever its status).
    scaled = any(v is not None for v in (min_containers, max_containers, scaledown_window))
    if scaled and updated.status in ("paused", "provisioning"):
        when = "when it is resumed" if updated.status == "paused" else "once it is up"
        err_console.print(f"{updated.name} is {updated.status}: it takes this scaling {when}.")
    print_json(_scaled(updated))


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
