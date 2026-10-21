"""``pictograph deployments {list,get,create,predict,pause,resume,delete}``.

Stand a trained model up as a live inference endpoint, then call it::

    pictograph deployments create "Swift Falcon" --gpu t4     # prints the one-time token
    pictograph deployments create "Swift Falcon" --compute cpu  # the cheapest tier
    pictograph deployments predict "Swift Falcon" photo.jpg --token pk_deploy_...
    pictograph deployments predict "Swift Falcon" https://example.com/x.jpg --token pk_deploy_... --confidence 0.4

``predict`` looks up the deployment's ``endpoint_url`` with your org key, then calls
that endpoint directly (minimum latency) with the per-deployment bearer token.
"""

from __future__ import annotations

from typing import Annotated, Any, cast, get_args

import typer

from pictograph import DeploymentClient
from pictograph.cli import _inputs
from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.models.deployment import ComputeType, DeploymentGpuType

app = command_group(no_args_is_help=True)


@app.command("list", help="List model deployments in your organization.")
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
            "status": d.status,
            "gpu": d.gpu_type or d.compute_type,
            "endpoint_url": d.endpoint_url,
        }
        for d in deployments
    ]
    print_table(rows, title=f"Deployments ({len(rows)})")


@app.command("get", help="Fetch a single deployment by name.")
def get_deployment(
    deployment: Annotated[str, typer.Argument(help="Deployment name (a UUID also works).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.deployments.get(deployment)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command("compute-options", help="List selectable compute tiers + their per-minute rate.")
def compute_options(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    options = client.deployments.compute_options()
    if json_output:
        print_json([o.model_dump(mode="json") for o in options])
        return
    rows = [
        {
            "key": o.key,
            "label": o.label,
            "gpu": o.gpu_type or o.compute_type,
            "vram": f"{o.vram_gb} GB" if o.vram_gb else "",
            "rate_per_min_micro_usd": o.rate_per_min_micro_usd,
        }
        for o in options
    ]
    print_table(rows, title=f"Compute options ({len(rows)})")


_COMPUTE_HELP = "gpu (default) or cpu: the cheapest tier, for a light model."
_GPU_HELP = "t4 (default), l4 or a100. GPU deployments only."


def _compute_and_gpu(
    compute_type: str, gpu: str | None
) -> tuple[ComputeType, DeploymentGpuType | None]:
    """``--compute-type`` / ``--gpu`` checked together, before any request."""
    compute = compute_type.strip().lower()
    if compute == "cpu":
        if gpu:
            raise typer.BadParameter(
                "A CPU deployment has no GPU: drop --gpu, or use --compute-type gpu.",
                param_hint="--gpu",
            )
        return "cpu", None
    if compute != "gpu":
        raise typer.BadParameter(f"{compute_type!r}: use gpu or cpu.", param_hint="--compute-type")
    choice = (gpu or "t4").strip().lower()
    tiers = get_args(DeploymentGpuType)
    if choice not in tiers:
        raise typer.BadParameter(
            f"{gpu!r} is not a deployment GPU: use {', '.join(tiers)} "
            "(`pictograph deployments compute-options` lists them with prices).",
            param_hint="--gpu",
        )
    return "gpu", cast("DeploymentGpuType", choice)


@app.command("quote", help="Cost quote (micro-USD) for a deployment before creating it.")
def quote(
    compute_type: Annotated[
        str, typer.Option("--compute-type", "--compute", help=_COMPUTE_HELP)
    ] = "gpu",
    gpu_type: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    min_containers: Annotated[
        int, typer.Option("--min-containers", help="0 = scale-to-zero; >=1 = always warm.")
    ] = 0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    compute, gpu = _compute_and_gpu(compute_type, gpu_type)
    client = get_client(api_key)
    q = client.deployments.quote(compute_type=compute, gpu_type=gpu, min_containers=min_containers)
    print_json(q.model_dump(mode="json"))


@app.command("create", help="Deploy a trained model to a live endpoint (prints a one-time token).")
def create_deployment(
    model: Annotated[
        str, typer.Argument(help="Trained model name (status must be 'ready'). A UUID also works.")
    ],
    name: Annotated[str | None, typer.Option("--name")] = None,
    compute_type: Annotated[
        str, typer.Option("--compute-type", "--compute", help=_COMPUTE_HELP)
    ] = "gpu",
    gpu: Annotated[str | None, typer.Option("--gpu", help=_GPU_HELP)] = None,
    min_containers: Annotated[
        int, typer.Option("--min", help="Warm pool size (0 = scale to zero).")
    ] = 0,
    max_containers: Annotated[int, typer.Option("--max")] = 1,
    scaledown_window: Annotated[
        int, typer.Option("--scaledown", help="Idle seconds before scale-down.")
    ] = 60,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    compute, gpu_type = _compute_and_gpu(compute_type, gpu)
    client = get_client(api_key)
    created = client.deployments.create(
        model,
        name=name,
        compute_type=compute,
        gpu_type=gpu_type,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    err_console.print(
        "[bold yellow]Save this token now - the API won't show it again "
        "(a member can reveal it from the deployment's settings in the app).[/bold yellow]"
    )
    print_json(
        {
            "id": created.deployment.id,
            "name": created.deployment.name,
            "status": created.deployment.status,
            "auth_token": created.auth_token,
            "endpoint_url": created.deployment.endpoint_url,
        }
    )


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


@app.command("resume", help="Resume one or more paused deployments.")
def resume_deployment(
    deployments: Annotated[
        list[str], typer.Argument(help="One or more deployment names (UUIDs also work).")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if len(deployments) == 1:
        deployment = client.deployments.resume(deployments[0])
        print_json({"id": deployment.id, "status": deployment.status})
    else:
        print_json(client.deployments.bulk_resume(deployments).model_dump(mode="json"))


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
