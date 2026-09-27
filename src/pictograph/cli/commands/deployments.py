"""``pictograph deployments {list,get,create,predict,pause,resume,delete}``.

Stand a trained model up as a live inference endpoint, then call it::

    pictograph deployments create "Swift Falcon" --gpu t4     # prints the one-time token
    pictograph deployments predict "Swift Falcon" photo.jpg --token pk_deploy_...
    pictograph deployments predict "Swift Falcon" https://example.com/x.jpg --token pk_deploy_... --confidence 0.4

``predict`` looks up the deployment's ``endpoint_url`` with your org key, then calls
that endpoint directly (minimum latency) with the per-deployment bearer token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from pictograph import DeploymentClient
from pictograph.cli._client import get_client
from pictograph.cli._format import err_console, print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("list", help="List model deployments in your organization.")
def list_deployments(
    status: Annotated[
        str | None, typer.Option("--status", help="provisioning/active/paused/failed/terminated.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    deployments = client.deployments.list(status=status, limit=limit)  # type: ignore[arg-type]
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
            "rate_per_min_micro_usd": o.rate_per_min_micro_usd,
        }
        for o in options
    ]
    print_table(rows, title=f"Compute options ({len(rows)})")


@app.command("quote", help="Cost quote (micro-USD) for a deployment before creating it.")
def quote(
    compute_type: Annotated[str, typer.Option("--compute-type", help="cpu / gpu.")] = "gpu",
    gpu_type: Annotated[str | None, typer.Option("--gpu", help="t4 / l4 / a10g / a100.")] = None,
    min_containers: Annotated[
        int, typer.Option("--min-containers", help="0 = scale-to-zero; >=1 = always warm.")
    ] = 0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    q = client.deployments.quote(
        compute_type=compute_type,  # type: ignore[arg-type]
        gpu_type=gpu_type,  # type: ignore[arg-type]
        min_containers=min_containers,
    )
    print_json(q.model_dump(mode="json"))


@app.command("create", help="Deploy a trained model to a live endpoint (prints a one-time token).")
def create_deployment(
    model: Annotated[
        str, typer.Argument(help="Trained model name (status must be 'ready'). A UUID also works.")
    ],
    name: Annotated[str | None, typer.Option("--name")] = None,
    gpu: Annotated[str, typer.Option("--gpu", help="t4 / l4 / a10g / a100.")] = "t4",
    min_containers: Annotated[
        int, typer.Option("--min", help="Warm pool size (0 = scale to zero).")
    ] = 0,
    max_containers: Annotated[int, typer.Option("--max")] = 1,
    scaledown_window: Annotated[
        int, typer.Option("--scaledown", help="Idle seconds before scale-down.")
    ] = 60,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    created = client.deployments.create(
        model,
        name=name,
        gpu_type=gpu,  # type: ignore[arg-type]
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    err_console.print(
        "[bold yellow]Save this token now - it is shown once and never retrievable again.[/bold yellow]"
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


@app.command("predict", help="Run inference on an image against a deployment's endpoint.")
def predict(
    deployment: Annotated[
        str,
        typer.Argument(
            help="Deployment name (a UUID also works). Needs an account API key "
            "to resolve; pass --endpoint instead to call the URL directly with "
            "only --token."
        ),
    ],
    image: Annotated[str, typer.Argument(help="Local file path or http(s):// URL.")],
    token: Annotated[
        str,
        typer.Option(
            "--token", help="The per-deployment bearer token (pk_deploy_...) from create."
        ),
    ],
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
            "credential needed - no account API key, matching DeploymentClient.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    # A `pk_deploy_` token is meant to stand alone: it authorizes exactly one
    # endpoint and is what the cURL and Python snippets use by itself. Resolving
    # a name, though, is an ACCOUNT operation, so that form additionally needs
    # an account key. Given --endpoint there is nothing to resolve, so the
    # deployment token alone is sufficient here too.
    if endpoint:
        conn: Any = DeploymentClient(endpoint=endpoint, api_key=token)
    else:
        client = get_client(api_key)
        found = client.deployments.get(deployment)
        conn = client.deployments.connect(found, token)
    image_arg: str | Path = image if image.startswith(("http://", "https://")) else Path(image)
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] if class_filter else None
    # The endpoint's own JSON, not the typed result's dump: someone reaching for
    # the CLI on a deployment is inspecting what the endpoint actually returns.
    # The typed shape is what the Python API gives you (`DeploymentClient.infer`).
    result = conn.infer_raw(image_arg, confidence=confidence, class_filter=classes, top_k=top_k)
    print_json(result)


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
