"""``pictograph webhooks {create,list,event-types,get,update,delete,test,rotate-secret,deliveries,replay}``.

Register an https endpoint, then Pictograph POSTs signed events to it (e.g. when a
workflow run finishes)::

    pictograph webhooks create https://example.com/hook --event workflow_run.completed
    pictograph webhooks deliveries --endpoint <endpoint-id> --status failed
    pictograph webhooks replay <delivery-id>

``create`` prints the HMAC signing secret ONCE - store it to verify the
``X-Pictograph-Signature`` (``t=<ts>,v1=<hmac>``) header on each delivery.
Deliveries are durable and retried; inspect them with ``deliveries`` and
re-queue a failed one with ``replay``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import err_console, print_json, print_table

app = typer.Typer(no_args_is_help=True)


def _parse_auth_headers(items: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``--auth-header 'Name: Value'`` (or ``Name=Value``) options.

    Returns ``None`` when none were passed (leave unchanged on update), or a dict
    (which may be empty via ``--auth-header ''`` semantics is NOT supported here -
    clear headers with the SDK/app). The server validates names/values.
    """
    if not items:
        return None
    out: dict[str, str] = {}
    for raw in items:
        sep = ":" if ":" in raw else ("=" if "=" in raw else None)
        if sep is None:
            raise typer.BadParameter(f"--auth-header must be 'Name: Value', got {raw!r}")
        name, value = raw.split(sep, 1)
        out[name.strip()] = value.strip()
    return out


@app.command("create", help="Register a webhook endpoint (prints the signing secret once).")
def create_endpoint(
    url: Annotated[
        str, typer.Argument(help="Destination URL. Must be https and publicly routable.")
    ],
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Human-readable label.")
    ] = None,
    event_type: Annotated[
        list[str] | None,
        typer.Option(
            "--event",
            "-e",
            help="Event type to subscribe to (repeatable). Omit to receive all events.",
        ),
    ] = None,
    auth_header: Annotated[
        list[str] | None,
        typer.Option(
            "--auth-header",
            help="Custom request header sent on every delivery, 'Name: Value' "
            "(repeatable). Stored encrypted; only names are shown afterwards.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    created = client.webhooks.create(
        url,
        description=description,
        event_types=event_type,
        auth_headers=_parse_auth_headers(auth_header),
    )
    err_console.print(
        "[bold yellow]Save this secret now - it is shown once and never retrievable again.[/bold yellow]"
    )
    print_json(
        {
            "id": created.endpoint.id,
            "url": created.endpoint.url,
            "event_types": created.endpoint.event_types,
            "secret": created.secret,
        }
    )


@app.command("list", help="List webhook endpoints in your organization.")
def list_endpoints(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    endpoints = client.webhooks.list()
    if json_output:
        print_json([e.model_dump(mode="json", exclude_none=True) for e in endpoints])
        return
    rows = [
        {
            "id": e.id,
            "url": e.url,
            "enabled": e.enabled,
            "events": e.event_types or "all",
            "failures": e.consecutive_failures,
        }
        for e in endpoints
    ]
    print_table(rows, title=f"Webhook endpoints ({len(rows)})")


@app.command("event-types", help="List the event types you can subscribe an endpoint to.")
def list_event_types(
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    print_json({"event_types": client.webhooks.event_types()})


@app.command("get", help="Fetch a single webhook endpoint by its URL.")
def get_endpoint(
    endpoint: Annotated[
        str, typer.Argument(help="The endpoint's registered URL (a UUID also works).")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.webhooks.get(endpoint)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command(
    "update", help="Update a webhook endpoint's url / description / events / enabled state."
)
def update_endpoint(
    endpoint: Annotated[
        str, typer.Argument(help="The endpoint's registered URL (a UUID also works).")
    ],
    url: Annotated[str | None, typer.Option("--url", help="New https target URL.")] = None,
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
    event_type: Annotated[
        list[str] | None,
        typer.Option("--event", "-e", help="Replace the subscribed event types (repeatable)."),
    ] = None,
    enabled: Annotated[
        bool | None, typer.Option("--enabled/--disabled", help="Enable or disable delivery.")
    ] = None,
    auth_header: Annotated[
        list[str] | None,
        typer.Option(
            "--auth-header",
            help="Replace custom delivery headers, 'Name: Value' (repeatable).",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    updated = client.webhooks.update(
        endpoint,
        url=url,
        description=description,
        event_types=event_type,
        enabled=enabled,
        auth_headers=_parse_auth_headers(auth_header),
    )
    print_json(updated.model_dump(mode="json", exclude_none=True))


@app.command("delete", help="Delete a webhook endpoint and its delivery history.")
def delete_endpoint(
    endpoint: Annotated[
        str, typer.Argument(help="The endpoint's registered URL (a UUID also works).")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete webhook endpoint {endpoint!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.webhooks.delete(endpoint)
    print_json({"endpoint": endpoint, "deleted": True})


@app.command("test", help="Send a synthetic signed test event to an endpoint.")
def send_test_event(
    endpoint: Annotated[
        str, typer.Argument(help="The endpoint's registered URL (a UUID also works).")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.webhooks.test(endpoint)
    print_json(result)


@app.command(
    "rotate-secret",
    help="Mint a new signing secret (printed once; the old one stays valid during the grace window).",
)
def rotate_secret(
    endpoint: Annotated[
        str, typer.Argument(help="The endpoint's registered URL (a UUID also works).")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    rotated = client.webhooks.rotate_secret(endpoint)
    err_console.print(
        "[bold yellow]Save this secret now - it is shown once. The previous secret stays "
        "valid during the rotation grace window.[/bold yellow]"
    )
    print_json(
        {
            "id": rotated.endpoint.id,
            "secret_version": rotated.endpoint.secret_version,
            "secret": rotated.secret,
        }
    )


@app.command(
    "deliveries", help="List webhook deliveries, optionally filtered by endpoint / status."
)
def list_deliveries(
    endpoint: Annotated[
        str | None,
        typer.Option("--endpoint", help="Only show deliveries for this endpoint URL or UUID."),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="pending / delivered / failed / dead_letter."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    offset: Annotated[int, typer.Option("--offset")] = 0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    deliveries = client.webhooks.deliveries(
        endpoint=endpoint,
        status=status,  # type: ignore[arg-type]
        limit=limit,
        offset=offset,
    )
    if json_output:
        print_json([d.model_dump(mode="json", exclude_none=True) for d in deliveries])
        return
    rows = [
        {
            "id": d.id,
            "event": d.event_type,
            "status": d.status,
            "attempts": d.attempts,
            "code": d.last_status_code,
        }
        for d in deliveries
    ]
    print_table(rows, title=f"Webhook deliveries ({len(rows)})")


@app.command("replay", help="Re-queue a failed / dead-letter delivery with a fresh retry budget.")
def replay_delivery(
    delivery_id: Annotated[str, typer.Argument(help="Delivery UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    client.webhooks.replay(delivery_id)
    print_json({"id": delivery_id, "replayed": True})
