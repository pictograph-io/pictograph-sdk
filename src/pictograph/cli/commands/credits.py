"""``pictograph credits {balance,history,usage,estimate}``."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("balance", help="Show current compute-credit balance (USD) + recent activity.")
def balance(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    bal = client.credits.balance()
    if json_output:
        print_json(bal.model_dump(mode="json", exclude_none=True))
        return
    typer.echo(f"Remaining:         ${bal.remaining_usd:,.2f}")
    typer.echo(f"Monthly allowance: ${bal.allowance_usd:,.2f}")
    if bal.budget_micro_usd:
        typer.echo(f"Overage budget:    ${bal.budget_usd:,.2f}")
        typer.echo(f"Overage spent:     ${bal.overage_usd:,.2f}")
    if bal.credits_reset_at:
        typer.echo(f"Resets:            {bal.credits_reset_at}")


@app.command("history", help="Page through the credit ledger (newest first).")
def history(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    offset: Annotated[int, typer.Option("--offset")] = 0,
    all_entries: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Auto-page the ENTIRE ledger (ignores --limit/--offset; honors --max-total).",
        ),
    ] = False,
    max_total: Annotated[
        int | None,
        typer.Option("--max-total", help="Cap on total entries fetched when using --all."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if all_entries:
        entries = list(client.credits.iter(max_total=max_total))
    else:
        entries = client.credits.history(limit=limit, offset=offset)
    if json_output:
        print_json([e.model_dump(mode="json", exclude_none=True) for e in entries])
        return
    rows = [
        {
            "when": str(e.created_at),
            "operation": e.operation,
            "amount (USD)": f"{e.amount / 1_000_000:+,.4f}",
            "balance (USD)": (
                f"${e.balance_after / 1_000_000:,.2f}" if e.balance_after is not None else None
            ),
        }
        for e in entries
    ]
    print_table(rows, title=f"Credit history ({len(rows)})")


@app.command("usage", help="Per-operation compute spend (USD) over a rolling window.")
def usage(
    range_: Annotated[
        str,
        typer.Option("--range", "-r", help="Window: day / week / month / all."),
    ] = "month",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    report = client.credits.usage_by_operation(range=range_)  # type: ignore[arg-type]
    if json_output:
        print_json(report.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "operation": op.operation,
            "spend_usd": f"${op.total_usd:,.4f}",
            "events": op.event_count,
        }
        for op in report.operations
    ]
    print_table(rows, title=f"Usage by operation ({report.range}) - total ${report.total_usd:,.2f}")


@app.command("estimate", help="Estimate the compute-credit cost (USD) of a planned operation.")
def estimate(
    operation: Annotated[
        str,
        typer.Argument(
            help=(
                "Operation slug (e.g., training_a10g, sam3_auto_annotation, "
                "inference_t4, image_generate_imagen_fast, image_edit_gemini_flash)."
            ),
        ),
    ],
    quantity: Annotated[int, typer.Option("--quantity", "-q")] = 1,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    est = client.credits.estimate(operation, quantity=quantity)
    if json_output:
        print_json(est.model_dump(mode="json", exclude_none=True))
        return
    typer.echo(f"Operation:  {est.operation}")
    typer.echo(f"Quantity:   {est.quantity} {est.unit}(s)")
    typer.echo(f"Per unit:   ${est.per_unit_usd:,.4f}")
    typer.echo(f"Total cost: ${est.total_usd:,.4f}")
    typer.echo(f"Remaining:  ${est.remaining_usd:,.2f}")
    typer.echo(f"Sufficient: {'yes' if est.sufficient else 'no'}")
