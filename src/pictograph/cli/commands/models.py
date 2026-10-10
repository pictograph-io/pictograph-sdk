"""``pictograph models {list,get,download,fork,delete}``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from pictograph._image_input import read_image
from pictograph.cli import _inputs
from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table
from pictograph.inference import ClassificationModel

app = typer.Typer(no_args_is_help=True)

# What `client.models.predict` sends when the caller names no top_k.
_HOSTED_TOP_K = 3


@app.command("list", help="List trained models in your organization.")
def list_models(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    models = client.models.list(limit=limit)
    if json_output:
        print_json([m.model_dump(mode="json", exclude_none=True) for m in models])
        return
    rows = [
        {
            "name": m.name,
            "id": m.id,
            "type": m.model_type,
            "arch": m.architecture,
            "status": m.status,
            "version": m.version,
        }
        for m in models
    ]
    print_table(rows, title=f"Models ({len(rows)})")


@app.command("get", help="Fetch a single trained model by name or UUID.")
def get_model(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.models.get_by_name(model)
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("update", help="Update a model's name / description / readme / license / visibility.")
def update_model(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    name: Annotated[str | None, typer.Option("--name", help="Rename the model to this.")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    readme: Annotated[str | None, typer.Option("--readme")] = None,
    visibility: Annotated[
        str | None, typer.Option("--visibility", help="'private' or 'public'.")
    ] = None,
    license_id: Annotated[str | None, typer.Option("--license-id")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if visibility is not None and visibility not in ("private", "public"):
        raise typer.BadParameter(
            "--visibility must be 'private' or 'public'", param_hint="--visibility"
        )
    client = get_client(api_key)
    model_id = client.models.get_by_name(model).id
    result = client.models.update(
        model_id=model_id,
        new_name=name,
        description=description,
        readme=readme,
        visibility=visibility,  # type: ignore[arg-type]
        license_id=license_id,
    )
    print_json(result.model_dump(mode="json", exclude_none=True))


# The weights containers `models download` can fetch. This is the DOWNLOAD route's
# own vocabulary (a raw file fetch names the FILE), which is deliberately not the
# loader's `format=` - see `Models.download` for the correspondence. It used to list
# only onnx/pytorch, so `--format safetensors` was refused client-side on a model
# that published one.
_DOWNLOAD_FORMATS = ("onnx", "pytorch", "safetensors", "pte", "engine")


@app.command("download", help="Download a trained model's weights file.")
def download_model(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Weights format: 'onnx' (default), 'pytorch' (the native .pth), "
                "'safetensors', 'pte' (ExecuTorch) or 'engine' (TensorRT). A format "
                "the model does not publish is refused, never substituted."
            ),
        ),
    ] = "onnx",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if fmt not in _DOWNLOAD_FORMATS:
        raise typer.BadParameter(
            f"--format must be one of {', '.join(_DOWNLOAD_FORMATS)}", param_hint="--format"
        )
    client = get_client(api_key)
    model_id = client.models.get_by_name(model).id
    output.parent.mkdir(parents=True, exist_ok=True)
    client.models.download(model_id=model_id, output_path=output, format=fmt)  # type: ignore[arg-type]
    print_json(
        {
            "model_id": model_id,
            "format": fmt,
            "output_path": str(output),
            "size_bytes": output.stat().st_size if output.is_file() else None,
        }
    )


@app.command("fork", help="Import (fork) a public model into your organization.")
def fork_model(
    organization: Annotated[
        str, typer.Argument(help="Organization slug that owns the source model.")
    ],
    model_name: Annotated[str, typer.Argument(help="Source public model slug or name.")],
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name for the copy in your organization (default: the source's name).",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    model = client.models.fork(organization, model_name, name=name)
    if json_output:
        print_json(model.model_dump(mode="json", exclude_none=True))
        return
    print_table(
        [
            {
                "name": model.name,
                "id": model.id,
                "type": model.model_type,
                "status": model.status,
                "forked_from": model.forked_from_model_id,
            }
        ],
        title="Imported model",
    )


@app.command(
    "delete",
    help="Delete one or more trained models (requires admin or owner role). "
    "Pass multiple UUIDs for a single server-side bulk delete.",
)
def delete_model(
    model_ids: Annotated[
        list[str], typer.Argument(help="A model name/UUID, or multiple UUIDs for a bulk delete.")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    n = len(model_ids)
    prompt = (
        f"Delete model {model_ids[0]!r}? This cannot be undone."
        if n == 1
        else f"Delete {n} models? This cannot be undone."
    )
    if not yes and not typer.confirm(prompt):
        raise typer.Abort()
    client = get_client(api_key)
    if n == 1:
        # Single arg may be a name or a UUID - resolve, then delete by id.
        model_id = client.models.get_by_name(model_ids[0]).id
        client.models.delete(model_id=model_id)
        print_json({"model_id": model_id, "deleted": True})
        return
    # Multiple ids → one atomic server-side bulk delete (no client fan-out).
    result = client.models.bulk_delete(model_ids)
    print_json(result.model_dump(mode="json"))


@app.command(
    "predict",
    help="Run a trained model on one or more images: locally (needs: pip install 'pictograph[inference]') or, with --remote, on Pictograph. Images can be paths, globs, URLs or - (stdin). `pictograph tools describe model/<name>` shows every way to run it.",
)
def predict(
    model: Annotated[str, typer.Argument(help="Model name.")],
    images: Annotated[list[str], typer.Argument(help=_inputs.HELP)],
    confidence: Annotated[
        float, typer.Option("--confidence", "-c", help="Minimum score to keep (0-1).")
    ] = 0.5,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            min=1,
            max=10,
            help="Classification models: how many classes to return, best first "
            "(default 3 with --remote, 1 locally).",
        ),
    ] = None,
    remote: Annotated[
        bool,
        typer.Option(
            "--remote",
            help="Run on Pictograph instead of locally: nothing to install, free, "
            "one hosted call per image.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl",
            help="One JSON object per image, one per line: {input, result} or "
            "{input, error}. Always used for more than one image.",
        ),
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    inputs = _inputs.expand_inputs(images)
    stdin = _inputs.read_stdin() if _inputs.STDIN in inputs else None
    client = get_client(api_key)
    # Resolve the model ONCE, before the first image: an unknown name stops here,
    # and a batch costs one lookup rather than one per image.
    local_model = None if remote else client.models.load(model, confidence=confidence)
    hosted_id = client.models.get_by_name(model).id if remote else None

    def run(value: str) -> Any:
        image = _inputs.load(value, stdin)
        if local_model is not None:
            if top_k is not None and isinstance(local_model, ClassificationModel):
                return local_model.predict(image, top_k=top_k)
            return local_model.predict(image)
        data, filename = read_image(image)
        return client.models._predict_resolved(
            str(hosted_id),
            data,
            filename,
            confidence=confidence,
            top_k=_HOSTED_TOP_K if top_k is None else top_k,
        )

    if jsonl or len(inputs) > 1:
        _inputs.emit_jsonl(
            inputs, lambda value: run(value).model_dump(mode="json", exclude_none=True)
        )
        return

    result = run(inputs[0])
    if json_output:
        print_json(result.model_dump(mode="json", exclude_none=True))
        return
    if remote:
        if result.tags:
            print_table(
                [{"class": t} for t in result.tags],
                title=f"{result.model_type} - top classes",
            )
        else:
            print_table(
                [
                    {
                        "name": a.get("name"),
                        "type": a.get("type"),
                        "confidence": round(float(a.get("confidence") or 0), 4),
                    }
                    for a in result.annotations
                ],
                title=f"{result.model_type} - {len(result.annotations)} prediction(s)",
            )
        return
    if result.model_type == "classification":
        rows = [{"class": c.name, "confidence": round(c.confidence, 4)} for c in result.classes]
        print_table(rows, title=f"{result.model_type} - top classes")
    else:
        rows = [
            {"name": p.name, "type": p.type, "confidence": round(p.confidence, 4)}
            for p in result.predictions
        ]
        print_table(rows, title=f"{result.model_type} - {len(rows)} prediction(s)")
