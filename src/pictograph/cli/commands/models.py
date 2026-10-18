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
# Printed when a --remote batch stops at a 429: hosted capacity is shared by every
# user and resets daily, while a local run has no platform limit at all.
_REMOTE_STOP_HINT = (
    " Hosted capacity is shared and resets daily; drop --remote to run the model on "
    "this machine (pip install 'pictograph[inference]'), which has no platform limit."
)


@app.command("list", help="List trained models in your organization.")
def list_models(
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
    models = list(client.models.iter(page_size=min(limit, 100), max_total=limit))
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
    help="Run a trained model on images, a video's frames (--video) or a dataset's "
    # `\\[`: Rich renders --help, and read `[inference]` as markup (it dropped it).
    "images (--dataset, read-only): locally (needs: pip install 'pictograph\\[inference]') "
    "or, with --remote, on Pictograph. Images can be paths, globs, URLs or - (stdin). "
    "`pictograph tools describe model/<name>` shows every way to run it.",
)
def predict(
    model: Annotated[str, typer.Argument(help="Model name.")],
    images: Annotated[list[str] | None, typer.Argument(help=_inputs.HELP)] = None,
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
            help="Run on Pictograph instead of locally: nothing to install, free, one "
            "hosted call per image or frame from a capacity every user shares each "
            "day - run locally for a video or a dataset of any size.",
        ),
    ] = False,
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
    json_output: Annotated[bool, typer.Option("--json")] = False,
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
    client = get_client(api_key)
    # A dataset's storage state and first page are checked here, before the model
    # is loaded: a typo in the dataset name should not cost a weights download.
    items = _inputs.open_items(source, client, frames="jpeg" if remote else "array")
    # Resolve the model ONCE, before the first image: an unknown name stops here,
    # and a batch costs one lookup rather than one per image.
    local_model = None if remote else client.models.load(model, confidence=confidence)
    hosted_id = client.models.get_by_name(model).id if remote else None

    def run(image: Any) -> Any:
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

    if jsonl or not source.single_image:
        _inputs.emit(
            items,
            lambda image: run(image).model_dump(mode="json", exclude_none=True),
            stop_hint=_REMOTE_STOP_HINT if remote else "",
        )
        return

    (item,) = list(items)
    result = run(item.load())
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
