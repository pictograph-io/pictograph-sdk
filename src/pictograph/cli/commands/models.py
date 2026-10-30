"""``pictograph models {list,get,update,download,fork,delete,predict,evaluate,evaluations,evaluation,versions,set-version,files,download-file}``."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pictograph._image_input import read_image
from pictograph.cli import _inputs
from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import err_console, print_error, print_json, print_table
from pictograph.exceptions import PollTimeoutError
from pictograph.inference import ClassificationModel

if TYPE_CHECKING:
    from pictograph.client import Client
    from pictograph.models.evaluation import ModelEvaluation
    from pictograph.models.model import ModelVersionEntry

app = command_group(no_args_is_help=True)

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
    result = client.models.get(model)
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("update", help="Change a model's name, description, README, visibility or license.")
def update_model(
    model: Annotated[str, typer.Argument(help="The model's name.")],
    name: Annotated[str | None, typer.Option("--name", help="Rename the model to this.")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    readme: Annotated[str | None, typer.Option("--readme")] = None,
    visibility: Annotated[
        str | None, typer.Option("--visibility", help="'public' (Explore) or 'private'. Admin+.")
    ] = None,
    license: Annotated[
        str | None,
        typer.Option("--license", help="A license id, e.g. mit ('custom' with --license-text)."),
    ] = None,
    license_text: Annotated[
        str | None, typer.Option("--license-text", help="A custom license's terms.")
    ] = None,
    hf_mirror: Annotated[
        bool | None,
        typer.Option(
            "--hf-mirror/--no-hf-mirror",
            help="Mirror it to Hugging Face while it is public (on by default). Admin+.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if visibility is not None and visibility not in ("private", "public"):
        raise typer.BadParameter(
            "--visibility must be 'private' or 'public'", param_hint="--visibility"
        )
    client = get_client(api_key)
    result = client.models.update(
        model,
        name=name,
        description=description,
        readme=readme,
        visibility=visibility,  # type: ignore[arg-type]
        license=license,
        license_custom_text=license_text,
        hf_publish_enabled=hf_mirror,
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
    model_id = client.models.get(model).id
    output.parent.mkdir(parents=True, exist_ok=True)
    client.models.download(model=model_id, output_path=output, format=fmt)  # type: ignore[arg-type]
    print_json(
        {
            "model_id": model_id,
            "format": fmt,
            "output_path": str(output),
            "size_bytes": output.stat().st_size if output.is_file() else None,
        }
    )


@app.command(
    "fork",
    help=(
        "Import (fork) a public model into your organization. Name it as ORG MODEL or "
        "as one org/model (the REF `pictograph explore models` prints)."
    ),
)
def fork_model(
    organization: Annotated[
        str,
        typer.Argument(
            help="Organization slug that owns the source model, or the whole org/model."
        ),
    ],
    model_name: Annotated[
        str | None,
        typer.Argument(help="Source public model slug or name (omit with org/model)."),
    ] = None,
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
    if model_name is None:
        org, sep, model_name = organization.partition("/")
        if not sep or not org.strip() or not model_name.strip():
            raise typer.BadParameter(
                "name the model as ORG MODEL or org/model, e.g. "
                "pictograph-research/pictograph-sceneparse150-m",
                param_hint="organization",
            )
        organization, model_name = org.strip(), model_name.strip()
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
        model_id = client.models.get(model_ids[0]).id
        client.models.delete(model=model_id)
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
    hosted_id = client.models.get(model).id if remote else None

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


# ─────────────────────────── evaluations ───────────────────────────
# `client.model_evaluations` from the CLI. An evaluation runs the model
# over an EXPORT on Pictograph and scores it against that export's labels; the
# labels are never changed. The three commands are the noun's three questions:
# start one (`evaluate`), which exist (`evaluations`), and how did this one do
# (`evaluation`, which also waits, writes the report and cancels).

_EVAL_POLL_INTERVAL = 3.0
_EVAL_TIMEOUT = 1800.0
# The developer API serves the JSON report only (`GET .../{id}/export`); the HTML
# report is built in the app's evaluation view. A report path that asks for HTML
# or PDF is refused up front rather than filled with JSON under that name.
_NOT_JSON_REPORT = (".html", ".htm", ".pdf")

_WaitOption = Annotated[
    bool,
    typer.Option("--wait", help="Block until the evaluation finishes, then print its metrics."),
]
_TimeoutOption = Annotated[
    float,
    typer.Option(
        "--timeout",
        min=1.0,
        help="With --wait: seconds to wait. The evaluation keeps running after it.",
    ),
]
_PollOption = Annotated[
    float,
    typer.Option("--poll-interval", min=0.5, help="With --wait: seconds between status checks."),
]


def _check_thresholds(iou: float, confidence: float) -> None:
    # The API's own bounds for an evaluation, checked here so a typo costs no
    # request: 0 < IoU <= 1, 0 <= confidence <= 1.
    if not 0.0 < iou <= 1.0:
        raise typer.BadParameter("must be above 0 and at most 1", param_hint="--iou")
    if not 0.0 <= confidence <= 1.0:
        raise typer.BadParameter("must be between 0 and 1", param_hint="--confidence")


def _headline(ev: ModelEvaluation) -> tuple[str, float | None]:
    """The metric an evaluation of this model family is read by.

    The server fills exactly one family's headline (ModelEvaluation's own rule):
    mIoU for semantic segmentation, accuracy for classification, mAP at the
    evaluation's IoU threshold for detection and instance segmentation.
    """
    m = ev.overall_metrics
    if m is None:
        return ("", None)
    if m.miou is not None:
        return ("mIoU", m.miou)
    if m.accuracy is not None:
        return ("accuracy", m.accuracy)
    return (f"mAP@{ev.iou_threshold:g}", m.map)


def _num(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _images(ev: ModelEvaluation) -> str:
    done = f"{ev.evaluated_images}/{ev.total_images}"
    return f"{done} ({ev.failed_images} failed)" if ev.failed_images else done


def _print_evaluation(ev: ModelEvaluation) -> None:
    label, value = _headline(ev)
    m = ev.overall_metrics
    # One evaluation reads as field / value rows: a dozen columns side by side do not
    # fit a terminal, and Rich would cut the metric names first.
    fields: list[tuple[str, Any]] = [
        ("id", ev.id),
        ("status", ev.status),
        ("progress", f"{ev.progress}%"),
        ("images", _images(ev)),
        ("IoU threshold", ev.iou_threshold),
        ("confidence threshold", ev.confidence_threshold),
    ]
    if m is not None:
        fields += [
            (label, _num(value)),
            ("precision", _num(m.precision)),
            ("recall", _num(m.recall)),
            ("F1", _num(m.f1)),
        ]
    print_table([{"field": k, "value": v} for k, v in fields], title="Model evaluation")
    if ev.error_message:
        print_error(ev.error_message)
    if not ev.per_class_metrics:
        return
    semantic = m is not None and m.miou is not None
    detection = label.startswith("mAP")
    rows: list[dict[str, Any]] = []
    for c in ev.per_class_metrics:
        r: dict[str, Any] = {"class": c.class_name, "support": c.support}
        if semantic:
            r["iou"] = _num(c.iou)
            r["dice"] = _num(c.dice)
        r["precision"] = _num(c.precision)
        r["recall"] = _num(c.recall)
        r["f1"] = _num(c.f1)
        if detection:
            r["ap"] = _num(c.ap)
        rows.append(r)
    print_table(rows, title=f"Per class ({len(rows)})")


def _emit_evaluation(ev: ModelEvaluation, json_output: bool) -> None:
    if json_output:
        print_json(ev.model_dump(mode="json", exclude_none=True))
    else:
        _print_evaluation(ev)


def _wait_for_evaluation(
    client: Client, evaluation_id: str, *, timeout: float, poll_interval: float
) -> ModelEvaluation:
    err_console.print(
        f"[dim]Waiting for evaluation {evaluation_id} (up to {timeout:.0f}s)...[/dim]",
        soft_wrap=True,
    )
    try:
        return client.model_evaluations.wait_for_completion(
            evaluation_id, poll_interval=poll_interval, timeout=timeout
        )
    except PollTimeoutError:
        # The SDK's message names the Python call; on the CLI the next step is a command.
        raise CliError(
            f"Evaluation {evaluation_id} did not finish within {timeout:.0f}s. It keeps "
            f"running on Pictograph: wait again with "
            f"`pictograph models evaluation {evaluation_id} --wait`.",
            exit_code=ExitCode.STILL_RUNNING,
            code="wait_timeout",
        ) from None


@app.command(
    "evaluate",
    help="Score a model against one of a dataset's exports, on Pictograph: precision, "
    "recall, F1 and mAP per class (accuracy for a classifier, mIoU for semantic "
    "segmentation). The export's images and classes are the eval set; your labels are "
    "never changed. Billed as batch inference. Prints the new evaluation's id at once; "
    "--wait blocks until it finishes and prints its metrics.",
)
def evaluate(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    dataset: Annotated[str, typer.Option("--dataset", help="The dataset the export belongs to.")],
    export: Annotated[
        str,
        typer.Option(
            "--export",
            help="A completed export of that dataset. Pick one the model did not train on.",
        ),
    ],
    iou: Annotated[
        float,
        typer.Option(
            "--iou",
            help="Minimum IoU for a prediction to match a labelled object (above 0, at most 1).",
        ),
    ] = 0.5,
    confidence: Annotated[
        float,
        typer.Option("--confidence", "-c", help="Minimum score to keep a prediction (0-1)."),
    ] = 0.5,
    wait: _WaitOption = False,
    timeout: _TimeoutOption = _EVAL_TIMEOUT,
    poll_interval: _PollOption = _EVAL_POLL_INTERVAL,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    _check_thresholds(iou, confidence)
    client = get_client(api_key)
    created = client.model_evaluations.create(
        model, dataset=dataset, export=export, iou_threshold=iou, confidence_threshold=confidence
    )
    if not wait:
        if json_output:
            print_json(created.model_dump(mode="json", exclude_none=True))
        else:
            print_table(
                [{"id": created.id, "status": created.status, "images": _images(created)}],
                title="Evaluation started",
            )
            err_console.print(
                f"[dim]Check on it with: pictograph models evaluation {created.id} --wait[/dim]",
                soft_wrap=True,
            )
        return
    ev = _wait_for_evaluation(client, created.id, timeout=timeout, poll_interval=poll_interval)
    _emit_evaluation(ev, json_output)


@app.command(
    "evaluations",
    help="List your organization's model evaluations, newest first. Name a model, "
    "--dataset, or both to narrow the list.",
)
def list_evaluations(
    model: Annotated[
        str | None, typer.Argument(help="Only this model's evaluations (name or UUID).")
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Only evaluations against this dataset's exports."),
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
    pager = client.model_evaluations.iter(model=model, dataset=dataset, page_size=min(limit, 100))
    evaluations = list(itertools.islice(pager, limit))
    if json_output:
        print_json([e.model_dump(mode="json", exclude_none=True) for e in evaluations])
        return
    rows = []
    for e in evaluations:
        label, value = _headline(e)
        rows.append(
            {
                "id": e.id,
                "status": e.status,
                "model_id": e.model_id,
                "score": f"{label} {_num(value)}" if value is not None else None,
                "images": _images(e),
                "created": e.created_at.isoformat(timespec="seconds") if e.created_at else None,
            }
        )
    print_table(rows, title=f"Model evaluations ({len(rows)})")


@app.command(
    "evaluation",
    help="Show one model evaluation: its status and metrics, overall and per class. "
    "--wait blocks until it finishes; --report writes its full report as JSON; --cancel "
    "stops a pending or running one.",
)
def show_evaluation(
    evaluation_id: Annotated[
        str,
        typer.Argument(help="Evaluation id, from `models evaluate` or `models evaluations`."),
    ],
    wait: _WaitOption = False,
    timeout: _TimeoutOption = _EVAL_TIMEOUT,
    poll_interval: _PollOption = _EVAL_POLL_INTERVAL,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Write the finished evaluation's full report to this JSON file: every "
            "metric, the PR curves, the confusion matrix and the best and worst images "
            "(the app's Download JSON). The HTML report is made in the app.",
        ),
    ] = None,
    cancel: Annotated[
        bool, typer.Option("--cancel", help="Cancel a pending or running evaluation.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if cancel and (wait or report is not None):
        raise typer.BadParameter(
            "cannot be combined with --wait or --report", param_hint="--cancel"
        )
    if report is not None and report.suffix.lower() in _NOT_JSON_REPORT:
        raise typer.BadParameter(
            "the CLI writes the JSON report (name the file .json); the HTML report is "
            "made in the app, from the evaluation's Report button",
            param_hint="--report",
        )
    client = get_client(api_key)
    if cancel:
        _emit_evaluation(client.model_evaluations.cancel(evaluation_id), json_output)
        return
    if wait:
        ev = _wait_for_evaluation(
            client, evaluation_id, timeout=timeout, poll_interval=poll_interval
        )
    else:
        ev = client.model_evaluations.get(evaluation_id)
    if report is None:
        _emit_evaluation(ev, json_output)
        return
    if ev.status != "completed":
        # The API answers 409 here; say what to do next instead.
        running = ev.status in ("pending", "running")
        why = (
            f"is still {ev.status} ({ev.progress}%). Add --wait to wait for it, then write the report."
            if running
            else f"ended {ev.status}, so it has no report."
        )
        raise CliError(
            f"Evaluation {ev.id} {why}",
            exit_code=ExitCode.CONFLICT,
            code="evaluation_not_finished" if running else "evaluation_has_no_report",
            details={"id": ev.id, "status": ev.status},
        )
    data = client.model_evaluations.report(ev.id)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(data, indent=2) + "\n")
    print_json(
        {
            "evaluation_id": ev.id,
            "report_path": str(report),
            "size_bytes": report.stat().st_size,
            "schema_version": data.get("schema_version"),
        }
    )


# ───────────── versions, the pin, and the file manifest ─────────────


def _version_label(v: ModelVersionEntry) -> str:
    if v.version_label:
        return v.version_label
    return f"v{v.version_number}" if v.version_number is not None else v.version_id


def _pick_version(versions: list[ModelVersionEntry], wanted: str, model: str) -> ModelVersionEntry:
    """The version ``wanted`` names: its id, its label (``2.0.0``) or its number (``2``)."""
    key = wanted.strip()
    number = key[1:] if key[:1] in ("v", "V") and key[1:].isdigit() else key
    for v in versions:
        if key in (v.version_id, v.version_label) or (
            number.isdigit() and v.version_number == int(number)
        ):
            return v
    known = ", ".join(_version_label(v) for v in versions) or "none"
    raise CliError(
        f"Model {model!r} has no version {wanted!r} (it has: {known}).",
        exit_code=ExitCode.NOT_FOUND,
        code="version_not_found",
        fix=f"List them with `pictograph models versions {_quoted_arg(model)}`.",
    )


def _quoted_arg(value: str) -> str:
    return f"'{value}'" if any(c.isspace() for c in value) else value


def _size(n: int | None) -> str | None:
    if n is None:
        return None
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return None


@app.command(
    "versions",
    help="List a model's versions: label, status, precision, architecture, its headline "
    "metric, and which one is current (what downloads, deployments and auto-annotate use).",
)
def model_versions(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print the payload as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    payload = client.models.versions(model)
    if json_output:
        print_json(payload.model_dump(mode="json", exclude_none=True))
        return
    rows = []
    for v in payload.versions:
        label, value = _version_metric(v.metrics)
        rows.append(
            {
                "version": _version_label(v),
                "status": v.status,
                "precision": v.precision,
                "architecture": v.architecture,
                "metric": f"{label} {_num(value)}" if value is not None else None,
                "current": v.is_current,
                "latest": v.is_latest,
                "created": v.created_at.date().isoformat() if v.created_at else None,
                "id": v.version_id,
            }
        )
    pin = "pinned" if payload.pinned_version_id else "follows the newest ready version"
    print_table(rows, title=f"Versions of {model} ({len(rows)}; {pin})")


def _version_metric(metrics: dict[str, Any]) -> tuple[str, float | None]:
    """The headline metric a version's training recorded (``headline_metric`` /
    ``headline_value``, which training writes for every pipeline), else the first of
    the usual keys an older run carries."""
    name, value = metrics.get("headline_metric"), metrics.get("headline_value")
    if isinstance(name, str) and isinstance(value, int | float):
        return name, float(value)
    for key in ("mAP", "mAP50", "miou", "accuracy", "oks", "f1"):
        found = metrics.get(key)
        if isinstance(found, int | float):
            return key, float(found)
    return "", None


@app.command(
    "set-version",
    help="Pin the version a model serves everywhere (downloads, deployments, auto-annotate): "
    "promote a new one or roll back. The pin survives later trainings. --clear follows the "
    "newest ready version again. Admin role.",
)
def set_model_version(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    version: Annotated[
        str | None,
        typer.Argument(help="The version: its label (2.0.0), its number (2 or v2) or its id."),
    ] = None,
    clear: Annotated[
        bool, typer.Option("--clear", help="Remove the pin: follow the newest ready version.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if (version is None) == (not clear):
        raise typer.BadParameter("give a VERSION, or --clear (not both).", param_hint="VERSION")
    client = get_client(api_key)
    version_id: str | None = None
    if version is not None:
        version_id = _pick_version(
            client.models.versions(model).versions, version, model
        ).version_id
    payload = client.models.set_current_version(model, version=version_id)
    current = next((v for v in payload.versions if v.is_current), None)
    print_json(
        {
            "model": model,
            "current_version": _version_label(current) if current else None,
            "current_version_id": payload.current_version_id,
            "pinned": payload.pinned_version_id is not None,
        }
    )


@app.command(
    "files",
    help="List a model version's downloadable files: weights (ONNX, PyTorch, safetensors, "
    "ExecuTorch, TensorRT; fp32 / fp16), config.json, README.md and LICENSE.md.",
)
def model_files(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    version: Annotated[
        str | None,
        typer.Option(
            "--version", help="Which version (label, number or id); the current one by default."
        ),
    ] = None,
    all_versions: Annotated[
        bool, typer.Option("--all-versions", help="Every version's files.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the manifest as JSON.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if version is not None and all_versions:
        raise typer.BadParameter(
            "give --version or --all-versions, not both.", param_hint="--version"
        )
    client = get_client(api_key)
    manifest = client.models.files(model)
    labels = {v.version_id: _version_label(v) for v in manifest.versions}
    if all_versions:
        files = manifest.files
    else:
        chosen: ModelVersionEntry | None
        if version is not None:
            chosen = _pick_version(manifest.versions, version, model)
        else:
            chosen = next(
                (v for v in manifest.versions if v.is_current),
                manifest.versions[0] if manifest.versions else None,
            )
        files = [
            f for f in manifest.files if chosen is not None and f.version_id == chosen.version_id
        ]
    if json_output:
        print_json([f.model_dump(mode="json", exclude_none=True) for f in files])
        return
    rows = [
        {
            "name": f.name,
            "version": labels.get(f.version_id, f.version_id),
            "format": f.format,
            "runtime": f.runtime,
            "precision": f.precision,
            "target": f.target_key,
            "size": _size(f.size_bytes),
            "stale": f.stale,
        }
        for f in files
    ]
    print_table(rows, title=f"Files of {model} ({len(rows)})")
    if any(f.stale for f in files):
        err_console.print(
            "[dim]A stale TensorRT engine will not load on the current runtime; a stale "
            ".pte still loads.[/dim]",
            soft_wrap=True,
        )


@app.command(
    "download-file",
    help="Download one file of a model's manifest by its name (see pictograph models "
    "files): weights of any format, config.json, README.md or LICENSE.md.",
)
def download_model_file(
    model: Annotated[str, typer.Argument(help="Model name (org-unique) or UUID.")],
    filename: Annotated[str, typer.Argument(help="The file's name, as models files lists it.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write it.")],
    version: Annotated[
        str | None,
        typer.Option("--version", help="Which version (label, number or id); the current one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    version_ref: str | None = None
    if version is not None:
        version_ref = _pick_version(
            client.models.versions(model).versions, version, model
        ).version_id
    output.parent.mkdir(parents=True, exist_ok=True)
    path = client.models.download_file(
        model, filename=filename, version=version_ref, output_path=output
    )
    print_json(
        {
            "model": model,
            "file": filename,
            "output_path": str(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    )
