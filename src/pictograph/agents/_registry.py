"""Tool registry - single source of truth for agent tool definitions.

Every adapter (Claude, OpenAI, dynamic-discovery via tools.json) reads
from this registry. Each entry pairs a Pydantic args schema with a
handler that takes ``(client, **validated_args)`` and returns a
JSON-serialisable result.

Descriptions follow Anthropic's "use when X" pattern - agents read
descriptions to decide *when* to call a tool, so the first sentence
must describe the trigger condition. Param names mirror the SDK
(``dataset_name``, ``image_uuid``) so an agent that learned the SDK
docstrings can use the tools without re-learning the surface.

Why a registry (vs. one decorator per resource method): keeping the
list explicit lets us (1) curate which methods are agent-safe, (2)
attach role + cost metadata for guardrails, (3) emit one
canonical JSON Schema for dynamic-discovery agents.

Tool cost is carried as ``cost_micro_usd`` - an estimate of the compute
credit a tool consumes, in **micro-USD (µUSD)** (``1 USD = 1_000_000 µUSD``).
Read-only / metadata tools are ``0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from pictograph.agents import _args
from pictograph.augment import build_ops

if TYPE_CHECKING:
    from collections.abc import Callable

    from pictograph import Client


RequiredRole = Literal["viewer", "member", "admin", "owner"]
"""Minimum role required to invoke a tool. Backend re-enforces server-side."""


@dataclass(frozen=True)
class ToolDescriptor:
    """A single agent-callable tool.

    Attributes:
        name: Snake-case identifier exposed to agents (e.g. ``upload_dataset_from_directory``).
        description: Agent-facing description. Lead with "Use when X" - agents
            read this to choose between tools.
        args_schema: Pydantic v2 model class. Generates the JSON Schema agents see;
            handler receives an instance of this model.
        handler: ``(client, args_model) -> Any``. Returns a JSON-serialisable result
            (dict, list, primitive). Pydantic models are dumped via ``model_dump``.
        idempotent: When True, agents may safely retry on transient failures.
            False for create/delete/upload (avoid double-charge).
        required_role: Minimum org-member role.
        cost_micro_usd: Approximate compute-credit cost in micro-USD (µUSD;
            ``1 USD = 1_000_000 µUSD``); ``0`` for read-only / free ops. Agents
            may gate based on this + ``client.credits.balance()``.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[[Client, BaseModel], Any]
    idempotent: bool
    required_role: RequiredRole
    cost_micro_usd: int


# ═════════════════════════════════════════════════════════════════════
# Handlers
#
# Each handler accepts (client, args) where args is the Pydantic model
# instance. Returns a JSON-serialisable result. Pydantic models from
# the SDK are dumped via model_dump(mode="json"); workflow report
# dataclasses are converted via __dict__ + manual dataclass walking
# (asdict isn't safe with Pydantic models inside).
# ═════════════════════════════════════════════════════════════════════


def _dump(obj: Any) -> Any:
    """Recursively coerce Pydantic models / dataclasses to JSON-friendly dicts."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, list | tuple):
        return [_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dump(getattr(obj, f)) for f in obj.__dataclass_fields__}
    return obj


# ───────────── multi-resource operations ─────────────
#
# Tool NAMES are the agent-facing vocabulary and are deliberately flat - they do
# not track the Python call path (``upload_dataset_from_directory`` reaches
# ``client.images.upload_from_directory``). Renaming them would force every agent
# prompt and saved tool-call transcript to be rewritten for no gain.


def _h_upload_dataset_from_directory(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.UploadDatasetFromDirectoryArgs)
    report = client.images.upload_from_directory(
        args.dataset_name,
        args.directory,
        organize_by_class=args.organize_by_class,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
    )
    return _dump(report)


def _h_auto_annotate_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.AutoAnnotateDatasetArgs)
    report = client.auto_annotate.dataset(
        args.dataset_name,
        args.classes,
        mode=args.mode,
        confidence_threshold=args.confidence_threshold,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )
    return _dump(report)


def _h_augment_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.AugmentDatasetArgs)
    report = client.images.augment(
        args.dataset_name,
        build_ops(args.ops),
        multiplier=args.multiplier,
        into=args.into,
        include_original=args.include_original,
        seed=args.seed,
        max_source_images=args.max_source_images,
    )
    return _dump(report)


def _h_tile_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.TileDatasetArgs)
    report = client.images.tile(
        args.dataset_name,
        rows=args.rows,
        cols=args.cols,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        include_empty=args.include_empty,
        into=args.into,
        max_source_images=args.max_source_images,
    )
    return _dump(report)


def _h_train_pipeline(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.TrainPipelineArgs)
    # EXPORT-driven. An agent must name the export it trains on, so it cannot
    # silently mint one (and cannot mint an empty one).
    run = client.training.create(
        dataset_name=args.dataset_name,
        export_name=args.export_name,
        pipeline_type=args.pipeline,
        gpu_type=args.gpu,
        name=args.name or f"{args.pipeline}-run",
        config=args.config,
        wait=args.wait,
    )
    model = client.models.get(model_id=run.model_id) if run.model_id else None
    return {"training_run": _dump(run), "model": _dump(model)}


# ───────────── datasets ─────────────


def _h_list_datasets(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ListDatasetsArgs)
    datasets = client.datasets.list(limit=args.limit)
    return {"datasets": _dump(datasets), "count": len(datasets)}


def _h_get_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.GetDatasetArgs)
    dataset = client.datasets.get(
        args.name,
        include_images=args.include_images,
        images_limit=args.images_limit,
    )
    return _dump(dataset)


def _h_create_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.CreateDatasetArgs)
    dataset = client.datasets.create(args.name, description=args.description)
    return _dump(dataset)


def _h_delete_dataset(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.DeleteDatasetArgs)
    client.datasets.delete(args.name)
    return {"deleted": True, "name": args.name}


# ───────────── images ─────────────


def _h_upload_image(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.UploadImageArgs)
    dataset = client.datasets.get(args.dataset_name)
    image = client.images.upload(
        # already resolved above - a uuid short-circuits _resolve.
        dataset_name=dataset.id,
        file_path=Path(args.file_path),
        directory_path=args.directory_path,
    )
    return _dump(image)


def _h_delete_image(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.DeleteImageArgs)
    client.images.delete(args.dataset_name, args.image)
    return {"deleted": True, "dataset_name": args.dataset_name, "image": args.image}


def _h_review_image(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ReviewImageArgs)
    status = client.images.review(args.dataset_name, args.image, args.action, note=args.note)
    return {"image": args.image, "action": args.action, "status": status}


def _h_set_image_split(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.SetImageSplitArgs)
    split = client.images.set_split(args.dataset_name, args.image, args.split)
    return {"image": args.image, "split": split}


def _h_rebalance_splits(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.RebalanceSplitsArgs)
    return client.images.assign_splits(
        args.dataset_id, train=args.train, val=args.val, test=args.test, seed=args.seed
    )


# ───────────── annotations ─────────────


def _h_get_annotations(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.GetAnnotationsArgs)
    anns = client.annotations.get(args.dataset_name, args.image, directory_path=args.directory_path)
    return {"annotations": _dump(anns), "count": len(anns)}


def _h_save_annotations(client: Client, args: BaseModel) -> Any:
    from pydantic import TypeAdapter

    from pictograph.models.annotation import Annotation

    assert isinstance(args, _args.SaveAnnotationsArgs)
    adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    parsed = adapter.validate_python(args.annotations)
    result = client.annotations.save(
        args.dataset_name, args.image, parsed, directory_path=args.directory_path
    )
    return _dump(result)


# ───────────── auto-annotate (single prompt) ─────────────


def _h_auto_annotate_point(client: Client, args: BaseModel) -> Any:
    if not isinstance(args, _args.AutoAnnotatePointArgs):
        raise TypeError(f"Expected AutoAnnotatePointArgs, got {type(args).__name__}")
    pos = [(p[0], p[1]) for p in args.positive_points] if args.positive_points else None
    neg = [(p[0], p[1]) for p in args.negative_points] if args.negative_points else None
    result = client.auto_annotate.point(
        dataset_name=args.dataset_name,
        image_filename=args.image_filename,
        x=args.x,
        y=args.y,
        name=args.name,
        positive_points=pos,
        negative_points=neg,
        score_threshold=args.score_threshold,
    )
    return _dump(result)


def _h_auto_annotate_box(client: Client, args: BaseModel) -> Any:
    if not isinstance(args, _args.AutoAnnotateBoxArgs):
        raise TypeError(f"Expected AutoAnnotateBoxArgs, got {type(args).__name__}")
    result = client.auto_annotate.box(
        dataset_name=args.dataset_name,
        image_filename=args.image_filename,
        box=args.box,
        name=args.name,
        confidence_threshold=args.confidence_threshold,
        return_polygon=args.return_polygon,
        negative_boxes=args.negative_boxes,
    )
    return _dump(result)


def _h_auto_annotate_text(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.AutoAnnotateTextArgs)
    result = client.auto_annotate.text(
        dataset_name=args.dataset_name,
        image_filename=args.image_filename,
        text_prompt=args.text_prompt,
        output_type=args.output_type,
        confidence_threshold=args.confidence_threshold,
    )
    return _dump(result)


# ───────────── search ─────────────


def _h_search_by_tag(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.SearchByTagArgs)
    results = client.search.by_tag(
        dataset_name=args.dataset_name,
        objects=args.objects,
        scenes=args.scenes,
        attributes=args.attributes,
        limit=args.limit,
    )
    return {"images": _dump(results), "count": len(results)}


def _h_search_by_similarity(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.SearchBySimilarityArgs)
    results = client.search.by_similarity(
        args.dataset_name,
        args.image,
        threshold=args.threshold,
        limit=args.limit,
        directory_path=args.directory_path,
    )
    return {"images": _dump(results), "count": len(results)}


# ───────────── exports ─────────────


def _h_create_export(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.CreateExportArgs)
    export = client.exports.create(
        args.dataset_name,
        args.name,
        format=args.format,
        include_images=args.include_images,
        class_filter=args.class_filter,
        status_filter=args.status_filter,
        organize_by_split=args.organize_by_split,
        wait=True,
    )
    return _dump(export)


def _h_list_exports(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ListExportsArgs)
    exports = client.exports.list(limit=args.limit)
    return {"exports": _dump(exports), "count": len(exports)}


def _h_download_export(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.DownloadExportArgs)
    path = client.exports.download(
        args.dataset_name, args.export_name, output_path=Path(args.output_path)
    )
    return {"output_path": str(path)}


# ───────────── training ─────────────


def _h_get_training_status(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.GetTrainingStatusArgs)
    run = client.training.get(run_id=args.run_id)
    return _dump(run)


def _h_cancel_training(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.CancelTrainingArgs)
    run = client.training.cancel(run_id=args.run_id)
    return _dump(run)


# ───────────── models ─────────────


def _h_list_models(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ListModelsArgs)
    models = client.models.list(limit=args.limit)
    return {"models": _dump(models), "count": len(models)}


def _h_download_model(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.DownloadModelArgs)
    path = client.models.download(
        model_id=args.model_id,
        output_path=Path(args.output_path),
        format=args.format,
    )
    return {"output_path": str(path), "format": args.format}


# ───────────── deployments ─────────────


def _h_list_deployments(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ListDeploymentsArgs)
    deployments = client.deployments.list(limit=args.limit)
    return {"deployments": _dump(deployments), "count": len(deployments)}


def _h_get_deployment(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.GetDeploymentArgs)
    return _dump(client.deployments.get(args.deployment))


def _h_create_deployment(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.CreateDeploymentArgs)
    created = client.deployments.create(
        args.model,
        name=args.name,
        compute_type=args.compute_type,
        gpu_type=args.gpu_type,
        min_containers=args.min_containers,
        max_containers=args.max_containers,
        scaledown_window=args.scaledown_window,
    )
    return _dump(created)


def _h_delete_deployment(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.DeleteDeploymentArgs)
    client.deployments.delete(args.deployment)
    return {"deleted": True, "deployment": args.deployment}


# ───────────── credits ─────────────


def _h_get_credit_balance(
    client: Client,
    args: BaseModel,  # noqa: ARG001
) -> Any:
    balance = client.credits.balance()
    return _dump(balance)


def _h_estimate_credit_cost(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.EstimateCreditCostArgs)
    estimate = client.credits.estimate(args.operation, quantity=args.quantity)
    return _dump(estimate)


def _h_list_notifications(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ListNotificationsArgs)
    items = client.notifications.list(unread_only=args.unread_only, limit=args.limit)
    return {
        "notifications": _dump(items),
        "count": len(items),
        "unread_count": client.notifications.unread_count(),
    }


# ───────────── connectors ─────────────


def _h_validate_connector(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ValidateConnectorArgs)
    result = client.connectors.validate(
        provider=args.provider,
        api_key=args.api_key,
    )
    return _dump(result)


def _h_import_from_connector(client: Client, args: BaseModel) -> Any:
    assert isinstance(args, _args.ImportFromConnectorArgs)
    job = client.connectors.import_(
        args.provider,
        args.api_key,
        args.datasets,
    )
    return _dump(job)


# ═════════════════════════════════════════════════════════════════════
# REGISTRY - the canonical, ordered list of every agent-callable tool
# ═════════════════════════════════════════════════════════════════════


REGISTRY: list[ToolDescriptor] = [
    # ─── workflows (highest leverage, agent-friendly orchestration) ───
    ToolDescriptor(
        name="upload_dataset_from_directory",
        description=(
            "Use when the user asks to upload a directory of images to a dataset. "
            "Walks the directory recursively, ensures the dataset exists, and uploads "
            "every supported image (jpg/png/webp/etc.). Subdirectories become virtual "
            "directories by default. Returns a per-file failure report - partial success "
            "does not raise."
        ),
        args_schema=_args.UploadDatasetFromDirectoryArgs,
        handler=_h_upload_dataset_from_directory,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="auto_annotate_dataset",
        description=(
            "Use when the user wants to auto-annotate a dataset with one or more "
            "classes via SAM3. Default 'batch' mode runs an async job over many "
            "images (best for >10 images); 'text' mode runs synchronous per-image "
            "prompts (better for small datasets). Skips already-annotated images "
            "unless overwrite=True."
        ),
        args_schema=_args.AutoAnnotateDatasetArgs,
        handler=_h_auto_annotate_dataset,
        idempotent=False,
        required_role="member",
        cost_micro_usd=25_000,  # batch SAM3 over many images (~0.025 USD est.)
    ),
    ToolDescriptor(
        name="augment_dataset",
        description=(
            "Use when the user wants to expand/augment a dataset (a 'generate a "
            "version' request). For every source image it produces N augmented "
            "variants - flip/rotate/crop/brightness/etc. - with the annotation "
            "geometry remapped, and uploads them. Writes into a new dataset (the "
            "source's classes are copied) or the source's /augmented directory. Every "
            "generated image counts toward the org's image quota (no compute credits)."
        ),
        args_schema=_args.AugmentDatasetArgs,
        handler=_h_augment_dataset,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,  # uploads are quota-metered, not credit-charged
    ),
    ToolDescriptor(
        name="tile_dataset",
        description=(
            "Use when the user wants to TILE a dataset for small-object detection "
            "(a Roboflow-style 'Tile' preprocessing step). Slices every source image "
            "into a rowsxcols grid of smaller tiles, translating + clipping each "
            "annotation into its tile, and uploads the tiles. Writes into a new "
            "dataset (the source's classes are copied) or the source's /tiles directory. "
            "Every generated tile counts toward the org's image quota (no compute "
            "credits). Good for aerial / satellite / microscopy imagery."
        ),
        args_schema=_args.TileDatasetArgs,
        handler=_h_tile_dataset,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,  # uploads are quota-metered, not credit-charged
    ),
    ToolDescriptor(
        name="train_pipeline",
        description=(
            "Use when the user wants to train a CV model from an existing annotated "
            "dataset. Creates a timestamped export, kicks off training on the chosen "
            "GPU tier, and (when wait=True) returns the trained model. Pipeline "
            "choice maps to model architecture (yolox=detection, "
            "rfdetr_segmentation=segmentation, etc.)."
        ),
        args_schema=_args.TrainPipelineArgs,
        handler=_h_train_pipeline,
        idempotent=False,
        required_role="member",
        cost_micro_usd=2_000_000,  # rough per-run GPU est. (~2.00 USD)
    ),
    # ─── datasets ───
    ToolDescriptor(
        name="list_datasets",
        description=(
            "Use to enumerate datasets in the user's organization. Returns up to "
            "`limit` datasets with name, image counts, and class definitions."
        ),
        args_schema=_args.ListDatasetsArgs,
        handler=_h_list_datasets,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="get_dataset",
        description=(
            "Use when you need details on a specific dataset by name. "
            "Set include_images=True to also get image summaries (id, filename, "
            "annotation_count) - useful before iterating images."
        ),
        args_schema=_args.GetDatasetArgs,
        handler=_h_get_dataset,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="create_dataset",
        description=(
            "Use to create a new empty dataset. Names must be unique within "
            "the organization. Most workflows auto-create datasets - only call this "
            "directly when the user explicitly asks to create one."
        ),
        args_schema=_args.CreateDatasetArgs,
        handler=_h_create_dataset,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="delete_dataset",
        description=(
            "Use to permanently delete a dataset and all its images. THIS IS "
            "IRREVERSIBLE. Only call after explicit user confirmation."
        ),
        args_schema=_args.DeleteDatasetArgs,
        handler=_h_delete_dataset,
        idempotent=True,
        required_role="admin",
        cost_micro_usd=0,
    ),
    # ─── images ───
    ToolDescriptor(
        name="upload_image",
        description=(
            "Use to upload a single image to a dataset. For bulk uploads from a "
            "directory, prefer upload_dataset_from_directory."
        ),
        args_schema=_args.UploadImageArgs,
        handler=_h_upload_image,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="delete_image",
        description=(
            "Use to permanently delete a single image. THIS IS IRREVERSIBLE. "
            "Only call after explicit user confirmation."
        ),
        args_schema=_args.DeleteImageArgs,
        handler=_h_delete_image,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="review_image",
        description=(
            "Use to approve or request changes on a single image in the annotation "
            "review workflow. 'approve' marks it complete (accept the annotations); "
            "'request_changes' sends it back to the annotator with an optional note. "
            "Ideal for programmatic QA - e.g. auto-approve high-confidence predictions "
            "and bounce low-confidence ones for human correction."
        ),
        args_schema=_args.ReviewImageArgs,
        handler=_h_review_image,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="set_image_split",
        description=(
            "Use to assign an image to a train/val/test dataset split (or clear it). "
            "Enables programmatic dataset organization - e.g. deterministically "
            "partition a dataset 80/10/10 before training."
        ),
        args_schema=_args.SetImageSplitArgs,
        handler=_h_set_image_split,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="rebalance_dataset_splits",
        description=(
            "Use to assign a WHOLE dataset a train/val/test split by ratio in one call "
            "(e.g. 70/20/10) - the fast way to organize a dataset before training, far "
            "quicker than per-image set_image_split. Deterministic under seed; pair with "
            "create_export(organize_by_split=true) for a directly-trainable ZIP."
        ),
        args_schema=_args.RebalanceSplitsArgs,
        handler=_h_rebalance_splits,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    # ─── annotations ───
    ToolDescriptor(
        name="get_annotations",
        description=(
            "Use to read the current annotations on an image. Returns a list of "
            "typed annotation objects (bbox, polygon, polyline, keypoint)."
        ),
        args_schema=_args.GetAnnotationsArgs,
        handler=_h_get_annotations,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="save_annotations",
        description=(
            "Use to replace the annotations on an image (FULL OVERWRITE - pass "
            "the complete list every time). Each annotation requires id, name "
            "(class label), type, and the type-specific geometry field. The class "
            "label MUST exist in the dataset's project_config."
        ),
        args_schema=_args.SaveAnnotationsArgs,
        handler=_h_save_annotations,
        idempotent=True,  # POST is full-replacement, so retries are safe
        required_role="member",
        cost_micro_usd=0,
    ),
    # ─── auto-annotate (single prompt) ───
    ToolDescriptor(
        name="auto_annotate_point",
        description=(
            "Use when the user knows the location of an object and wants SAM3 to "
            "segment it from a click. Provide one or more positive points (label=1) "
            "to include and optional negative points (label=0) to exclude. Returns "
            "annotation(s) - does NOT save them. Call save_annotations to persist."
        ),
        args_schema=_args.AutoAnnotatePointArgs,
        handler=_h_auto_annotate_point,
        idempotent=False,
        required_role="member",
        cost_micro_usd=2_500,  # a few T4 GPU-seconds (~0.0025 USD)
    ),
    ToolDescriptor(
        name="auto_annotate_box",
        description=(
            "Use when the user wants SAM3 to segment everything inside a bounding "
            "box. Returns annotation(s) - does NOT save them. Call save_annotations "
            "to persist."
        ),
        args_schema=_args.AutoAnnotateBoxArgs,
        handler=_h_auto_annotate_box,
        idempotent=False,
        required_role="member",
        cost_micro_usd=2_500,  # a few T4 GPU-seconds (~0.0025 USD)
    ),
    ToolDescriptor(
        name="auto_annotate_text",
        description=(
            "Use when the user wants SAM3 to find all instances of a described class "
            "in an image (e.g. 'find all red cars'). Returns annotation(s) - does NOT "
            "save them. Call save_annotations to persist."
        ),
        args_schema=_args.AutoAnnotateTextArgs,
        handler=_h_auto_annotate_text,
        idempotent=False,
        required_role="member",
        cost_micro_usd=2_500,  # a few T4 GPU-seconds (~0.0025 USD)
    ),
    # ─── search ───
    ToolDescriptor(
        name="search_by_tag",
        description=(
            "Use when the user wants to find images by automatic content tags "
            "(objects/scenes/attributes). Pass at least one of objects, scenes, or "
            "attributes. Empty/None means 'any'."
        ),
        args_schema=_args.SearchByTagArgs,
        handler=_h_search_by_tag,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="search_by_similarity",
        description=(
            "Use when the user wants images visually similar to a reference image "
            "(SigLIP2 cosine similarity). Returns up to `limit` similar images, "
            "ranked by similarity score."
        ),
        args_schema=_args.SearchBySimilarityArgs,
        handler=_h_search_by_similarity,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    # ─── exports ───
    ToolDescriptor(
        name="create_export",
        description=(
            "Use when the user wants a downloadable ZIP of a dataset's images and "
            "annotations in a chosen format. Blocks until the export is built. "
            "Default format is 'pictograph' (canonical JSON); other options support "
            "common ML formats (COCO, YOLO, etc.)."
        ),
        args_schema=_args.CreateExportArgs,
        handler=_h_create_export,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="list_exports",
        description="Use to list previously created exports.",
        args_schema=_args.ListExportsArgs,
        handler=_h_list_exports,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="download_export",
        description=(
            "Use to download an export ZIP to a local file path. Streams the file in "
            "chunks; safe for large exports."
        ),
        args_schema=_args.DownloadExportArgs,
        handler=_h_download_export,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    # ─── training ───
    ToolDescriptor(
        name="get_training_status",
        description=(
            "Use to check the status of a training run by ID. Returns current "
            "status, progress %, current epoch, and metrics."
        ),
        args_schema=_args.GetTrainingStatusArgs,
        handler=_h_get_training_status,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="cancel_training",
        description=(
            "Use to cancel an in-flight training run. Stops the GPU job in-flight; "
            "a run cancelled before it completes is never charged. Only call after "
            "user confirmation."
        ),
        args_schema=_args.CancelTrainingArgs,
        handler=_h_cancel_training,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    # ─── models ───
    ToolDescriptor(
        name="list_models",
        description=(
            "Use to list trained models in the organization. Returns model "
            "metadata (architecture, metrics, status)."
        ),
        args_schema=_args.ListModelsArgs,
        handler=_h_list_models,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="download_model",
        description=(
            "Use to download a trained model's weights to a local file. "
            "Defaults to ONNX; pass format='pytorch' for the native .pth "
            "checkpoint. Only 'ready' models are downloadable."
        ),
        args_schema=_args.DownloadModelArgs,
        handler=_h_download_model,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    # ─── deployments ───
    ToolDescriptor(
        name="list_deployments",
        description="Use to list the organization's model deployments (status, endpoint, accrued cost).",
        args_schema=_args.ListDeploymentsArgs,
        handler=_h_list_deployments,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="get_deployment",
        description="Use to fetch one deployment by UUID - its status, endpoint URL, cost rate, and accrued credits.",
        args_schema=_args.GetDeploymentArgs,
        handler=_h_get_deployment,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="create_deployment",
        description=(
            "Use to deploy a 'ready' trained model to a live inference endpoint on the "
            "chosen compute (CPU/T4/L4/A10G/A100). Billed by uptime; min_containers=0 "
            "scales to zero. Returns a one-time bearer token for the endpoint."
        ),
        args_schema=_args.CreateDeploymentArgs,
        handler=_h_create_deployment,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,  # metered by uptime, not at create time
    ),
    ToolDescriptor(
        name="delete_deployment",
        description=(
            "Use to terminate a deployment and tear down its serving endpoint (stops all billing)."
        ),
        args_schema=_args.DeleteDeploymentArgs,
        handler=_h_delete_deployment,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
    # ─── notifications ───
    ToolDescriptor(
        name="list_notifications",
        description=(
            "Use to poll the organization's notification feed for job-lifecycle "
            "events you (or a teammate) kicked off - training complete, export ready, "
            "batch auto-annotate done. Set unread_only to see just new events. Returns "
            "the notifications plus the org's total unread count."
        ),
        args_schema=_args.ListNotificationsArgs,
        handler=_h_list_notifications,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    # ─── credits ───
    ToolDescriptor(
        name="get_credit_balance",
        description=(
            "Use to check the organization's current compute-credit balance (USD). "
            "Returns the remaining included allowance, overage budget, period spend, "
            "and last 20 ledger entries. Amounts are micro-USD (1 USD = 1,000,000)."
        ),
        args_schema=_args.GetCreditBalanceArgs,
        handler=_h_get_credit_balance,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="estimate_credit_cost",
        description=(
            "Use BEFORE invoking a paid operation (training, batch SAM3, image "
            "generation) to check whether the org has enough compute credit. Returns "
            "the cost in micro-USD (total_micro_usd); inspect .sufficient to gate."
        ),
        args_schema=_args.EstimateCreditCostArgs,
        handler=_h_estimate_credit_cost,
        idempotent=True,
        required_role="viewer",
        cost_micro_usd=0,
    ),
    # ─── connectors ───
    ToolDescriptor(
        name="validate_connector",
        description=(
            "Use to validate a V7 or Roboflow API key and list the user's available "
            "remote datasets. Call before import_from_connector."
        ),
        args_schema=_args.ValidateConnectorArgs,
        handler=_h_validate_connector,
        idempotent=True,
        required_role="member",
        cost_micro_usd=0,
    ),
    ToolDescriptor(
        name="import_from_connector",
        description=(
            "Use to import one or more remote datasets from V7 or Roboflow into "
            "Pictograph. Blocks until the import completes and returns the "
            "terminal import job (status, counts)."
        ),
        args_schema=_args.ImportFromConnectorArgs,
        handler=_h_import_from_connector,
        idempotent=False,
        required_role="member",
        cost_micro_usd=0,
    ),
]
"""Canonical tool list. Adapter outputs derive entirely from this."""


def get_tool(name: str) -> ToolDescriptor:
    """Look up a tool by name. Raises ``KeyError`` if missing."""
    for tool in REGISTRY:
        if tool.name == name:
            return tool
    raise KeyError(f"No agent tool registered with name {name!r}")


def tool_names() -> list[str]:
    """All registered tool names, in registry order."""
    return [t.name for t in REGISTRY]
