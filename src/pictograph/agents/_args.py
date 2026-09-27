"""Pydantic argument schemas for every agent tool.

These models are the source of truth for the tool registry's
``input_schema`` JSON Schemas. Keeping them in their own module
ensures: (1) handlers in ``_registry.py`` import only from here,
(2) Pydantic generates a stable JSON Schema per tool, and
(3) descriptions in ``Field(..., description=...)`` flow through to
the agent-facing schema unmodified - agents read these descriptions
to decide *when* to use a parameter.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ToolArgs(BaseModel):
    """Common config: agents must pass exactly the declared fields."""

    model_config = ConfigDict(extra="forbid")


# ───────────── workflows ─────────────


class UploadDatasetFromDirectoryArgs(_ToolArgs):
    """Walk a local directory, ensure the dataset exists, upload every supported image."""

    dataset_name: str = Field(
        description="Destination dataset name within your organization. Created if missing.",
    )
    directory: str = Field(
        description=(
            "Absolute or ~-relative path to the local directory. Walked recursively. "
            "Supported extensions: jpg/jpeg/png/webp/bmp/tif/tiff/gif/heic."
        ),
    )
    organize_by_class: bool = Field(
        default=True,
        description=(
            "When true, each first-level subdirectory becomes a virtual directory on "
            "the dataset (e.g. 'images/cars/x.jpg' lands under '/cars'). When false, "
            "every image lands at root."
        ),
    )
    max_workers: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Concurrent upload threads. Default 8.",
    )
    skip_existing: bool = Field(
        default=True,
        description=(
            "When true (default), images that already exist in the same directory are "
            "recorded as 'skipped' rather than failures."
        ),
    )


class AutoAnnotateDatasetArgs(_ToolArgs):
    """Run SAM3 over a dataset and save the resulting annotations."""

    dataset_name: str = Field(
        description="Dataset name. Must already exist.",
    )
    classes: list[dict[str, str]] = Field(
        description=(
            "Class configs to detect. Each entry: "
            "{'name': 'car', 'output_type': 'polygon'|'bbox'|'tag'}. "
            "Default output_type is 'polygon'."
        ),
        min_length=1,
    )
    mode: Literal["batch", "text"] = Field(
        default="batch",
        description=(
            "'batch' (default) - async batch job over many images. "
            "'text' - synchronous per-image text prompt (slow; small datasets only)."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="SAM3 confidence cutoff. Detections below this are dropped.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "When false (default), images that already have annotations are skipped. "
            "When true, every image is re-annotated."
        ),
    )
    max_images: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of images processed (useful for dry-runs).",
    )


class TrainPipelineArgs(_ToolArgs):
    """Start a training run on an EXISTING, completed export."""

    dataset_name: str = Field(description="Dataset name.")
    export_name: str = Field(
        description=(
            "Name of the COMPLETED export to train on. Training runs off an "
            "export, never off a dataset - create and inspect the export first."
        )
    )
    pipeline: Literal[
        "yolox",
        "sm_pytorch",
        "classification",
        "rfdetr_detection",
        "rfdetr_segmentation",
        "rfdetr_keypoint",
    ] = Field(
        description=(
            "Training pipeline. yolox=object detection, "
            "sm_pytorch=semantic segmentation, classification=image-level labels, "
            "rfdetr_detection/segmentation/keypoint=DETR-based."
        ),
    )
    gpu: Literal["a10g", "a100", "h100"] = Field(
        default="a10g",
        description="GPU tier. Higher tiers cost more credits per minute.",
    )
    name: str | None = Field(
        default=None,
        description="Training run name. Defaults to a timestamped slug.",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Pipeline-specific hyperparameters (e.g. {'epochs': 50}). Defaults to pipeline defaults.",
    )
    wait: bool = Field(
        default=True,
        description="Block until training reaches a terminal state. Set false to fire-and-forget.",
    )


class AugmentDatasetArgs(_ToolArgs):
    """Generate an augmented version of a dataset and upload the variants."""

    dataset_name: str = Field(
        description="Source dataset name. Must already exist.",
    )
    ops: list[dict[str, Any]] = Field(
        description=(
            "Augmentation ops applied in order, each a dict with an 'op' key. Ops: "
            "flip, vflip, rotate90, rotate (degrees), shear (degrees), resize "
            "(width,height), crop (scale), brightness/contrast/saturation (factor), "
            "hue_shift (degrees), grayscale, blur (radius), noise (amount), cutout "
            "(size). A scalar strength is a range (rotate degrees=15 => +/-15 deg; "
            "brightness factor=0.2 => 0.8-1.2). "
            "Example: [{'op':'flip'},{'op':'rotate','degrees':15},{'op':'brightness','factor':0.2}]."
        ),
        min_length=1,
    )
    multiplier: int = Field(
        default=3, ge=1, description="Augmented variants generated per source image."
    )
    into: str | None = Field(
        default=None,
        description=(
            "Target dataset name (created if missing, copying the source's classes). "
            "Omit to append the variants into the source dataset's /augmented directory."
        ),
    )
    include_original: bool = Field(
        default=True,
        description="When writing to a new dataset, also copy each original image + annotations.",
    )
    seed: int | None = Field(default=None, description="RNG seed for reproducible variants.")
    max_source_images: int | None = Field(
        default=None, ge=1, description="Cap the number of source images processed."
    )


class TileDatasetArgs(_ToolArgs):
    """Slice a dataset into a grid of tiles (small-object-detection preprocessing)."""

    dataset_name: str = Field(
        description="Source dataset name. Must already exist.",
    )
    rows: int = Field(default=2, ge=1, le=10, description="Grid rows per image.")
    cols: int = Field(default=2, ge=1, le=10, description="Grid columns per image.")
    overlap: float = Field(
        default=0.0,
        ge=0.0,
        lt=0.9,
        description="Fractional overlap added to each tile edge (0.0-0.9).",
    )
    min_visibility: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Drop an annotation from a tile when less than this fraction of its area "
            "survives the clip."
        ),
    )
    include_empty: bool = Field(
        default=True,
        description="Keep tiles that have no annotations (set false to exclude them).",
    )
    into: str | None = Field(
        default=None,
        description=(
            "Target dataset name (created if missing, copying the source's classes). "
            "Omit to append the tiles into the source dataset's /tiles directory."
        ),
    )
    max_source_images: int | None = Field(
        default=None, ge=1, description="Cap the number of source images processed."
    )


# ───────────── datasets ─────────────


class ListDatasetsArgs(_ToolArgs):
    """List datasets in the authenticated organization."""

    limit: int = Field(default=100, ge=1, le=1000, description="Max datasets returned.")


class GetDatasetArgs(_ToolArgs):
    """Fetch a dataset by name."""

    name: str = Field(description="Dataset name.")
    include_images: bool = Field(
        default=False,
        description="When true, include the first images_limit image summaries.",
    )
    images_limit: int = Field(default=1000, ge=1, le=10000)


class CreateDatasetArgs(_ToolArgs):
    """Create a new dataset."""

    name: str = Field(description="Dataset name. Unique within the organization.")
    description: str | None = Field(default=None)


class DeleteDatasetArgs(_ToolArgs):
    """Permanently delete a dataset and all its images."""

    name: str = Field(description="Dataset name.")


# ───────────── images ─────────────


class UploadImageArgs(_ToolArgs):
    """Upload a single image to a dataset."""

    dataset_name: str = Field(description="Destination dataset name.")
    file_path: str = Field(description="Absolute path to the local image file.")
    directory_path: str = Field(
        default="/",
        description="Virtual directory path on the dataset (e.g. '/cars').",
    )


class DeleteImageArgs(_ToolArgs):
    """Delete a single image by UUID."""

    dataset_name: str = Field(description="The dataset's name. A UUID is also accepted.")
    image: str = Field(description="The image's FILENAME. An id is also accepted.")


class ReviewImageArgs(_ToolArgs):
    """Approve or request changes on a single image (annotation review workflow)."""

    dataset_name: str = Field(description="The dataset's name. A UUID is also accepted.")
    image: str = Field(description="The image's FILENAME. An id is also accepted.")
    action: Literal["approve", "request_changes"] = Field(
        description=(
            "'approve' marks the image complete (annotations accepted); "
            "'request_changes' sends it back to 'annotate' for correction."
        )
    )
    note: str | None = Field(
        default=None,
        description="Optional note surfaced to the annotator on 'request_changes'.",
    )


class SetImageSplitArgs(_ToolArgs):
    """Assign (or clear) an image's train/val/test dataset split."""

    dataset_name: str = Field(description="The dataset's name. A UUID is also accepted.")
    image: str = Field(description="The image's FILENAME. An id is also accepted.")
    split: Literal["train", "val", "test"] | None = Field(
        default=None,
        description="'train' | 'val' | 'test', or null to clear the split assignment.",
    )


class RebalanceSplitsArgs(_ToolArgs):
    """One-click Rebalance: assign a whole dataset a train/val/test split by ratio."""

    dataset_id: str = Field(description="Dataset UUID to rebalance.")
    train: int = Field(default=70, ge=0, le=100, description="Train weight (percent).")
    val: int = Field(default=20, ge=0, le=100, description="Validation weight (percent).")
    test: int = Field(default=10, ge=0, le=100, description="Test weight (percent).")
    seed: int = Field(default=42, description="Deterministic shuffle seed.")


# ───────────── annotations ─────────────


class GetAnnotationsArgs(_ToolArgs):
    """Fetch the annotations attached to an image."""

    dataset_name: str = Field(description="Dataset name.")
    image: str = Field(description="Image FILENAME within the dataset. An image id also works.")
    directory_path: str | None = Field(
        default=None,
        description="Directory, when the same filename exists in more than one.",
    )


class SaveAnnotationsArgs(_ToolArgs):
    """Replace the annotations on an image (full overwrite)."""

    dataset_name: str = Field(description="Dataset name.")
    image: str = Field(description="Image FILENAME within the dataset. An image id also works.")
    directory_path: str | None = Field(
        default=None,
        description="Directory, when the same filename exists in more than one.",
    )
    annotations: list[dict[str, Any]] = Field(
        description=(
            "List of annotation dicts in canonical Pictograph JSON. Each requires: "
            "id, name, type ('bbox'|'polygon'|'polyline'|'keypoint'), and the type-specific "
            "geometry field (bounding_box / polygon / polyline / keypoint)."
        ),
    )


# ───────────── auto-annotate (single prompt) ─────────────


class AutoAnnotatePointArgs(_ToolArgs):
    """Run a single SAM3 point prompt on an image. Use for 'click here, segment that object'."""

    dataset_name: str = Field(description="Dataset name.")
    image_filename: str = Field(description="Image filename (within the dataset).")
    x: int = Field(description="Anchor point X coordinate in absolute pixels.")
    y: int = Field(description="Anchor point Y coordinate in absolute pixels.")
    name: str = Field(default="object", description="Class label for the resulting annotation.")
    # Each point is exactly [x, y]. Using ``tuple[int, int]`` (not ``list[int]``)
    # makes Pydantic enforce the 2-element shape (minItems/maxItems=2 in the
    # emitted JSON Schema) so a malformed point like ``[5]`` fails with a clean
    # ValidationError at dispatch - NOT a bare IndexError inside the handler's
    # ``(p[0], p[1])`` unpacking.
    positive_points: list[tuple[int, int]] | None = Field(
        default=None,
        description="Extra positive (include) anchor points as [[x, y], ...].",
    )
    negative_points: list[tuple[int, int]] | None = Field(
        default=None,
        description="Negative (exclude) anchor points as [[x, y], ...].",
    )
    score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class AutoAnnotateBoxArgs(_ToolArgs):
    """Run a single SAM3 box prompt on an image. Use for 'segment everything inside this box'."""

    dataset_name: str = Field(description="Dataset name.")
    image_filename: str = Field(description="Image filename.")
    box: dict[str, float] = Field(
        description="Box prompt as {x, y, w, h} in pixel coordinates.",
    )
    name: str = Field(description="Class label.")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_polygon: bool = Field(
        default=True,
        description="When true, return a polygon annotation in addition to the refined bbox.",
    )
    negative_boxes: list[dict[str, float]] | None = Field(
        default=None,
        description="Boxes to exclude from segmentation (shift-drag exclusion zones).",
    )


class AutoAnnotateTextArgs(_ToolArgs):
    """Run a SAM3 open-vocabulary text prompt on an image. Use for 'find all <thing>'."""

    dataset_name: str = Field(description="Dataset name.")
    image_filename: str = Field(description="Image filename.")
    text_prompt: str = Field(
        description="Open-vocabulary text prompt (e.g. 'red cars', 'damaged tiles').",
    )
    output_type: Literal["polygon", "bbox"] = Field(default="polygon")
    # Match the underlying auto_annotate.text resource/CLI default (0.3). The
    # box/point tools use 0.5 because that IS the box/point resource default;
    # text is 0.3, so the agent path must not silently apply a stricter cutoff.
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


# ───────────── search ─────────────


class SearchByTagArgs(_ToolArgs):
    """Find images tagged with one or more auto-tags (objects/scenes/attributes)."""

    dataset_name: str = Field(description="Dataset name.")
    objects: list[str] | None = Field(
        default=None,
        description="Object tags (e.g. ['car', 'truck']). At least one of objects/scenes/attributes required.",
    )
    scenes: list[str] | None = Field(
        default=None,
        description="Scene tags (e.g. ['outdoor', 'urban']).",
    )
    attributes: list[str] | None = Field(
        default=None,
        description="Attribute tags (e.g. ['blurry', 'low-light']).",
    )
    limit: int = Field(default=50, ge=1, le=500)


class SearchBySimilarityArgs(_ToolArgs):
    """Find images visually similar to a reference image (SigLIP2 cosine similarity)."""

    dataset_name: str = Field(description="The dataset's name. A UUID is also accepted.")
    image: str = Field(
        description="The reference image's FILENAME. An id is also accepted. "
        "Scope is its dataset + directory."
    )
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity. Default 0.6 = 'visually related' cutoff.",
    )
    limit: int = Field(default=50, ge=1, le=500)
    directory_path: str | None = Field(
        default=None,
        description="Override the source image's directory. Pass '/' to search the dataset root.",
    )


# ───────────── exports ─────────────


class CreateExportArgs(_ToolArgs):
    """Create a dataset export (ZIP of images + annotations in chosen format)."""

    dataset_name: str = Field(description="Dataset name.")
    name: str = Field(description="Export name. Must be unique within the dataset.")
    format: Literal[
        "pictograph", "darwin", "coco", "csv", "cvat", "datumaro", "labelme", "pascal_voc", "yolo"
    ] = Field(default="pictograph", description="Export format.")
    include_images: bool = Field(
        default=True,
        description="When true, embeds the original image files in the ZIP.",
    )
    class_filter: list[str] | None = Field(
        default=None,
        description="Limit the export to these class names. None = all classes.",
    )
    status_filter: Literal["all", "complete", "in_progress", "new"] = Field(
        default="complete",
        description="Image-status filter. Default 'complete' = annotation-finalised images only.",
    )
    organize_by_split: bool = Field(
        default=False,
        description=(
            "When true, organize the ZIP into train/valid/test directories by each image's "
            "assigned split (unassigned images go to train) - a directly-trainable "
            "YOLO/COCO layout. Assign splits first via set_image_split."
        ),
    )


class ListExportsArgs(_ToolArgs):
    """List all exports in the authenticated organization."""

    limit: int = Field(default=50, ge=1, le=500)


class DownloadExportArgs(_ToolArgs):
    """Download an export ZIP to a local file."""

    dataset_name: str = Field(description="Dataset name.")
    export_name: str = Field(description="Export name.")
    output_path: str = Field(description="Local path to write the ZIP file.")


# ───────────── training ─────────────


class GetTrainingStatusArgs(_ToolArgs):
    """Fetch the current state of a training run."""

    run_id: str = Field(description="Training run UUID.")


class CancelTrainingArgs(_ToolArgs):
    """Cancel an in-flight training run."""

    run_id: str = Field(description="Training run UUID.")


# ───────────── models ─────────────


class ListModelsArgs(_ToolArgs):
    """List trained models in the authenticated organization."""

    limit: int = Field(default=50, ge=1, le=500)


class DownloadModelArgs(_ToolArgs):
    """Download a trained model's weights (ONNX or PyTorch) to a local file."""

    model_id: str = Field(description="Model UUID.")
    output_path: str = Field(description="Local path to write the weights file.")
    format: Literal["onnx", "pytorch", "safetensors"] = Field(
        default="onnx",
        description=(
            "Weights format. 'onnx' (default) for the exported ONNX graph; "
            "'pytorch' for the native .pth checkpoint. 'pytorch' is unavailable "
            "for models trained before dual-format export."
        ),
    )


# ───────────── deployments ─────────────


class ListDeploymentsArgs(_ToolArgs):
    """List model deployments in the authenticated organization."""

    limit: int = Field(default=50, ge=1, le=100)


class GetDeploymentArgs(_ToolArgs):
    """Fetch a single deployment by name (status, endpoint, cost)."""

    deployment: str = Field(description="Deployment name. A UUID is also accepted.")


class CreateDeploymentArgs(_ToolArgs):
    """Deploy a 'ready' trained model to a live inference endpoint. Billed by uptime."""

    model: str = Field(
        description="Name of a 'ready' model with ONNX weights. A UUID is also accepted."
    )
    name: str | None = Field(default=None, description="Optional deployment name.")
    compute_type: Literal["cpu", "gpu"] = Field(default="gpu", description="Compute class.")
    gpu_type: Literal["t4", "l4", "a10g", "a100"] | None = Field(
        default="t4", description="GPU tier - required when compute_type='gpu'."
    )
    min_containers: int = Field(
        default=0, ge=0, le=5, description="Warm instances (0 = scale to zero, cheapest)."
    )
    max_containers: int = Field(default=1, ge=1, le=10, description="Autoscale ceiling.")
    scaledown_window: int = Field(
        default=60, ge=2, le=3600, description="Idle seconds before scaling down."
    )


class DeleteDeploymentArgs(_ToolArgs):
    """Terminate a deployment and tear down its serving endpoint."""

    deployment: str = Field(description="Deployment name. A UUID is also accepted.")


# ───────────── notifications ─────────────


class ListNotificationsArgs(_ToolArgs):
    """List the organization's notifications (job-lifecycle event feed)."""

    unread_only: bool = Field(
        default=False, description="Only return notifications not yet marked read."
    )
    limit: int = Field(default=50, ge=1, le=100, description="Page size (max 100).")


# ───────────── credits ─────────────


class GetCreditBalanceArgs(_ToolArgs):
    """Fetch the organization's current credit balance and last 20 ledger entries."""


class EstimateCreditCostArgs(_ToolArgs):
    """Estimate the compute-credit cost (USD/µUSD) of an operation before invoking it."""

    operation: str = Field(
        description=(
            "Operation slug. Common values: 'training_a10g', 'training_a100', "
            "'training_h100', 'sam3_auto_annotation', 'inference_t4', "
            "'image_generate_imagen_fast', 'image_edit_gemini_flash'."
        ),
    )
    quantity: int = Field(default=1, ge=1, description="Units (minutes/images/runs).")


# ───────────── connectors ─────────────


class ValidateConnectorArgs(_ToolArgs):
    """Validate a V7/Roboflow API key and list available remote datasets."""

    provider: Literal["v7", "roboflow"] = Field(description="Connector provider.")
    api_key: str = Field(description="API key for the remote service.")


class ImportFromConnectorArgs(_ToolArgs):
    """Kick off a remote dataset import from V7 or Roboflow."""

    provider: Literal["v7", "roboflow"] = Field(description="Connector provider.")
    api_key: str = Field(description="API key for the remote service.")
    datasets: list[dict[str, Any]] = Field(
        description=(
            "Remote datasets to import. Each entry: {id, name, slug, "
            "image_count?, version?}. Typically obtained from validate_connector."
        ),
        min_length=1,
    )
