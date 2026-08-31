"""Tests for individual ``pictograph.agents._registry`` handlers.

Each handler maps registry input → SDK call → JSON-serialisable output.
We mock ``client.<resource>.<method>`` to verify each handler:
1. Validates and unpacks args correctly.
2. Calls the right SDK method with the right shape.
3. Coerces the SDK return into JSON-friendly form.

Errors / edge cases are covered in the resource tests; this file verifies
plumbing only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from pictograph.agents import Toolkit
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.auto_annotate import PromptResult
from pictograph.models.common import BoundingBox
from pictograph.models.dataset import Dataset
from pictograph.models.image import Image


def _project() -> Dataset:
    return Dataset(
        id="proj-uuid",
        organization_id="org-uuid",
        name="ds",
        description=None,
        annotation_types=["bbox"],
        classes=[],
        image_count=0,
        completed_image_count=0,
        total_size=0,
        archived_image_count=0,
        created_at="2026-04-19T00:00:00Z",  # type: ignore[arg-type]
    )


def _dataset() -> Dataset:
    return Dataset(
        id="proj-uuid",
        name="ds",
        description=None,
        image_count=0,
        completed_image_count=0,
        total_size=0,
        archived_image_count=0,
        classes=[],
        images=None,
        created_at=datetime.now(timezone.utc),
    )


# ───────────── datasets ─────────────


def test_list_datasets_handler_returns_count_and_items() -> None:
    client = MagicMock()
    client.datasets.list.return_value = [_dataset(), _dataset()]
    tk = Toolkit(client)
    result = tk.dispatch("list_datasets", {"limit": 10})
    client.datasets.list.assert_called_once_with(limit=10)
    assert result["count"] == 2
    assert len(result["datasets"]) == 2


def test_get_dataset_handler_passes_include_images() -> None:
    client = MagicMock()
    client.datasets.get.return_value = _dataset()
    tk = Toolkit(client)
    tk.dispatch("get_dataset", {"name": "ds", "include_images": True, "images_limit": 50})
    kwargs = client.datasets.get.call_args.kwargs
    assert kwargs == {"include_images": True, "images_limit": 50}
    assert client.datasets.get.call_args.args == ("ds",)


def test_create_dataset_handler() -> None:
    client = MagicMock()
    client.datasets.create.return_value = _project()
    tk = Toolkit(client)
    result = tk.dispatch("create_dataset", {"name": "ds"})
    client.datasets.create.assert_called_once_with("ds", description=None)
    assert result["name"] == "ds"


def test_delete_dataset_handler_returns_marker() -> None:
    client = MagicMock()
    tk = Toolkit(client)
    result = tk.dispatch("delete_dataset", {"name": "ds"})
    client.datasets.delete.assert_called_once_with("ds")
    assert result == {"deleted": True, "name": "ds"}


# ───────────── images ─────────────


def test_upload_image_handler_resolves_project_id_first() -> None:
    """Workflow: get project by name → upload with that ID."""
    client = MagicMock()
    client.datasets.get.return_value = _project()
    client.images.upload.return_value = Image(
        id="img-uuid",
        project_id="proj-uuid",
        filename="x.jpg",
        gcs_image_path="x",
        directory_path="/",
        status="new",
        annotation_count=0,
        file_size=100,
        image_url="https://x/img",
        created_at=datetime.now(timezone.utc),
    )
    tk = Toolkit(client)
    result = tk.dispatch(
        "upload_image",
        {"dataset_name": "ds", "file_path": "x.jpg", "directory_path": "/"},
    )
    client.datasets.get.assert_called_once_with("ds")
    upload_kwargs = client.images.upload.call_args.kwargs
    # Images now takes a dataset NAME (an id is accepted and short-circuits).
    # The handler still pre-resolves so it does not repeat the lookup per file.
    assert upload_kwargs["dataset_name"] == "proj-uuid"
    assert upload_kwargs["directory_path"] == "/"
    # file_path is converted to Path internally; the str form survives the dump.
    assert result["filename"] == "x.jpg"


def test_delete_image_handler() -> None:
    client = MagicMock()
    tk = Toolkit(client)
    result = tk.dispatch("delete_image", {"dataset_name": "road-signs", "image": "img-uuid"})
    client.images.delete.assert_called_once_with("road-signs", "img-uuid")
    assert result == {"deleted": True, "dataset_name": "road-signs", "image": "img-uuid"}


# ───────────── annotations ─────────────


def test_save_annotations_handler_validates_via_typeadapter() -> None:
    """Raw dicts are parsed through Annotation TypeAdapter (catches typos)."""
    from pictograph.resources.annotations import SaveResult

    client = MagicMock()
    client.annotations.save.return_value = SaveResult(
        image_id="img", previous_count=0, new_count=1, status="new"
    )
    tk = Toolkit(client)
    result = tk.dispatch(
        "save_annotations",
        {
            # An agent now names the dataset and the image FILENAME, the pair a
            # user can read off the grid - not a uuid it had to fetch first.
            "dataset_name": "road-signs",
            "image": "img-001.jpg",
            "annotations": [
                {
                    "id": "a1",
                    "name": "person",
                    "type": "bbox",
                    "bounding_box": {"x": 0, "y": 0, "w": 10, "h": 10},
                }
            ],
        },
    )
    client.annotations.save.assert_called_once()
    # save(dataset, image, annotations) - the annotations are third now.
    saved_anns = client.annotations.save.call_args.args[2]
    assert len(saved_anns) == 1
    assert isinstance(saved_anns[0], BBoxAnnotation)
    assert result["new_count"] == 1


# ───────────── auto-annotate ─────────────


def test_auto_annotate_point_handler_unpacks_anchor_and_extras() -> None:
    """Args mapped to point() signature; extra anchor lists become tuples."""
    client = MagicMock()
    client.auto_annotate.point.return_value = PromptResult(
        status="success",
        annotations=[],
        score=0.9,
        inference_time=0.5,
    )
    tk = Toolkit(client)
    tk.dispatch(
        "auto_annotate_point",
        {
            "dataset_name": "ds",
            "image_filename": "a.jpg",
            "x": 100,
            "y": 200,
            "name": "car",
            "positive_points": [[110, 210]],
            "negative_points": [[50, 50]],
        },
    )
    kwargs = client.auto_annotate.point.call_args.kwargs
    assert kwargs["x"] == 100
    assert kwargs["y"] == 200
    assert kwargs["name"] == "car"
    assert kwargs["positive_points"] == [(110, 210)]
    assert kwargs["negative_points"] == [(50, 50)]


def test_auto_annotate_point_handler_rejects_malformed_points() -> None:
    """A point that isn't exactly [x, y] fails as ValidationError at dispatch.

    Regression: the handler unpacks ``(p[0], p[1])`` for every supplied
    anchor. Before the args schema constrained inner points to ``tuple[int,
    int]``, a 1-element point (``[5]``) sailed through validation and then
    raised a bare ``IndexError`` inside the handler instead of the documented
    ``ValidationError`` on bad input.
    """
    import pytest
    from pydantic import ValidationError

    tk = Toolkit(MagicMock())
    for bad in ([[5]], [[5, 6, 7]]):
        with pytest.raises(ValidationError):
            tk.dispatch(
                "auto_annotate_point",
                {
                    "dataset_name": "ds",
                    "image_filename": "a.jpg",
                    "x": 1,
                    "y": 2,
                    "positive_points": bad,
                },
            )


def test_auto_annotate_box_handler() -> None:
    client = MagicMock()
    client.auto_annotate.box.return_value = PromptResult(
        status="success",
        annotations=[
            BBoxAnnotation(
                id="a1",
                name="car",
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
            )
        ],
    )
    tk = Toolkit(client)
    result = tk.dispatch(
        "auto_annotate_box",
        {
            "dataset_name": "ds",
            "image_filename": "a.jpg",
            "box": {"x": 0, "y": 0, "w": 100, "h": 100},
            "name": "car",
        },
    )
    client.auto_annotate.box.assert_called_once()
    assert result["status"] == "success"
    assert len(result["annotations"]) == 1


def test_auto_annotate_text_handler() -> None:
    client = MagicMock()
    client.auto_annotate.text.return_value = PromptResult(
        status="success",
        annotations=[],
    )
    tk = Toolkit(client)
    tk.dispatch(
        "auto_annotate_text",
        {
            "dataset_name": "ds",
            "image_filename": "a.jpg",
            "text_prompt": "find all cars",
        },
    )
    kwargs = client.auto_annotate.text.call_args.kwargs
    assert kwargs["text_prompt"] == "find all cars"
    assert kwargs["output_type"] == "polygon"  # default
    # The omitted threshold must materialize as the text resource/CLI default
    # (0.3), NOT a stricter 0.5 - the agent path must not silently raise the
    # cutoff for an agent that doesn't pass one.
    assert kwargs["confidence_threshold"] == 0.3


# ───────────── search ─────────────


def test_search_by_tag_handler() -> None:
    client = MagicMock()
    client.search.by_tag.return_value = []
    tk = Toolkit(client)
    result = tk.dispatch(
        "search_by_tag",
        {"dataset_name": "ds", "objects": ["car"], "limit": 25},
    )
    client.search.by_tag.assert_called_once_with(
        dataset_name="ds",
        objects=["car"],
        scenes=None,
        attributes=None,
        limit=25,
    )
    assert result == {"images": [], "count": 0}


def test_search_by_similarity_handler_passes_the_dataset_and_filename_positionally() -> None:
    client = MagicMock()
    client.search.by_similarity.return_value = []
    tk = Toolkit(client)
    tk.dispatch(
        "search_by_similarity",
        {"dataset_name": "road-signs", "image": "stop.jpg", "threshold": 0.7, "limit": 30},
    )
    args = client.search.by_similarity.call_args
    assert args.args == ("road-signs", "stop.jpg")
    assert args.kwargs["threshold"] == 0.7
    assert args.kwargs["limit"] == 30


# ───────────── credits ─────────────


def test_get_credit_balance_handler_takes_no_args() -> None:
    from pictograph.models.credit import CreditBalance

    client = MagicMock()
    client.credits.balance.return_value = CreditBalance(
        included_remaining_micro_usd=999_000_000,
        included_allowance_micro_usd=1_000_000_000,
        credits_reset_at=None,
        recent_history=[],
    )
    tk = Toolkit(client)
    result = tk.dispatch("get_credit_balance", {})
    assert result["included_remaining_micro_usd"] == 999_000_000


# ───────────── connectors ─────────────


def test_validate_connector_handler() -> None:
    from pictograph.models.connector import ValidationResult

    client = MagicMock()
    client.connectors.validate.return_value = ValidationResult(
        valid=True,
        provider="v7",
        datasets=[],
    )
    tk = Toolkit(client)
    result = tk.dispatch(
        "validate_connector",
        {"provider": "v7", "api_key": "x"},
    )
    client.connectors.validate.assert_called_once_with(provider="v7", api_key="x")
    assert result["valid"] is True


def test_import_from_connector_handler_passes_datasets_positionally() -> None:
    from pictograph.models.connector import ImportJob

    client = MagicMock()
    client.connectors.import_.return_value = ImportJob(
        import_id="imp-1",
        status="processing",
        datasets=[],
    )
    tk = Toolkit(client)
    tk.dispatch(
        "import_from_connector",
        {
            "provider": "v7",
            "api_key": "x",
            "datasets": [{"id": "1", "name": "n", "slug": "s"}],
        },
    )
    args = client.connectors.import_.call_args.args
    assert args[0] == "v7"
    assert args[1] == "x"
    assert args[2] == [{"id": "1", "name": "n", "slug": "s"}]


# ───────────── augment ─────────────


def test_augment_dataset_handler_builds_ops_and_calls_the_images_resource() -> None:
    from pictograph.resources.images import AugmentReport

    client = MagicMock()
    tk = Toolkit(client)
    report = AugmentReport(source="ds", target="ds-aug", source_images=2, variants_created=6)
    client.images.augment.return_value = report
    result = tk.dispatch(
        "augment_dataset",
        {
            "dataset_name": "ds",
            "ops": [{"op": "flip"}, {"op": "rotate", "degrees": 15}],
            "multiplier": 3,
            "into": "ds-aug",
        },
    )
    assert result["variants_created"] == 6
    args, kwargs = client.images.augment.call_args
    assert args[0] == "ds"
    # the JSON specs were turned into typed ops in order
    assert [o.name for o in args[1]] == ["horizontal_flip", "rotate"]
    assert kwargs["multiplier"] == 3
    assert kwargs["into"] == "ds-aug"


def test_augment_dataset_handler_rejects_unknown_op() -> None:
    import pytest

    tk = Toolkit(MagicMock())
    with pytest.raises(ValueError, match="Unknown augmentation op"):
        tk.dispatch("augment_dataset", {"dataset_name": "ds", "ops": [{"op": "teleport"}]})


# ───────────── tile ─────────────


def test_tile_dataset_handler_passes_grid_and_calls_the_images_resource() -> None:
    from pictograph.resources.images import TileReport

    client = MagicMock()
    tk = Toolkit(client)
    report = TileReport(source="ds", target="ds-tiled", source_images=2, tiles_created=8)
    client.images.tile.return_value = report
    result = tk.dispatch(
        "tile_dataset",
        {
            "dataset_name": "ds",
            "rows": 3,
            "cols": 2,
            "overlap": 0.1,
            "into": "ds-tiled",
            "include_empty": False,
        },
    )
    assert result["tiles_created"] == 8
    _args, kwargs = client.images.tile.call_args
    assert kwargs["rows"] == 3
    assert kwargs["cols"] == 2
    assert kwargs["overlap"] == 0.1
    assert kwargs["into"] == "ds-tiled"
    assert kwargs["include_empty"] is False


def test_tile_dataset_handler_rejects_bad_overlap() -> None:
    import pytest

    tk = Toolkit(MagicMock())
    # overlap must be < 0.9 (Field lt constraint) → validation error before dispatch
    with pytest.raises(ValueError):
        tk.dispatch("tile_dataset", {"dataset_name": "ds", "overlap": 1.5})


# ───────────── notifications ─────────────


def test_list_notifications_handler_returns_feed_and_unread_count() -> None:
    from pictograph.models.notification import Notification

    client = MagicMock()
    client.notifications.list.return_value = [
        Notification(
            id="n1",
            organization_id="o",
            type="training_complete",
            title="Done",
            read=False,
            created_at="2026-07-07T00:00:00Z",  # type: ignore[arg-type]
        )
    ]
    client.notifications.unread_count.return_value = 3
    tk = Toolkit(client)
    result = tk.dispatch("list_notifications", {"unread_only": True, "limit": 10})
    assert result["count"] == 1
    assert result["unread_count"] == 3
    assert result["notifications"][0]["type"] == "training_complete"
    _args, kwargs = client.notifications.list.call_args
    assert kwargs["unread_only"] is True
    assert kwargs["limit"] == 10
