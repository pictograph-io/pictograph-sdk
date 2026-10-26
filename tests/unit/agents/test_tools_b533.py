"""Agent tools that mirror the CLI an agent drives (B533, 1.69.123).

* ``auto_annotate_dataset`` drafts one folder (``directory_path``) or with a trained
  ``model``, as ``pictograph auto-annotate dataset`` does - and refuses the two
  combinations the SDK would refuse only after reading the dataset.
* ``train_pipeline`` and ``quote_training_run`` take ``gpu="auto"``.
* ``quote_training_run`` reaches ``client.training.quote`` with the plan unchanged.

Each tool is DISPATCHED (``Toolkit.dispatch``) against a mock client, so what is
asserted is the call the handler really makes, not the schema alone.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from pictograph.agents import Toolkit, get_tool
from pictograph.agents._args import AutoAnnotateDatasetArgs


def _dispatch(tool: str, args: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    Toolkit(client).dispatch(tool, args)
    return client


def test_dataset_drafts_take_a_folder_and_a_model() -> None:
    client = _dispatch(
        "auto_annotate_dataset",
        {"dataset_name": "shelves", "model": "Shelf Detector", "directory_path": "/val"},
    )
    call = client.auto_annotate.dataset.call_args
    assert call.args == ("shelves", [])
    assert call.kwargs["model"] == "Shelf Detector"
    assert call.kwargs["directory_path"] == "/val"


def test_dataset_drafts_with_sam3_still_name_classes() -> None:
    client = _dispatch(
        "auto_annotate_dataset",
        {"dataset_name": "shelves", "classes": [{"name": "car", "output_type": "bbox"}]},
    )
    call = client.auto_annotate.dataset.call_args
    assert call.args == ("shelves", [{"name": "car", "output_type": "bbox"}])
    assert call.kwargs["model"] is None and call.kwargs["directory_path"] is None


@pytest.mark.parametrize(
    ("args", "says"),
    [
        ({"dataset_name": "shelves"}, "name the classes"),
        ({"dataset_name": "shelves", "model": "m", "mode": "text"}, "'batch' mode only"),
    ],
)
def test_dataset_drafts_refuse_what_the_sdk_cannot_run(args: dict[str, Any], says: str) -> None:
    with pytest.raises(ValidationError, match=says):
        AutoAnnotateDatasetArgs.model_validate(args)


def test_the_schema_offers_the_folder_and_the_model() -> None:
    schema = get_tool("auto_annotate_dataset").args_schema.model_json_schema()
    assert {"directory_path", "model"} <= set(schema["properties"])
    assert "classes" not in schema.get("required", [])


def test_quote_reaches_the_sdk_quote_with_the_plan() -> None:
    client = _dispatch(
        "quote_training_run",
        {
            "dataset_name": "shelves",
            "export_name": "v1",
            "pipeline": "rfdetr_detection",
            "gpu": "auto",
            "config": {"epochs": 12},
        },
    )
    client.training.quote.assert_called_once_with(
        "shelves", "v1", pipeline_type="rfdetr_detection", gpu_type="auto", config={"epochs": 12}
    )
    client.training.create.assert_not_called()


def test_the_quote_is_free_and_any_role_may_ask() -> None:
    tool = get_tool("quote_training_run")
    assert tool.cost_micro_usd == 0 and tool.required_role == "viewer" and tool.idempotent


def test_train_pipeline_takes_auto() -> None:
    client = _dispatch(
        "train_pipeline",
        {"dataset_name": "shelves", "export_name": "v1", "pipeline": "yolox", "gpu": "auto"},
    )
    assert client.training.create.call_args.kwargs["gpu_type"] == "auto"


def test_train_pipeline_no_longer_says_it_creates_an_export() -> None:
    """Training runs off an EXPORT the caller names; the tool used to tell agents it
    made one, which it has not done since 2026-07-31."""
    description = get_tool("train_pipeline").description
    assert "Creates a timestamped export" not in description
    assert "COMPLETED export" in description and "quote_training_run" in description
