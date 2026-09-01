"""Wire-field rename: the developer-facing primitive is a *dataset*, so every
response model exposes ``dataset_id`` (and the Task's dataset name as ``dataset``),
not the leaked internal ``project_id``/``project``.

The backend still EMITS the legacy ``project_id``/``project`` keys today, so each
field carries ``validation_alias=AliasChoices("dataset_id", "project_id")`` for
zero-break back-compat. These tests bind BOTH wire directions to the canonical
attribute, and prove the old attribute name is gone.

RED against HEAD before the rename: the models exposed ``.project_id`` and had no
``.dataset_id`` attribute, so every ``m.dataset_id`` assertion below raised
``AttributeError`` and every ``not hasattr(m, "project_id")`` assertion failed.
"""

from __future__ import annotations

import pytest

from pictograph.models.connector import DatasetImportProgress
from pictograph.models.directory import Directory
from pictograph.models.evaluation import ModelEvaluation
from pictograph.models.export import Export
from pictograph.models.search import TaggedImage
from pictograph.models.task import Task

# (model, minimal-required-payload-without-the-dataset-key) for each renamed model.
# The dataset key is injected per-wire-shape by the parametrized test.
_CASES = [
    (
        Task,
        {
            "id": "task-1",
            "title": "label the signs",
            "kind": "annotate",
            "status": "open",
            "created_at": "2026-07-10T00:00:00Z",
            "image_count": 3,
            "assignee_count": 1,
        },
    ),
    (
        Export,
        {
            "id": "exp-1",
            "dataset_name": "road-signs",
            "name": "export-1",
            "format": "coco",
            "status": "completed",
            "created_at": "2026-07-10T00:00:00Z",
        },
    ),
    (
        ModelEvaluation,
        {
            "id": "eval-1",
            "organization_id": "org-1",
            "model_id": "model-1",
            "status": "completed",
        },
    ),
    (
        TaggedImage,
        {
            "id": "img-1",
            "filename": "a.jpg",
            "status": "complete",
            "annotation_count": 2,
        },
    ),
    (
        Directory,
        {
            "id": "dir-1",
            "name": "positive",
            "full_path": "/train/positive",
        },
    ),
    (
        DatasetImportProgress,
        {
            "name": "road-signs",
            "status": "processing",
        },
    ),
]


@pytest.mark.parametrize("model, base", _CASES, ids=[c[0].__name__ for c in _CASES])
@pytest.mark.parametrize("wire_key", ["project_id", "dataset_id"])
def test_both_wire_keys_populate_dataset_id(model: type, base: dict, wire_key: str) -> None:
    """The legacy ``project_id`` (emitted today) and the canonical ``dataset_id``
    both land on ``.dataset_id``."""
    instance = model.model_validate({**base, wire_key: "ds-uuid"})
    assert instance.dataset_id == "ds-uuid"


@pytest.mark.parametrize("model, base", _CASES, ids=[c[0].__name__ for c in _CASES])
def test_legacy_project_id_attribute_is_gone(model: type, base: dict) -> None:
    """The internal name is no longer an attribute a developer can read - so a
    consumer that still says ``.project_id`` fails loudly instead of silently
    diverging."""
    instance = model.model_validate({**base, "project_id": "ds-uuid"})
    assert not hasattr(instance, "project_id")


def test_task_dataset_name_reads_both_wire_keys() -> None:
    """Task's dataset *name* is ``.dataset`` (was ``.project``), reading either the
    legacy ``project`` or the canonical ``dataset`` key."""
    base = {
        "id": "task-1",
        "dataset_id": "ds-uuid",
        "title": "t",
        "kind": "annotate",
        "status": "open",
        "created_at": "2026-07-10T00:00:00Z",
        "image_count": 1,
        "assignee_count": 1,
    }
    assert Task.model_validate({**base, "project": "Road Signs"}).dataset == "Road Signs"
    assert Task.model_validate({**base, "dataset": "Road Signs"}).dataset == "Road Signs"
    assert not hasattr(Task.model_validate(base), "project")
