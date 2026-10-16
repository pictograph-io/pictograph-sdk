"""Live: batch operations - move / copy / delete / update."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.batch import BatchResult

pytestmark = pytest.mark.live


def test_batch_update_status(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    result = client.batch.update(
        project.name,
        [img.id for img in images[:2]],
        status="review",
    )
    assert isinstance(result, BatchResult)
    assert result.processed >= 1


def test_batch_update_requires_at_least_one_field(
    client: Client, scratch_dataset_with_images
) -> None:
    project, images = scratch_dataset_with_images
    with pytest.raises(ValueError):
        client.batch.update(project.name, [images[0].id])


def test_batch_move(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    result = client.batch.move(
        project.name,
        [images[0].id],
        target_directory_path="/moved",
    )
    assert result.processed >= 1


def test_batch_copy(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    result = client.batch.copy(
        project.name,
        [images[0].id],
        target_directory_path="/copies",
    )
    assert result.processed >= 1


def test_batch_soft_delete_and_restore(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    ids = [images[0].id, images[1].id]
    archive = client.batch.delete(project.name, ids)
    assert archive.processed >= 1

    restore = client.batch.update(project.name, ids, is_archived=False)
    assert restore.processed >= 1
