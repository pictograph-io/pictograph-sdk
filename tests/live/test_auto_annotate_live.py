"""Live: SAM3 auto-annotation - point / box / text / batch.

Each test hits the GPU pipeline, spending at least 3 credits.
"""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.auto_annotate import BatchJob, PromptResult

pytestmark = [pytest.mark.live, pytest.mark.credits]


def test_sam3_point_prompt(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    img = images[0]
    result = client.auto_annotate.point(
        project.name,
        img.filename,
        x=64,
        y=64,
        name="thing",
    )
    assert isinstance(result, PromptResult)
    assert result.status in {"success", "no_detection", "below_threshold"}


def test_sam3_box_prompt(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    img = images[0]
    result = client.auto_annotate.box(
        project.name,
        img.filename,
        box={"x": 10, "y": 10, "w": 100, "h": 100},
        name="thing",
        confidence_threshold=0.1,
        return_polygon=True,
    )
    assert isinstance(result, PromptResult)


def test_sam3_text_prompt(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    img = images[0]
    result = client.auto_annotate.text(
        project.name,
        img.filename,
        text_prompt="shape",
        output_type="bbox",
        confidence_threshold=0.1,
        max_detections=5,
    )
    assert isinstance(result, PromptResult)


@pytest.mark.slow
def test_sam3_batch_job(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    job = client.auto_annotate.batch(
        project.name,
        [img.filename for img in images[:2]],
        classes=[{"name": "thing", "output_type": "polygon"}],
        confidence_threshold=0.1,
        wait=True,
        poll_interval=3,
        timeout=300,
    )
    assert isinstance(job, BatchJob)
    assert job.status in {"completed", "failed", "completed_with_errors"}
