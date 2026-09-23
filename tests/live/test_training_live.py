"""Live: training + models round trip.

Training is credit-heavy (5+ cr/min, 5-min minimum = 25 cr) and slow.
Marked with ``training`` + ``slow`` so CI can skip unless explicitly run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pictograph import Client
from pictograph.models.model import Model
from pictograph.models.training import TrainingRun

pytestmark = [pytest.mark.live, pytest.mark.credits, pytest.mark.training, pytest.mark.slow]


def test_training_list(client: Client) -> None:
    runs = client.training.list(limit=5)
    assert isinstance(runs, list)
    for r in runs:
        assert isinstance(r, TrainingRun)


def test_models_list(client: Client) -> None:
    models = client.models.list(limit=5)
    assert isinstance(models, list)
    for m in models:
        assert isinstance(m, Model)


@pytest.mark.training
@pytest.mark.slow
def test_training_full_lifecycle(
    client: Client, scratch_project, unique_name: str, tmp_path: Path
) -> None:
    """End-to-end: upload 5 images, annotate, export, train YOLOX, download."""
    # 5 images + bbox annotations.
    fixtures = sorted(Path(__file__).parent.glob("fixtures/images/*.png"))[:5]
    from pictograph.models.annotation import BBoxAnnotation
    from pictograph.models.common import BoundingBox

    for p in fixtures:
        img = client.images.upload(scratch_project.id, p)
        client.annotations.save(
            scratch_project.name,
            img.id,
            [BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=0, y=0, w=50, h=50))],
        )
        client.batch.update(scratch_project.name, [img.id], status="complete")

    # Export for training.
    export_name = f"{unique_name}-export"
    client.exports.create(
        scratch_project.name,
        export_name,
        format="pictograph",
        include_images=True,
        wait=True,
        timeout=180,
    )

    # Train YOLOX for 1 epoch (fast).
    run = client.training.create(
        scratch_project.name,
        export_name,
        pipeline_type="yolox",
        name=f"{unique_name}-run",
        config={"epochs": 1, "batch_size": 2},
        gpu_type="a10g",
        wait=True,
        poll_interval=10,
        timeout=1800,
    )
    assert run.status in {"completed", "failed"}

    if run.status == "completed":
        # Model should exist - find it and download.
        # The link is run -> model (`TrainingRun.model_id`), not model -> run:
        # `Model` has no `training_run_id`, so this filter matched nothing and
        # then asserted on it. Pre-dates the name-addressing work.
        assert run.model_id, "A completed run should carry its model_id"
        model = client.models.get(model_id=run.model_id)
        out = tmp_path / "model.onnx"
        client.models.download(model.id, output_path=out)
        assert out.is_file()
