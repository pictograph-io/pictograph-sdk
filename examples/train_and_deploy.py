"""Train a model from a dataset, then stand it up as a live endpoint.

Training and deployment spend GPU credits, so the actual launch is gated behind
an environment flag. By default this script does the cheap parts (dataset +
export) and prints the training/deploy code without launching it.

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/train_and_deploy.py                 # cheap: prepares data only
    PICTOGRAPH_RUN_TRAINING=1 python examples/train_and_deploy.py   # spends GPU
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image, reusing_existing

from pictograph import BBoxAnnotation, BoundingBox, Client

DATASET = "sdk-example-train"


def prepare_export(client: Client) -> None:
    """Create a tiny annotated dataset and a completed export to train from."""
    with reusing_existing(f"Dataset {DATASET!r}"):
        client.datasets.create(name=DATASET, annotation_types=["bbox"])

    with TemporaryDirectory() as tmp:
        image = demo_image(Path(tmp) / "sample.jpg")
        with reusing_existing(image.name):
            client.images.upload(dataset_name=DATASET, file_path=str(image))
        client.annotations.save(
            dataset_name=DATASET,
            image="sample.jpg",
            annotations=[
                BBoxAnnotation(name="object", bounding_box=BoundingBox(x=40, y=40, w=160, h=120))
            ],
        )
        with reusing_existing("Export 'v1'"):
            client.exports.create(dataset_name=DATASET, name="v1", format="coco")
    print(f"Prepared dataset {DATASET!r} with export 'v1'.")


def train_and_deploy(client: Client) -> None:
    """Launch a real training run, then deploy the resulting model. Spends GPU."""
    run = client.training.create(
        dataset_name=DATASET,
        export_name="v1",
        pipeline_type="rfdetr_detection",
        name="sdk-example-detector",
        config={"epochs": 5},
    )
    print(f"Training run {run.id} started; waiting for completion...")
    client.training.wait_for_completion(run_id=run.id)

    created = client.deployments.create(model="sdk-example-detector", gpu_type="t4")
    print(f"Deployed. Endpoint: {created.deployment.endpoint_url}")
    print("Store this bearer token now - it is shown only once:")
    print(f"  {created.auth_token}")


def main() -> None:
    client = Client()
    prepare_export(client)

    if os.environ.get("PICTOGRAPH_RUN_TRAINING") == "1":
        train_and_deploy(client)
    else:
        print("Set PICTOGRAPH_RUN_TRAINING=1 to actually train + deploy (this spends GPU credits).")


if __name__ == "__main__":
    main()
