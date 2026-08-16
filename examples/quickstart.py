"""End-to-end quick start: create a dataset, upload an image, annotate it, export.

The complete data journey with no GPU spend. Run it with a live key:

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/quickstart.py

Resources are addressed by NAME everywhere - there are no ids to look up first.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image, reusing_existing

from pictograph import BBoxAnnotation, BoundingBox, Client

DATASET = "sdk-example-quickstart"


def main() -> None:
    client = Client()  # reads PICTOGRAPH_API_KEY from the environment

    # 1. Create the dataset (idempotent: reuse it if a previous run made it).
    with reusing_existing(f"Dataset {DATASET!r}"):
        client.datasets.create(name=DATASET, annotation_types=["bbox"])
        print(f"Created dataset {DATASET!r}.")

    with TemporaryDirectory() as tmp:
        # 2. Upload an image (generated locally so this runs with no assets).
        image_path = demo_image(Path(tmp) / "street.jpg")
        with reusing_existing("street.jpg"):
            client.images.upload(dataset_name=DATASET, file_path=str(image_path))
        print("Uploaded street.jpg.")

        # 3. Save a bounding-box annotation. The class field is `name`, never `class`.
        saved = client.annotations.save(
            dataset_name=DATASET,
            image="street.jpg",
            annotations=[
                BBoxAnnotation(
                    name="stop sign",
                    bounding_box=BoundingBox(x=40, y=40, w=160, h=120),
                ),
            ],
        )
        print(f"Saved {saved.new_count} annotation(s); image is now {saved.status}.")

        # 4. Read them back. The result is a discriminated union on `type`, so
        #    narrow before touching geometry - a keypoint has no bounding box.
        for ann in client.annotations.get(dataset_name=DATASET, image="street.jpg"):
            if isinstance(ann, BBoxAnnotation):
                print(f"  {ann.type}: {ann.name} @ {ann.bounding_box}")
            else:
                print(f"  {ann.type}: {ann.name}")

        # 5. Export the dataset as COCO and download the ZIP.
        with reusing_existing("Export 'v1'"):
            client.exports.create(dataset_name=DATASET, name="v1", format="coco")
        out = client.exports.download(
            dataset_name=DATASET,
            export_name="v1",
            output_path=str(Path(tmp) / "export.zip"),
        )
        print(f"Downloaded export: {out.name} ({out.stat().st_size} bytes).")

    # Delete the dataset from the app, or with client.datasets.delete(...), when done.


if __name__ == "__main__":
    main()
