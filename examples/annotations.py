"""Every annotation geometry type, constructed and saved programmatically.

Pictograph has four annotation types: bbox, polygon, polyline and keypoint. A
ROTATED box is a bbox with an ``oriented_box``; a multi-joint pose is several
keypoints sharing an ``instance_id`` (there is no separate skeleton type).

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/annotations.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image, reusing_existing

from pictograph import (
    Annotation,
    BBoxAnnotation,
    BoundingBox,
    Client,
    KeypointAnnotation,
    Point,
    PolygonAnnotation,
    PolygonGeometry,
    PolylineAnnotation,
    PolylineGeometry,
)

DATASET = "sdk-example-annotations"


def main() -> None:
    client = Client()

    with reusing_existing(f"Dataset {DATASET!r}"):
        client.datasets.create(
            name=DATASET,
            annotation_types=["bbox", "polygon", "polyline", "keypoint"],
        )

    with TemporaryDirectory() as tmp:
        image_path = demo_image(Path(tmp) / "scene.jpg", shapes=4)
        with reusing_existing("scene.jpg"):
            client.images.upload(dataset_name=DATASET, file_path=str(image_path))

        # Annotate explicitly: a bare list literal of mixed members infers as
        # the common base, which does not satisfy the discriminated union.
        annotations: list[Annotation] = [
            # A bounding box: top-left corner + size, in absolute pixels.
            BBoxAnnotation(name="car", bounding_box=BoundingBox(x=30, y=30, w=180, h=120)),
            # A polygon: a list of rings; the first is the outer boundary, each
            # later ring carves a hole (even-odd fill). Each ring needs >= 3 points.
            PolygonAnnotation(
                name="road",
                polygon=PolygonGeometry(
                    paths=[[Point(x=0, y=400), Point(x=640, y=400), Point(x=320, y=250)]]
                ),
            ),
            # A polyline: an open path of >= 2 points. Does not close.
            PolylineAnnotation(
                name="lane-marking",
                polyline=PolylineGeometry(path=[Point(x=100, y=460), Point(x=540, y=300)]),
            ),
            # Keypoints: a point has no bounding box. Two joints of ONE object
            # share an instance_id; the joint each denotes is its `name`.
            KeypointAnnotation(name="headlight", keypoint=Point(x=60, y=90), instance_id=1),
            KeypointAnnotation(name="wheel", keypoint=Point(x=180, y=140), instance_id=1),
        ]

        saved = client.annotations.save(
            dataset_name=DATASET, image="scene.jpg", annotations=annotations
        )
        print(f"Saved {saved.new_count} annotations of every type.")

        by_type: dict[str, int] = {}
        for ann in client.annotations.get(dataset_name=DATASET, image="scene.jpg"):
            by_type[ann.type] = by_type.get(ann.type, 0) + 1
        print("Read back:", by_type)


if __name__ == "__main__":
    main()
