"""Search images by auto-tag, by visual similarity, and find near-duplicates.

Every uploaded image is embedded (SigLIP) and auto-tagged automatically, which
powers three read paths:

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/search.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image, reusing_existing

from pictograph import Client
from pictograph.exceptions import NotFoundError

DATASET = "sdk-example-search"


def main() -> None:
    client = Client()

    with reusing_existing(f"Dataset {DATASET!r}"):
        client.datasets.create(name=DATASET, annotation_types=["bbox"])

    with TemporaryDirectory() as tmp:
        for i in range(3):
            path = demo_image(Path(tmp) / f"img_{i}.jpg", seed=i)
            with reusing_existing(path.name):
                client.images.upload(dataset_name=DATASET, file_path=str(path))

    # 1. Search by auto-tag. Each category ANDs its tags; categories AND together.
    tagged = client.search.by_tag(objects=["car"], limit=5)
    print(f"by_tag(objects=['car']): {len(tagged)} image(s) org-wide.")

    # 2. Visually similar images to one of the dataset's images. Embeddings are
    #    computed on upload, so a brand-new image may not be indexed for a moment.
    try:
        similar = client.search.by_similarity(
            dataset_name=DATASET, image="img_0.jpg", threshold=0.5, limit=5
        )
        print(f"by_similarity(img_0.jpg): {len(similar)} similar image(s).")
    except NotFoundError:
        print("by_similarity: img_0.jpg not embedded yet (retry in a moment).")

    # 3. Cluster near-duplicates across the dataset.
    dupes = client.datasets.near_duplicates(name=DATASET, threshold=0.9)
    print(
        f"near_duplicates: {dupes.group_count} cluster(s), "
        f"{dupes.redundant_count} redundant image(s)."
    )


if __name__ == "__main__":
    main()
