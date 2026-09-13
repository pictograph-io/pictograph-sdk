"""Bulk-upload a local directory of images, then list what landed.

``images.upload_from_directory`` walks a directory, creates the dataset if needed,
and maps subdirectory names onto virtual directories. Partial success does not
raise - inspect the report's ``failures``.

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/upload_directory.py [path/to/images]

With no path argument it generates a small demo directory so it runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image

from pictograph import Client

DATASET = "sdk-example-upload"


def run(directory: str | Path) -> None:
    client = Client()

    report = client.images.upload_from_directory(
        dataset_name=DATASET,
        directory=directory,
        skip_existing=True,  # re-runnable: already-uploaded files are skipped
        create_if_missing=True,  # create the dataset if it does not exist yet
    )
    print(
        f"Uploaded {report.images_uploaded}, skipped {report.images_skipped}, "
        f"{len(report.failures)} failed."
    )
    for failure in report.failures:
        print(f"  failed: {failure.path} - {failure.reason}")

    # List the first page of what is now in the dataset.
    for image in client.images.list(dataset_name=DATASET, limit=10):
        print(f"  {image.filename} [{image.status}]")


def main() -> None:
    if len(sys.argv) > 1:
        run(sys.argv[1])
        return

    # No directory given: fabricate one so the example is self-contained.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(3):
            demo_image(root / f"frame_{i:02d}.jpg", seed=i)
        run(root)


if __name__ == "__main__":
    main()
