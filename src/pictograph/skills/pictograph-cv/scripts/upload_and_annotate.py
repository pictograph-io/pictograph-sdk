#!/usr/bin/env python3
"""Upload a directory of images, and optionally auto-annotate it.

Usage:
    python upload_and_annotate.py --directory ./photos --dataset my-photos
    python upload_and_annotate.py --directory ./photos --dataset my-photos \\
        --classes "person:polygon,car:bbox"
    python upload_and_annotate.py --dataset existing --classes "damage:polygon" --no-upload

Class spec: "name:output_type,..." where output_type is polygon (default),
bbox, or tag. Auto-annotation spends compute credit; --quote prices it first.

Reads PICTOGRAPH_API_KEY from the environment. Prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from pictograph import Client
from pictograph.exceptions import PictographError


def parse_classes(spec: str) -> list[dict[str, str]]:
    """'car:polygon,person:bbox' -> [{'name': 'car', 'output_type': 'polygon'}, ...]"""
    classes: list[dict[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, output_type = part.partition(":")
        classes.append({"name": name.strip(), "output_type": (output_type or "polygon").strip()})
    if not classes:
        raise ValueError(f"No classes parsed from {spec!r}")
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--directory", help="Local directory of images. Required unless --no-upload.")
    parser.add_argument("--classes", help="Comma-separated specs. Omit to upload only.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip upload; annotate an existing dataset.")
    parser.add_argument("--mode", default="batch", choices=["batch", "text"])
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum confidence to keep (0-1).")
    parser.add_argument("--max-images", type=int,
                        help="Cap how many images to annotate.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate images that already have annotations.")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent uploads.")
    parser.add_argument("--quote", action="store_true",
                        help="Price the annotation job and exit without running it.")
    args = parser.parse_args()

    if not args.no_upload and not args.directory:
        parser.error("--directory is required unless --no-upload")
    if args.quote and not args.classes:
        parser.error("--quote needs --classes")

    client = Client()
    output: dict[str, object] = {"dataset": args.dataset}

    if args.quote:
        # Price exactly the images the annotate phase would submit - a quote
        # naming no images prices nothing and would read as "free".
        dataset = client.datasets.get(args.dataset, include_images=True, images_limit=1000)
        candidates = dataset.images or []
        if not args.overwrite:
            candidates = [i for i in candidates if i.annotation_count == 0]
        if args.max_images is not None:
            candidates = candidates[: args.max_images]
        quote = client.auto_annotate.quote(
            dataset_name=args.dataset,
            image_filenames=[i.filename for i in candidates],
            classes=parse_classes(args.classes),
        )
        print(json.dumps({"quote": quote.model_dump(mode="json")}, indent=2))
        return 0

    if not args.no_upload:
        uploaded = client.images.upload_from_directory(
            args.dataset, args.directory, max_workers=args.max_workers,
        )
        output["upload"] = {
            "uploaded": uploaded.images_uploaded,
            "skipped": uploaded.images_skipped,
            "failed": len(uploaded.failures),
            "success": uploaded.success,
        }
        if not uploaded.success:
            output["upload_failures"] = [
                {"path": str(f.path), "reason": f.reason} for f in uploaded.failures[:20]
            ]
            print(json.dumps(output, indent=2))
            return 1

    if args.classes:
        annotated = client.auto_annotate.dataset(
            args.dataset,
            parse_classes(args.classes),
            mode=args.mode,
            confidence_threshold=args.confidence,
            overwrite=args.overwrite,
            max_images=args.max_images,
        )
        output["annotate"] = {
            "processed": annotated.images_processed,
            "annotations_added": annotated.annotations_added,
            "skipped": annotated.images_skipped,
            "capped": annotated.images_capped,
            "failed": len(annotated.failures),
            "job_id": annotated.job_id,
            "success": annotated.success,
        }
        print(json.dumps(output, indent=2))
        return 0 if annotated.success else 1

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PictographError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        sys.exit(1)
