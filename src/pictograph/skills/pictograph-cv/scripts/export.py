#!/usr/bin/env python3
"""Create and download a dataset export.

Usage:
    python export.py --dataset road-signs --name for-training \\
        --format coco --output ./road-signs.zip

Reads PICTOGRAPH_API_KEY from the environment. Prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pictograph import Client
from pictograph.exceptions import PictographError

FORMATS = (
    "pictograph", "darwin", "coco", "yolo", "yolo_obb", "yolo_pose",
    "dota", "pascal_voc", "cvat", "datumaro", "labelme", "csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True, help="Export name (unique per dataset).")
    parser.add_argument("--format", default="pictograph", choices=FORMATS)
    parser.add_argument("--output", required=True, help="Local path for the ZIP.")
    parser.add_argument("--no-images", action="store_true",
                        help="Annotations only; leave the image files out of the ZIP.")
    parser.add_argument("--class-filter",
                        help="Comma-separated class names. Omits every other class.")
    parser.add_argument("--status-filter", default="complete",
                        choices=["all", "complete", "in_progress", "new"],
                        help="Which images to include (default: only finished ones).")
    parser.add_argument("--organize-by-split", action="store_true",
                        help="Write train/val/test subdirectories.")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Max seconds to wait for the export to build.")
    args = parser.parse_args()

    class_filter = (
        [c.strip() for c in args.class_filter.split(",") if c.strip()]
        if args.class_filter
        else None
    )

    client = Client()
    export = client.exports.create(
        args.dataset,
        args.name,
        format=args.format,
        include_images=not args.no_images,
        class_filter=class_filter,
        status_filter=args.status_filter,
        organize_by_split=args.organize_by_split,
        wait=True,
        timeout=args.timeout,
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client.exports.download(args.dataset, args.name, output_path=output_path)

    print(json.dumps({
        "export_id": export.id,
        "format": args.format,
        "status": export.status,
        "image_count": export.image_count,
        "annotation_count": export.annotation_count,
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }, indent=2))
    return 0 if export.status == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PictographError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        sys.exit(1)
