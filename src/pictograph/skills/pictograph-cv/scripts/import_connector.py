#!/usr/bin/env python3
"""List and import datasets from V7 (Darwin) or Roboflow.

Usage:
    # See what's available on the source account.
    python import_connector.py --provider v7 --api-key $V7_KEY --list

    # Import one or more of them.
    python import_connector.py --provider v7 --api-key $V7_KEY \\
        --dataset-ids ds_abc,ds_xyz

--api-key is the SOURCE provider's key. Your Pictograph key is read from
PICTOGRAPH_API_KEY. Prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import os
import json
import sys

from pictograph import Client
from pictograph.exceptions import PictographError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["v7", "roboflow"])
    # NOT required: a credential on argv is visible in shell history and to any
    # user running `ps`. PICTOGRAPH_SOURCE_KEY is the safe path; --api-key stays
    # for interactive use and is documented as the lesser option.
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PICTOGRAPH_SOURCE_KEY"),
        help="The SOURCE provider's API key. Prefer the PICTOGRAPH_SOURCE_KEY "
             "environment variable - a key passed here is visible in `ps`.",
    )
    parser.add_argument("--list", action="store_true",
                        help="List the available datasets and exit.")
    parser.add_argument("--dataset-ids",
                        help="Comma-separated remote dataset ids. Required unless --list.")
    parser.add_argument("--check-limits", action="store_true",
                        help="Check the import against your plan's limits before starting.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Start the import and exit; print the import id.")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Max seconds to wait when polling (default 3600 = 1h).")
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "no source credential. Set PICTOGRAPH_SOURCE_KEY (preferred - it keeps "
            "the key out of shell history and out of `ps`) or pass --api-key."
        )

    client = Client()

    validation = client.connectors.validate(provider=args.provider, api_key=args.api_key)
    if not validation.valid:
        print(json.dumps({"error": validation.error or "API key validation failed",
                          "valid": False}), file=sys.stderr)
        return 2

    if args.list:
        print(json.dumps({
            "provider": args.provider,
            "workspace": validation.workspace,
            "datasets": [d.model_dump(exclude_none=True) for d in validation.datasets],
        }, indent=2))
        return 0

    if not args.dataset_ids:
        parser.error("--dataset-ids is required (or use --list to enumerate)")

    requested = {x.strip() for x in args.dataset_ids.split(",") if x.strip()}
    selected = [d for d in validation.datasets if d.id in requested]
    if not selected:
        print(json.dumps({
            "error": "None of the requested ids exist on that account",
            "requested": sorted(requested),
            "available": [d.id for d in validation.datasets],
        }), file=sys.stderr)
        return 2

    if args.check_limits:
        limits = client.connectors.check_limits(
            total_images=sum(d.image_count or 0 for d in selected),
            estimated_size_bytes=0,
        )
        if not limits.allowed:
            print(json.dumps({"error": "Import exceeds plan limits",
                              "limits": limits.model_dump(mode="json", exclude_none=True)}),
                  file=sys.stderr)
            return 2

    job = client.connectors.import_(
        args.provider, args.api_key, selected,
        wait=not args.no_wait, timeout=args.timeout,
    )
    print(json.dumps({
        "import_id": job.import_id,
        "status": job.status,
        "progress": job.progress,
        "total_images": job.total_images,
        "imported_images": job.imported_images,
        "failed_images": job.failed_images,
        "datasets": [d.model_dump(exclude_none=True) for d in job.datasets],
    }, indent=2))
    return 0 if job.status in ("completed", "processing") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PictographError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        sys.exit(1)
