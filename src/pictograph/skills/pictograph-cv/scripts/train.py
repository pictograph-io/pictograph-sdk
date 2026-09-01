#!/usr/bin/env python3
"""Export a dataset and train a model from it.

Creates a timestamped export, starts the training run, and (by default)
polls until it finishes, then prints the trained model's id.

Usage:
    python train.py --dataset road-signs --pipeline yolox
    python train.py --dataset road-signs --pipeline rfdetr_segmentation \\
        --gpu auto --epochs 50

Reads PICTOGRAPH_API_KEY from the environment. Prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from pictograph import Client
from pictograph.exceptions import PictographError

PIPELINES = (
    "yolox", "rfdetr_detection", "rfdetr_segmentation",
    "rfdetr_keypoint", "sm_pytorch", "classification",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline", required=True, choices=PIPELINES)
    parser.add_argument("--gpu", default="a10g", choices=["a10g", "a100", "h100", "auto"],
                        help="'auto' picks the cheapest tier that fits.")
    parser.add_argument("--name", help="Run name. Defaults to a timestamped slug.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--class-filter",
                        help="Comma-separated class names to train on. Omits the rest.")
    parser.add_argument("--export-name", help="Reuse an existing completed export.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Start the run and exit; print the run id.")
    parser.add_argument("--timeout", type=float, default=7200.0,
                        help="Max seconds to wait when polling (default 7200 = 2h).")
    args = parser.parse_args()

    config: dict[str, object] = {}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    class_filter = (
        [c.strip() for c in args.class_filter.split(",") if c.strip()]
        if args.class_filter
        else None
    )

    client = Client()
    # Training runs off an EXPORT. If one is not named, create it here EXPLICITLY
    # (and wait for it) rather than having training mint one invisibly - an
    # invisible export can be empty, which the GPU only discovers after billing.
    export_name = args.export_name
    if not export_name:
        export_name = f"{args.dataset}-train"
        client.exports.create(
            dataset_name=args.dataset,
            name=export_name,
            format="pictograph",
            include_images=True,
            class_filter=class_filter,
            wait=True,
        )
    run = client.training.create(
        dataset_name=args.dataset,
        export_name=export_name,
        pipeline_type=args.pipeline,
        name=args.name or f"{args.pipeline}-run",
        gpu_type=args.gpu,
        config=config or None,
        wait=not args.no_wait,
        timeout=args.timeout,
    )
    model = client.models.get(model_id=run.model_id) if run.model_id else None

    print(json.dumps({
        "run_id": run.id,
        "status": run.status,
        "progress": run.progress,
        "current_epoch": run.current_epoch,
        "total_epochs": run.total_epochs,
        "metrics": run.metrics,
        "error_message": run.error_message,
        "model_id": model.id if model else None,
        "model_status": model.status if model else None,
    }, indent=2))

    if args.no_wait:
        # Not waiting: an accepted run is a success.
        return 0 if run.status in ("pending", "queued", "running", "completed") else 1
    return 0 if run.status == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PictographError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        sys.exit(1)
