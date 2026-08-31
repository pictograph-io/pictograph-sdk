"""Run inference with one of your trained models.

``models.predict`` runs a single image through a trained model on Pictograph's
GPU service - no local runtime needed. (For local inference use ``models.load``.)

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/predict.py [model-name]

With no model name it picks the first ready model in your organization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_image

from pictograph import Client


def pick_model(client: Client) -> str | None:
    if len(sys.argv) > 1:
        return sys.argv[1]
    for model in client.models.list(limit=50):
        if model.status == "ready":
            return model.name
    return None


def main() -> None:
    client = Client()

    name = pick_model(client)
    if name is None:
        print("No ready models in this organization - train one first (see train_and_deploy.py).")
        return

    print(f"Predicting with model {name!r}...")
    with TemporaryDirectory() as tmp:
        image = demo_image(Path(tmp) / "test.jpg")
        result = client.models.predict(name=name, image=str(image), confidence=0.25)

    # Detection/segmentation models return `annotations` (Pictograph JSON dicts);
    # classification models return `tags` with index-parallel `tag_scores`.
    if result.annotations:
        for ann in result.annotations:
            print(f"  {ann['name']} ({ann['type']}): {round(ann.get('confidence', 1.0), 3)}")
    elif result.tags:
        for tag, score in zip(result.tags, result.tag_scores, strict=False):
            print(f"  {tag}: {round(score, 3)}")
    else:
        print("  no predictions above the confidence threshold.")
    print(f"Inference took {result.inference_seconds:.2f}s.")


if __name__ == "__main__":
    main()
