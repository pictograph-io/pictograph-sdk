"""``draw_annotations`` takes what a model gives and the pixels the model read (B613).

A model's result draws as it is (``model.predict``'s, any task, and a hosted run's); a semantic model's
polygons are class REGIONS in THE look's region style; and an image given as a path or bytes is drawn with
its EXIF orientation applied - the pixels OpenCV gives ``model.predict`` - so a phone photo's predictions
land where they were found. A BGR array (a video frame) is read as BGR.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pictograph import draw_annotations
from pictograph.inference.results import (
    ClassificationResult,
    ClassScore,
    DetectionResult,
    InstanceSegmentationResult,
    SemanticSegmentationResult,
)
from pictograph.models.model import ModelPredictResult

GREY = (90, 96, 104)


def _img(size: tuple[int, int] = (400, 300)) -> Image.Image:
    return Image.new("RGB", size, GREY)


BOX: dict[str, Any] = {
    "id": "b",
    "name": "car",
    "type": "bbox",
    "bounding_box": {"x": 100, "y": 100, "w": 120, "h": 80},
    "confidence": 0.8,
}
POLY: dict[str, Any] = {
    "id": "p",
    "name": "road",
    "type": "polygon",
    "polygon": {
        "paths": [
            [{"x": 40, "y": 160}, {"x": 360, "y": 160}, {"x": 360, "y": 280}, {"x": 40, "y": 280}]
        ]
    },
    "confidence": 0.7,
}


def test_a_detection_result_draws_as_its_predictions() -> None:
    result = DetectionResult.model_validate({"predictions": [BOX]})
    assert draw_annotations(_img(), result).tobytes() == draw_annotations(_img(), [BOX]).tobytes()


def test_a_semantic_result_draws_regions_and_an_instance_result_masks() -> None:
    semantic = SemanticSegmentationResult.model_validate({"predictions": [POLY]})
    instance = InstanceSegmentationResult.model_validate({"predictions": [POLY]})
    as_regions = draw_annotations(_img(), semantic)
    assert as_regions.tobytes() == draw_annotations(_img(), [POLY], regions=True).tobytes()
    assert (
        draw_annotations(_img(), instance).tobytes() == draw_annotations(_img(), [POLY]).tobytes()
    )
    # A region is not a mask: 34% under a thin edge, no keyline, no score on its label.
    assert as_regions.tobytes() != draw_annotations(_img(), [POLY]).tobytes()
    # ... and regions=False draws a semantic result's polygons as masks after all.
    assert (
        draw_annotations(_img(), semantic, regions=False).tobytes()
        == draw_annotations(_img(), [POLY]).tobytes()
    )


def test_a_regions_label_names_it_without_a_score() -> None:
    labelled = draw_annotations(_img(), [POLY], regions=True)
    bare = draw_annotations(_img(), [POLY], regions=True, show_confidence=True)
    # The score is never a region's: show_confidence has nothing to add.
    assert labelled.tobytes() == bare.tobytes()


def test_a_classifier_result_as_annotations_is_its_badges() -> None:
    result = ClassificationResult(classes=[ClassScore(name="indoor", confidence=0.9)])
    assert (
        draw_annotations(_img(), result).tobytes()
        == draw_annotations(_img(), classification=result).tobytes()
    )


def test_a_hosted_runs_result_draws_its_annotations_badges_and_regions() -> None:
    boxes = ModelPredictResult(annotations=[BOX], model_type="object_detection")
    assert draw_annotations(_img(), boxes).tobytes() == draw_annotations(_img(), [BOX]).tobytes()
    regions = ModelPredictResult(annotations=[POLY], model_type="semantic_segmentation")
    assert (
        draw_annotations(_img(), regions).tobytes()
        == draw_annotations(_img(), [POLY], regions=True).tobytes()
    )
    scored = ModelPredictResult(
        tags=["indoor", "office"], tag_scores=[0.9, 0.2], model_type="classification"
    )
    expected = ClassificationResult(
        classes=[
            ClassScore(name="indoor", confidence=0.9),
            ClassScore(name="office", confidence=0.2),
        ]
    )
    assert (
        draw_annotations(_img(), scored).tobytes()
        == draw_annotations(_img(), classification=expected).tobytes()
    )
    # A class with no score (a backend before B209) keeps its badge, its name alone.
    unscored = ModelPredictResult(tags=["indoor"], model_type="classification")
    assert draw_annotations(_img(), unscored).tobytes() != _img().tobytes()


def _rotated_jpeg() -> bytes:
    """A 300 x 100 JPEG whose EXIF says: shown turned 90 degrees clockwise (100 x 300), as a phone writes it."""
    im = Image.new("RGB", (300, 100), GREY)
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif.tobytes(), quality=95)
    return buf.getvalue()


def test_a_path_or_bytes_is_drawn_in_its_shown_orientation(tmp_path: Path) -> None:
    data = _rotated_jpeg()
    path = tmp_path / "phone.jpg"
    path.write_bytes(data)
    box = {**BOX, "bounding_box": {"x": 20, "y": 200, "w": 60, "h": 60}}
    from_path = draw_annotations(path, [box], show_labels=False)
    from_bytes = draw_annotations(data, [box], show_labels=False)
    # Drawn on the photo as shown (100 x 300), where the model found the box - not on the stored 300 x 100.
    assert from_path.size == from_bytes.size == (100, 300)
    assert from_path.tobytes() == from_bytes.tobytes()
    assert from_path.getpixel((20, 230)) != from_path.getpixel(
        (50, 150)
    )  # its line, then the photo


def test_a_pillow_image_is_drawn_as_it_is() -> None:
    raw = Image.open(io.BytesIO(_rotated_jpeg()))
    assert draw_annotations(raw, []).size == (300, 100)  # the caller's own pixels: nothing turned


def test_a_bgr_array_is_read_as_bgr() -> None:
    np = pytest.importorskip("numpy")
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :] = (200, 30, 10)  # BGR: a blue frame, as cv2 gives it
    out = draw_annotations(frame, [])
    assert out.size == (60, 40)
    assert out.getpixel((5, 5)) == (10, 30, 200)
    grey = draw_annotations(np.full((20, 30), 77, dtype=np.uint8), [])
    assert grey.getpixel((1, 1)) == (77, 77, 77)


def test_anything_else_is_refused() -> None:
    with pytest.raises(TypeError, match="file path, the image's bytes, a Pillow image or a BGR"):
        draw_annotations(12345, [])  # type: ignore[arg-type]
