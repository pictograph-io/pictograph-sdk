"""`load_model()`'s config.json parsing - the offline-loader contract.

`_parse_model_config` reads exactly the fields the local ONNX builder needs from a
training pipeline's ``config.json`` artifact. Getting this wrong
silently mis-loads a customer's model, so pin the schema hard.
"""

from __future__ import annotations

import pytest

from pictograph.inference import _parse_model_config

# The real artifact shape the training service writes:
# {"_pictograph": {envelope}, "config": {raw training config}}.
_B194 = {
    "_pictograph": {
        "schema_version": 1,
        "architecture": "YOLOX",
        "model_type": "object_detection",
        "precision": "fp32",
        "frameworks": ["onnx", "pytorch"],
        "input_shape": [640, 640],
        "class_names": ["person", "car"],
        "num_classes": 2,
        "class_mapping": {"classes": ["person", "car"]},
        "dataset_name": "Road Signs",
        "export_name": "road-signs-v1",
    },
    "config": {"epochs": 50},
}


def test_parses_the_full_b194_artifact() -> None:
    model_type, arch, classes, shape, name = _parse_model_config(_B194)
    assert model_type == "object_detection"
    assert arch == "YOLOX"
    assert classes == ["person", "car"]
    assert shape == (640, 640)
    assert name == "road-signs-v1"  # export_name wins for the display name


def test_accepts_a_bare_envelope_without_the_pictograph_wrapper() -> None:
    bare = dict(_B194["_pictograph"])
    model_type, arch, classes, shape, _ = _parse_model_config(bare)
    assert (model_type, arch, classes, shape) == (
        "object_detection",
        "YOLOX",
        ["person", "car"],
        (640, 640),
    )


def test_falls_back_to_class_names_when_class_mapping_absent() -> None:
    cfg = {"model_type": "classification", "class_names": ["cat", "dog"]}
    _, _, classes, _, _ = _parse_model_config(cfg)
    assert classes == ["cat", "dog"]


def test_defaults_input_shape_when_missing_or_malformed() -> None:
    for bad in ({}, {"input_shape": None}, {"input_shape": [640]}, {"input_shape": "nope"}):
        cfg = {"model_type": "classification", "class_names": ["a", "b"], **bad}
        _, _, _, shape, _ = _parse_model_config(cfg)
        assert shape == (640, 640)


def test_rejects_a_config_with_no_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        _parse_model_config({"class_mapping": {"classes": ["a"]}})


def test_rejects_a_config_with_no_class_list() -> None:
    with pytest.raises(ValueError, match="class"):
        _parse_model_config({"model_type": "object_detection"})


def test_name_is_empty_when_no_dataset_or_export_name() -> None:
    _, _, _, _, name = _parse_model_config({"model_type": "object_detection", "class_names": ["a"]})
    assert name == ""
