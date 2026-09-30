"""Unit tests for the RF-DETR keypoint decode path -- the weakest-covered
family and where every confirmed keypoint defect has actually lived.

No trained keypoint weights are available (no API key can reach one), so this
validates the DECODE path with synthetic inputs:

1. `_keypoint_to_annotations` (`pictograph.inference._wrappers.dispatch`) --
   the ONE emitter both engines delegate to.
2. `_rfdetr_raw_outputs` (`pictograph.inference._torch`) -- the rfdetr
   forward-output -> canonical-decode adapter, which replaced the old
   supervision-``KeyPoints`` adapter (that path reduced the raw queries
   DIFFERENTLY from the ONNX canon; see `test_keypoint_backend_parity.py`).
3. `TorchEngine.node_names_for` -- schema-name padding/truncation.
4. `_fetch_keypoint_schema` (`pictograph.inference`) -- pins the
   `download_file(model_id=..., file_name=..., output_path=...)` signature
   (a wrong signature here silently degrades every keypoint model to anonymous
   joints -- it was fixed once already).
5. `KeypointResult` round-trip -- every prediction is one `KeypointAnnotation`,
   and `.instances` regroups them by `instance_id` (a joint is a CLASS, the
   instance is the OBJECT).

Part 1 builds a real `TorchEngine` as the shared emitter's `wrapper` (it
implements `node_names_for`/`keypoint_threshold`/`num_keypoints_per_class`/
`skeleton_edges` faithfully -- it's what the keypoint path itself passes), and
importing `dispatch` pulls in the whole `_wrappers` package (cv2 +
onnxruntime), so that class needs the `[inference]` extra. Parts 3-5 are pure
Python / pydantic and always run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pictograph.inference._torch import TorchEngine
from pictograph.models.model import Model


def _make_engine(
    classes: list[str],
    npc: list[int] | None = None,
    names: dict[str, list[str]] | None = None,
    edges: dict[str, list[list[int]]] | None = None,
    threshold: float = 0.5,
) -> TorchEngine:
    """A real `TorchEngine`, standing in as the `wrapper` argument
    `_keypoint_to_annotations` expects (it is what the torch keypoint path
    passes for itself)."""
    return TorchEngine(
        module=object(),
        family="rfdetr",
        device="cpu",
        dtype=None,
        checkpoint_path=Path("unused.pth"),
        model_type="keypoint_detection",
        architecture="RF-DETR Keypoint Preview",
        classes=classes,
        input_size=(576, 576),
        num_keypoints_per_class=npc,
        keypoint_names=names,
        skeleton_edges=edges,
        keypoint_threshold=threshold,
    )


class TestKeypointToAnnotations:
    """`_keypoint_to_annotations` -- the shared emitter both engines call."""

    @pytest.fixture(autouse=True)
    def _inference_extra(self) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("onnxruntime")

    def test_arity_one_emits_keypoint_with_no_skeleton(self) -> None:
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["nose_point"], npc=[1])
        boxes = [[10.0, 10.0, 30.0, 30.0]]
        scores = [0.9]
        class_ids = [0]
        keypoints = [[[20.0, 15.0, 0.9]]]  # 1 detection, 1 joint: x, y, conf

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["nose_point"], None, 0.5
        )

        assert len(anns) == 1
        ann = anns[0]
        assert ann["type"] == "keypoint"
        assert ann["keypoint"] == {"x": 20.0, "y": 15.0}
        assert "skeleton" not in ann

    def test_multi_joint_class_emits_one_keypoint_per_joint(self) -> None:
        """Arity > 1 -> N `keypoint` annotations sharing one `instance_id`.

        A joint is a CLASS, so each point is named for its OWN joint class, and the
        sub-threshold middle joint is OMITTED: a keypoint annotation has no
        `visibility` field, so absence IS the encoding of "not found"."""
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["person"], npc=[3], names={"person": ["nose", "l_eye", "r_eye"]})
        boxes = [[0.0, 0.0, 100.0, 100.0]]
        scores = [0.9]
        class_ids = [0]
        # Middle joint (l_eye) is below the 0.5 keypoint_threshold.
        keypoints = [[[10.0, 10.0, 0.9], [50.0, 50.0, 0.1], [90.0, 90.0, 0.8]]]

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["person"], None, 0.5
        )

        assert [a["type"] for a in anns] == ["keypoint", "keypoint"]
        assert [a["name"] for a in anns] == ["nose", "r_eye"]
        assert [a["keypoint"] for a in anns] == [
            {"x": 10.0, "y": 10.0},
            {"x": 90.0, "y": 90.0},
        ]
        # ONE object -> ONE shared id. No box on any of them: `KeypointAnnotation`
        # is `extra="forbid"`, so the key would RAISE on a typed read.
        assert [a["instance_id"] for a in anns] == [1, 1]
        assert all("bounding_box" not in a and "skeleton" not in a for a in anns)

    def test_a_wholly_unfindable_object_survives_as_its_best_joint(self) -> None:
        """The successor of "fall back to the detector's box". The OBJECT cleared the
        confidence gate, so dropping it is the silent-drop class; with no box to
        fall back to, the most-findable joint keeps the instance alive."""
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["person"], npc=[2])
        boxes = [[5.0, 5.0, 25.0, 45.0]]
        scores = [0.9]
        class_ids = [0]
        keypoints = [[[1.0, 1.0, 0.1], [2.0, 2.0, 0.05]]]  # both below threshold

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["person"], None, 0.5
        )

        assert len(anns) == 1
        assert anns[0]["keypoint"] == {"x": 1.0, "y": 1.0}  # the higher joint score
        assert anns[0]["instance_id"] == 1

    def test_node_names_fall_back_to_positional_without_a_schema(self) -> None:
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["cls"], npc=[2])  # no `names=` given
        boxes = [[0.0, 0.0, 10.0, 10.0]]
        scores = [0.9]
        class_ids = [0]
        keypoints = [[[1.0, 1.0, 0.9], [2.0, 2.0, 0.9]]]

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["cls"], None, 0.5
        )

        assert [a["name"] for a in anns] == ["point_0", "point_1"]

    def test_instance_ids_are_one_based_and_scoped_to_the_image(self) -> None:
        """Two objects on one image are instances 1 and 2, in DETECTION order."""
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["person"], npc=[2], names={"person": ["nose", "tail"]})
        boxes = [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
        scores = [0.9, 0.8]
        class_ids = [0, 0]
        keypoints = [
            [[1.0, 1.0, 0.9], [2.0, 2.0, 0.9]],
            [[51.0, 51.0, 0.9], [52.0, 52.0, 0.9]],
        ]

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["person"], None, 0.5
        )

        assert [a["instance_id"] for a in anns] == [1, 1, 2, 2]

    def test_the_class_template_never_reaches_an_annotation(self) -> None:
        """The template's edges stay MODEL METADATA. Stamping them onto every point
        is what the removed `skeleton` primitive did, and it was pure duplication -
        the same edge list on every instance of the class."""
        from pictograph.inference._wrappers import dispatch

        wrapper = _make_engine(["person"], npc=[3], edges={"person": [[0, 1], [1, 2]]})
        boxes = [[0.0, 0.0, 10.0, 10.0]]
        scores = [0.9]
        class_ids = [0]
        keypoints = [[[1.0, 1.0, 0.9], [2.0, 2.0, 0.9], [3.0, 3.0, 0.9]]]

        anns = dispatch._keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, wrapper, ["person"], None, 0.5
        )

        assert len(anns) == 3
        assert all("skeleton" not in a for a in anns)
        # ...and it is still on the wrapper, where a consumer can DRAW with it.
        assert wrapper.skeleton_edges == {"person": [[0, 1], [1, 2]]}


class TestRfdetrRawOutputs:
    """`_rfdetr_raw_outputs` -- the adapter that replaced the supervision
    ``KeyPoints`` one. Its whole job is to hand the canonical decode the three
    raw tensors whatever container rfdetr returned them in, so a version that
    switches between the dict and the tuple form cannot silently change what
    the torch backend predicts."""

    def test_dict_output_is_unpacked_in_decode_order(self) -> None:
        np = pytest.importorskip("numpy")
        from pictograph.inference._torch import _rfdetr_raw_outputs

        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        logits = np.ones((1, 2, 3), dtype=np.float32)
        kps = np.full((1, 2, 3, 3), 2.0, dtype=np.float32)

        out = _rfdetr_raw_outputs(
            {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": kps}
        )

        assert out is not None
        np.testing.assert_array_equal(out[0], boxes)
        np.testing.assert_array_equal(out[1], logits)
        np.testing.assert_array_equal(out[2], kps)

    def test_a_detection_only_dict_yields_a_none_keypoint_slot(self) -> None:
        """A 2-output graph is still decodable -- the shared decode treats a
        missing keypoint tensor as "no pose", not as an error."""
        np = pytest.importorskip("numpy")
        from pictograph.inference._torch import _rfdetr_raw_outputs

        out = _rfdetr_raw_outputs(
            {"pred_boxes": np.zeros((1, 2, 4)), "pred_logits": np.zeros((1, 2, 3))}
        )
        assert out is not None
        assert out[2] is None

    def test_tuple_output_from_a_compiled_module_is_accepted(self) -> None:
        np = pytest.importorskip("numpy")
        from pictograph.inference._torch import _rfdetr_raw_outputs

        arrays = (np.zeros((1, 2, 4)), np.zeros((1, 2, 3)), np.zeros((1, 2, 3, 3)))
        out = _rfdetr_raw_outputs(arrays)
        assert out is not None and len(out) == 3

    def test_torch_tensors_are_detached_to_float_numpy(self) -> None:
        """The decode is pure numpy; a live tensor (possibly fp16, possibly on
        an accelerator) has to arrive as a detached float32 host array."""
        torch = pytest.importorskip("torch")
        np = pytest.importorskip("numpy")
        from pictograph.inference._torch import _rfdetr_raw_outputs

        out = _rfdetr_raw_outputs(
            {
                "pred_boxes": torch.zeros((1, 2, 4), dtype=torch.float16),
                "pred_logits": torch.ones((1, 2, 3), dtype=torch.float16),
            }
        )

        assert out is not None
        assert isinstance(out[0], np.ndarray)
        assert out[0].dtype == np.float32

    def test_an_unrecognized_output_shape_is_reported_as_none(self) -> None:
        """Never guess: an rfdetr version returning something else must degrade
        to "no predictions" with a warning, not mis-read the wrong tensor."""
        # `_rfdetr_raw_outputs` hands its result to numpy, absent without the
        # [inference] extra. Its siblings in this module already guard (cv2 /
        # onnxruntime / numpy); this one did not, so it ERRORed in CI.
        pytest.importorskip("numpy", reason="needs the [inference] extra")

        from pictograph.inference._torch import _rfdetr_raw_outputs

        assert _rfdetr_raw_outputs({"logits": 1}) is None
        assert _rfdetr_raw_outputs(object()) is None


class TestNodeNamesFor:
    """`TorchEngine.node_names_for` -- pure Python, no [inference] extra
    needed."""

    def test_pads_with_positional_names_when_schema_names_fewer(self) -> None:
        engine = _make_engine(["person"], npc=[3], names={"person": ["nose"]})
        assert engine.node_names_for("person", 3) == ["nose", "point_1", "point_2"]

    def test_truncates_when_schema_names_more_than_returned(self) -> None:
        engine = _make_engine(
            ["person"], npc=[2], names={"person": ["nose", "l_eye", "r_eye", "l_ear"]}
        )
        assert engine.node_names_for("person", 2) == ["nose", "l_eye"]

    def test_exact_match_passes_through_unchanged(self) -> None:
        engine = _make_engine(["person"], npc=[2], names={"person": ["nose", "l_eye"]})
        assert engine.node_names_for("person", 2) == ["nose", "l_eye"]

    def test_class_with_no_schema_entry_is_fully_positional(self) -> None:
        engine = _make_engine(["person"], npc=[2], names={"other_class": ["a", "b"]})
        assert engine.node_names_for("person", 2) == ["point_0", "point_1"]

    def test_no_schema_at_all_is_fully_positional(self) -> None:
        engine = _make_engine(["person"], npc=[2])
        assert engine.node_names_for("person", 2) == ["point_0", "point_1"]


# ───────────── _fetch_keypoint_schema (pure Python; pins the real bugfix) ─────────────


def _model(**over: object) -> Model:
    base: dict[str, object] = {
        "id": "22222222-2222-2222-2222-222222222222",
        "organization_id": "org",
        "name": "My Pose Model",
        "model_type": "keypoint_detection",
        "architecture": "RF-DETR Keypoint Preview",
        "visibility": "private",
        "status": "ready",
        "class_mapping": {"classes": ["person"]},
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    base.update(over)
    return Model.model_validate(base)


class _FakeModels:
    """Records the EXACT kwargs `download_file` was called with -- that is
    what pins the `file_name=`/`output_path=` signature. `output_path` must
    be honored by actually writing the file there (the real method streams
    to disk and returns a Path; it does not return bytes)."""

    def __init__(
        self, config_json: dict[str, Any] | None = None, raise_on_download: bool = False
    ) -> None:
        self.config_json = config_json
        self.raise_on_download = raise_on_download
        self.download_file_calls: list[dict[str, Any]] = []

    def download_file(self, **kwargs: Any) -> Path:
        self.download_file_calls.append(kwargs)
        if self.raise_on_download:
            raise RuntimeError("simulated download failure")
        output_path = Path(kwargs["output_path"])
        output_path.write_text(json.dumps(self.config_json or {}), encoding="utf-8")
        return output_path


class TestFetchKeypointSchema:
    def test_non_keypoint_model_never_calls_download(self, tmp_path: Path) -> None:
        from pictograph.inference import _fetch_keypoint_schema

        model = _model(model_type="object_detection")
        models = _FakeModels()

        assert _fetch_keypoint_schema(models, model, tmp_path) is None
        assert models.download_file_calls == []

    def test_parses_the_schema_from_the_downloaded_config_and_pins_the_call_kwargs(
        self, tmp_path: Path
    ) -> None:
        from pictograph.inference import _fetch_keypoint_schema

        schema = {
            "class_names": ["person"],
            "num_keypoints_per_class": [3],
            "keypoint_names": {"person": ["nose", "l_eye", "r_eye"]},
            "skeleton": {"person": [[0, 1]]},
        }
        model = _model()
        models = _FakeModels(config_json={"_pictograph": {"keypoint_schema": schema}})

        result = _fetch_keypoint_schema(models, model, tmp_path)

        assert result == schema
        assert len(models.download_file_calls) == 1
        call = models.download_file_calls[0]
        # The exact signature that was previously wrong: `file_name` (not
        # `filename`), and a real `output_path` to stream to (the method
        # returns a Path, not bytes).
        assert call["model_id"] == model.id
        assert call["file_name"] == "config.json"
        assert "filename" not in call
        assert Path(call["output_path"]).exists()

    def test_does_not_redownload_when_the_config_is_already_cached(self, tmp_path: Path) -> None:
        from pictograph.inference import _cache_stem, _fetch_keypoint_schema

        schema = {"class_names": ["person"], "num_keypoints_per_class": [1]}
        model = _model()
        target = tmp_path / f"{_cache_stem(model)}-config.json"
        target.write_text(
            json.dumps({"_pictograph": {"keypoint_schema": schema}}), encoding="utf-8"
        )
        models = _FakeModels()  # would fail loudly if called: config_json is None

        result = _fetch_keypoint_schema(models, model, tmp_path)

        assert result == schema
        assert models.download_file_calls == []

    def test_warns_and_returns_none_when_the_download_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from pictograph.inference import _fetch_keypoint_schema

        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        model = _model()
        models = _FakeModels(raise_on_download=True)

        result = _fetch_keypoint_schema(models, model, tmp_path)

        assert result is None
        assert "config.json" in caplog.text

    def test_warns_and_returns_none_when_config_has_no_keypoint_schema(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from pictograph.inference import _fetch_keypoint_schema

        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        model = _model()
        models = _FakeModels(config_json={"_pictograph": {"model_type": "keypoint_detection"}})

        result = _fetch_keypoint_schema(models, model, tmp_path)

        assert result is None
        assert "no keypoint_schema" in caplog.text


# ───────────── KeypointResult round-trip ─────────────


class TestKeypointResultRoundTrip:
    def test_emitted_annotations_validate_as_keypoints_carrying_instance_id(self) -> None:
        """The emitter's dicts must survive the typed model both twins forbid extras
        on -- that is the round-trip every keypoint proof used to skip."""
        from pictograph.inference.results import KeypointResult

        result = KeypointResult.model_validate(
            {
                "predictions": [
                    {
                        "name": "nose",
                        "type": "keypoint",
                        "keypoint": {"x": 1, "y": 2},
                        "instance_id": 1,
                    },
                    {
                        "name": "l_eye",
                        "type": "keypoint",
                        "keypoint": {"x": 3, "y": 4},
                        "instance_id": 1,
                    },
                    {"name": "tip", "type": "keypoint", "keypoint": {"x": 9, "y": 9}},
                ]
            }
        )
        assert len(result.predictions) == 3
        assert result.points == result.predictions
        # Two joints of ONE object, then the unassociated landmark on its own.
        assert [[p.name for p in inst] for inst in result.instances] == [
            ["nose", "l_eye"],
            ["tip"],
        ]

    def test_a_skeleton_shaped_prediction_is_rejected(self) -> None:
        """The primitive is gone and `KeypointAnnotation` is `extra="forbid"`, so a
        stale producer fails loudly instead of round-tripping a dead shape."""
        from pictograph.inference.results import KeypointResult

        with pytest.raises(ValidationError):
            KeypointResult.model_validate(
                {
                    "predictions": [
                        {
                            "name": "person",
                            "type": "skeleton",
                            "skeleton": {
                                "nodes": [{"name": "nose", "x": 1, "y": 2, "visibility": 2}],
                                "edges": [],
                            },
                        }
                    ]
                }
            )
