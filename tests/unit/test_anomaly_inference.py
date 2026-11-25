"""Anomaly detection in the SDK: the typed result, its map, and every local runtime (ROADMAP Program N, N10).

The result builder and the map need no optional dependency. The runtime tests build a real (tiny, untrained) PatchCore
from the vendored definitions, export it to ONNX the way the training pipeline does, save its ``.pth`` and
``.safetensors`` the way the pipeline saves them, calibrate it the way the pipeline calibrates (the highest normal
score is the threshold), and then hold every local format to the same answer: ONNX Runtime and the rebuilt torch
module drive ONE vendored decode, so they may differ only by the forward pass's float noise.
"""

from __future__ import annotations

import base64
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from pictograph.inference.results import (
    TASK_RESULT_TYPES,
    AnomalyMap,
    AnomalyResult,
    build_result,
)

REGION = {
    "id": "r1",
    "name": "defect",
    "type": "polygon",
    "bounding_box": {"x": 10.0, "y": 12.0, "w": 30.0, "h": 20.0},
    "polygon": {
        "paths": [[{"x": 10.0, "y": 12.0}, {"x": 40.0, "y": 12.0}, {"x": 40.0, "y": 32.0}]]
    },
    "confidence": 0.91,
    "attributes": ["auto-annotate"],
}


def _payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": "anomaly_detection",
        "predictions": [REGION],
        "tags": ["defect"],
        "labels": [{"class": "defect", "confidence": 0.83}],
        "anomaly_score": 0.83,
        "is_anomalous": True,
    }
    base.update(over)
    return base


# ───────────── the result (pure) ─────────────


class TestBuildResult:
    def test_an_anomalous_image(self) -> None:
        r = build_result(_payload(), task="anomaly_detection", backend="onnxruntime", device="cpu")
        assert isinstance(r, AnomalyResult)
        assert (r.label, r.label_confidence, r.score, r.is_anomalous) == (
            "defect",
            0.83,
            0.83,
            True,
        )
        assert r.tags == ["defect"]
        assert len(r.predictions) == 1 and r.predictions[0].name == "defect"
        assert r.predictions[0].confidence == pytest.approx(0.91)
        assert r.anomaly_map is None

    def test_a_normal_image(self) -> None:
        r = build_result(
            _payload(
                predictions=[],
                tags=["normal"],
                labels=[{"class": "normal", "confidence": 0.9}],
                anomaly_score=0.1,
                is_anomalous=False,
            ),
            task="anomaly_detection",
        )
        assert (r.label, r.label_confidence, r.score, r.is_anomalous) == ("normal", 0.9, 0.1, False)
        assert r.predictions == [] and r.tags == ["normal"]

    def test_a_filtered_label_keeps_the_decision(self) -> None:
        """``class_filter`` may drop the label; the decision is still said (is_anomalous)."""
        r = build_result(_payload(predictions=[], tags=[], labels=[]), task="anomaly_detection")
        assert r.label is None and r.label_confidence is None and r.tags == []
        assert r.is_anomalous is True and r.score == pytest.approx(0.83)

    def test_an_answer_without_the_decision_reads_it_from_its_regions(self) -> None:
        payload = _payload()
        del payload["is_anomalous"]
        assert build_result(payload, task="anomaly_detection").is_anomalous is True
        payload.update(
            predictions=[], tags=["normal"], labels=[{"class": "normal", "confidence": 1.0}]
        )
        assert build_result(payload, task="anomaly_detection").is_anomalous is False

    def test_a_region_without_an_id_gets_one(self) -> None:
        region = {k: v for k, v in REGION.items() if k != "id"}
        r = build_result(_payload(predictions=[region]), task="anomaly_detection")
        assert r.predictions[0].id

    def test_the_map_rides_along(self) -> None:
        amap = {
            "width": 2,
            "height": 1,
            "encoding": "float16",
            "data": base64.b64encode(b"\x00<\x00@").decode(),
        }
        r = build_result(_payload(anomaly_map=amap), task="anomaly_detection")
        assert isinstance(r.anomaly_map, AnomalyMap) and r.anomaly_map.encoding == "float16"

    def test_the_task_tables_know_the_sixth_task(self) -> None:
        import pictograph
        from pictograph.inference.models import TASK_MODEL_TYPES, AnomalyModel

        assert TASK_RESULT_TYPES["anomaly_detection"] is AnomalyResult
        assert TASK_MODEL_TYPES["anomaly_detection"] is AnomalyModel
        assert pictograph.AnomalyModel is AnomalyModel and pictograph.AnomalyResult is AnomalyResult
        assert pictograph.AnomalyMap is AnomalyMap


class TestAnomalyMap:
    def test_float16_is_the_raw_map(self) -> None:
        np = pytest.importorskip("numpy")
        raw = np.array([[0.0, 1.5, -2.25], [65504.0, 0.000123, 7.0]], dtype=np.float32)
        data = base64.b64encode(raw.astype("<f2").tobytes()).decode()
        out = AnomalyMap(width=3, height=2, encoding="float16", data=data).to_array()
        assert out.dtype == np.float32 and out.shape == (2, 3)
        assert np.array_equal(out, raw.astype("<f2").astype(np.float32))

    def test_png_is_the_normalized_map(self) -> None:
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("PIL")
        gray = np.array([[0, 128, 255], [64, 32, 1]], dtype=np.uint8)
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        out = AnomalyMap(
            width=3, height=2, encoding="png", data=base64.b64encode(buf.tobytes()).decode()
        ).to_array()
        assert np.array_equal(out, gray.astype(np.float32) / np.float32(255.0))

    def test_a_png_of_the_wrong_size_is_refused(self) -> None:
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("PIL")
        ok, buf = cv2.imencode(".png", np.zeros((2, 3), np.uint8))
        assert ok
        bad = AnomalyMap(
            width=4, height=2, encoding="png", data=base64.b64encode(buf.tobytes()).decode()
        )
        with pytest.raises(ValueError, match="not the declared 4x2"):
            bad.to_array()


# ───────────── the online fetch of the calibration (pure) ─────────────


def _record(**over: Any) -> Any:
    from pictograph.models.model import Model

    base: dict[str, Any] = {
        "id": "33333333-3333-3333-3333-333333333333",
        "organization_id": "org",
        "name": "fixture-anomaly-capsule-patchcore",
        "model_type": "anomaly_detection",
        "architecture": "PatchCore",
        "visibility": "private",
        "status": "ready",
        "class_mapping": {"classes": ["normal", "defect"]},
        "created_at": "2026-11-25T00:00:00Z",
        "updated_at": "2026-11-25T00:00:00Z",
    }
    base.update(over)
    return Model.model_validate(base)


class _Models:
    def __init__(self, config: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.config, self.fail, self.calls = config, fail, 0

    def download_file(self, **kwargs: Any) -> Path:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated download failure")
        out = Path(kwargs["output_path"])
        out.write_text(json.dumps(self.config or {}), encoding="utf-8")
        return out


_FETCHED_META = {"family": "patchcore", "image_threshold": 1.0, "image_min": 0.0, "image_max": 1.0}


class TestFetchAnomalyMeta:
    def test_reads_the_calibration_from_the_models_config(self, tmp_path: Path) -> None:
        from pictograph.inference import _fetch_anomaly_meta

        models = _Models({"_pictograph": {"anomaly": _FETCHED_META}})
        assert _fetch_anomaly_meta(models, _record(), tmp_path) == _FETCHED_META
        assert _fetch_anomaly_meta(models, _record(), tmp_path) == _FETCHED_META
        assert models.calls == 1  # cached beside the weights

    def test_another_task_fetches_nothing(self, tmp_path: Path) -> None:
        from pictograph.inference import _fetch_anomaly_meta

        models = _Models()
        assert _fetch_anomaly_meta(models, _record(model_type="object_detection"), tmp_path) is None
        assert models.calls == 0

    def test_a_model_without_its_calibration_is_refused(self, tmp_path: Path) -> None:
        from pictograph.inference import _fetch_anomaly_meta

        with pytest.raises(ValueError, match=r"_pictograph\.anomaly"):
            _fetch_anomaly_meta(_Models({"_pictograph": {"name": "x"}}), _record(), tmp_path)

    def test_a_config_that_cannot_be_fetched_is_refused_not_guessed(self, tmp_path: Path) -> None:
        from pictograph.inference import _fetch_anomaly_meta

        with pytest.raises(ValueError, match="cannot run without it"):
            _fetch_anomaly_meta(_Models(fail=True), _record(), tmp_path)


# ───────────── every local runtime on one real (tiny) model ─────────────

SIZE = 64
META = {
    "family": "patchcore",
    "backbone": "resnet18",
    "layers": ["layer2", "layer3"],
    "memory_bank_size": 64,
    "num_neighbors": 9,
    "sigma": 4.0,
    "input_size": SIZE,
}


def _scene(seed: int, *, defect: bool) -> Any:
    """A BGR uint8 image: a smooth gradient with gentle noise; a defect is a dark square on it."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:128].astype(np.float32)
    base = np.stack([xx / 128.0, yy / 96.0, (xx + yy) / 224.0], axis=-1)
    img = (0.75 * base + 0.25 * rng.random((96, 128, 3), dtype=np.float32)) * 255.0
    if defect:
        img[30:62, 40:80] = 10.0
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def tiny_patchcore(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A PatchCore whose memory bank holds normal images' patches, calibrated on other normals, and its three
    artifacts: ``model.onnx`` (exported as the pipeline exports), ``model.pth`` and ``model.safetensors`` (the
    module's own state, as the pipeline saves it), and ``config.json``."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("cv2")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import numpy as np

    from pictograph.inference._anomaly import build_anomaly_module
    from pictograph.inference._wrappers.anomaly_wrapper import preprocess_bgr

    def batch(seeds: list[int], defect: bool = False) -> Any:
        return torch.from_numpy(
            np.stack([preprocess_bgr(_scene(s, defect=defect), SIZE, SIZE) for s in seeds])
        )

    torch.manual_seed(0)
    module = build_anomaly_module(META).eval()
    with torch.no_grad():
        patches = module.embed(batch([1, 2, 3, 4]))
        flat = patches.permute(0, 2, 3, 1).reshape(-1, patches.shape[1])
        pick = torch.randperm(flat.shape[0], generator=torch.Generator().manual_seed(0))[:64]
        module.memory_bank.copy_(flat[pick])
        maps, scores = module(batch([11, 12, 13, 14, 15]))
    scores = scores.reshape(-1).numpy().astype(np.float32)
    t = float(scores.max())
    calibration = {
        "image_threshold": t,
        "pixel_threshold": t,
        "image_min": float(scores.min()),
        "image_max": float(scores.max()),
        "pixel_min": float(maps.min()),
        "pixel_max": float(maps.max()),
    }
    out = tmp_path_factory.mktemp("tiny-patchcore")
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            batch([1]),
            str(out / "model.onnx"),
            **extra,
            opset_version=17,
            input_names=["input"],
            output_names=["anomaly_map", "pred_score"],
            dynamic_axes={
                "input": {0: "batch"},
                "anomaly_map": {0: "batch"},
                "pred_score": {0: "batch"},
            },
        )
    state = {k: v.detach().contiguous().cpu() for k, v in module.state_dict().items()}
    torch.save(state, str(out / "model.pth"))
    safetensors_torch.save_file(
        state, str(out / "model.safetensors"), metadata={"pictograph_anomaly_family": "patchcore"}
    )
    config = {
        "_pictograph": {
            "model_type": "anomaly_detection",
            "architecture": "PatchCore",
            "name": "tiny-patchcore",
            "class_names": ["normal", "defect"],
            "input_shape": [SIZE, SIZE],
            "anomaly": {**META, **calibration},
        }
    }
    (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return {"dir": out, "config": config, "module": module}


FORMATS = {"onnx": "model.onnx", "pytorch": "model.pth", "safetensors": "model.safetensors"}


def _load(tiny: dict[str, Any], fmt: str, config: Any = None) -> Any:
    from pictograph import load_model

    return load_model(
        tiny["dir"] / FORMATS[fmt],
        config if config is not None else tiny["dir"] / "config.json",
        task="anomaly_detection",
        device="cpu",
    )


class TestLocalRuntimes:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    def test_each_format_loads_as_an_anomaly_model(
        self, tiny_patchcore: dict[str, Any], fmt: str
    ) -> None:
        from pictograph import AnomalyModel

        model = _load(tiny_patchcore, fmt)
        assert isinstance(model, AnomalyModel)
        assert model.backend == ("onnxruntime" if fmt == "onnx" else "pytorch")
        assert model.classes == ["normal", "defect"] and model.device == "cpu"
        model.close()

    def test_every_format_gives_the_same_answer(self, tiny_patchcore: dict[str, Any]) -> None:
        images = [_scene(s, defect=False) for s in (21, 22)] + [
            _scene(s, defect=True) for s in (31, 32)
        ]
        answers = {}
        for fmt in FORMATS:
            model = _load(tiny_patchcore, fmt)
            answers[fmt] = model.predict_batch(images)
            single = model.predict(images[2])
            assert single.score == pytest.approx(answers[fmt][2].score, abs=1e-6)
            model.close()
        onnx = answers["onnx"]
        # The defect raises the image's score over the normal images' (the model works), and is flagged.
        assert min(r.score for r in onnx[2:]) > max(r.score for r in onnx[:2])
        assert all(r.is_anomalous and r.label == "defect" and r.predictions for r in onnx[2:])
        # The normalized score divides the raw one by the calibration's range, which five normal images of an
        # untrained backbone keep narrow - so ORT's and torch's float noise is compared at 1e-3 here (on the real
        # fixture models it is under float16's resolution: ROADMAP Program N, N10).
        for fmt in ("pytorch", "safetensors"):
            for a, b in zip(onnx, answers[fmt], strict=True):
                assert (a.is_anomalous, a.label, len(a.predictions)) == (
                    b.is_anomalous,
                    b.label,
                    len(b.predictions),
                )
                assert a.score == pytest.approx(b.score, abs=1e-3)
                assert a.label_confidence == pytest.approx(b.label_confidence, abs=1e-3)

    def test_the_decision_is_the_score_over_the_confidence(
        self, tiny_patchcore: dict[str, Any]
    ) -> None:
        model = _load(tiny_patchcore, "onnx")
        image = _scene(41, defect=True)
        score = model.predict(image).score
        for c in (0.05, 0.5, 0.95):
            r = model.predict(image, confidence=c)
            assert r.score == pytest.approx(score, abs=1e-7)
            assert r.is_anomalous == (score > c)
            assert bool(r.predictions) == r.is_anomalous
        model.close()

    @pytest.mark.parametrize("fmt", list(FORMATS))
    def test_the_map_on_request(self, tiny_patchcore: dict[str, Any], fmt: str) -> None:
        torch = pytest.importorskip("torch")
        import numpy as np

        from pictograph.inference._wrappers.anomaly_wrapper import preprocess_bgr

        model = _load(tiny_patchcore, fmt)
        image = _scene(51, defect=True)
        assert model.predict(image).anomaly_map is None
        half = model.predict(image, anomaly_map="float16").anomaly_map
        gray = model.predict(image, anomaly_map="png").anomaly_map
        model.close()
        assert half is not None and gray is not None
        assert (half.width, half.height, half.encoding) == (SIZE, SIZE, "float16")
        assert (gray.width, gray.height, gray.encoding) == (SIZE, SIZE, "png")
        with torch.no_grad():
            raw, _ = tiny_patchcore["module"](
                torch.from_numpy(preprocess_bgr(image, SIZE, SIZE)[None])
            )
        raw = raw[0, 0].numpy()
        assert np.allclose(half.to_array(), raw, rtol=2e-3, atol=2e-3 * float(np.abs(raw).max()))
        normalized = gray.to_array()
        assert (
            normalized.shape == (SIZE, SIZE) and 0.0 <= normalized.min() and normalized.max() <= 1.0
        )

    @pytest.mark.parametrize("fmt", list(FORMATS))
    def test_a_config_without_the_calibration_is_refused(
        self, tiny_patchcore: dict[str, Any], fmt: str
    ) -> None:
        config = json.loads(json.dumps(tiny_patchcore["config"]))
        del config["_pictograph"]["anomaly"]
        with pytest.raises(ValueError, match=r"_pictograph\.anomaly"):
            _load(tiny_patchcore, fmt, config)

    def test_the_task_cannot_be_misdeclared(self, tiny_patchcore: dict[str, Any]) -> None:
        from pictograph import load_model

        with pytest.raises(ValueError, match="is a 'anomaly_detection' model"):
            load_model(
                tiny_patchcore["dir"] / "model.onnx",
                tiny_patchcore["dir"] / "config.json",
                task="classification",
                device="cpu",
            )
