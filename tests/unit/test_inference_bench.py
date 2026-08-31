"""Unit tests for the local-inference benchmark harness's pure logic (B-inference-bench).

The harness lives under ``benchmarks/`` (operator tooling, not shipped in the
wheel), so it's loaded by file path - the same convention
``test_load_bench.py`` uses. Only network/hardware-free logic is exercised
here: class-count inference from fabricated ONNX output shapes, the
runtime-config <-> provider-list mapping, and the report's skip/error
visibility. Building a real ONNX/torch session is covered by actually running
the harness (operator-run, see benchmarks/README.md), not by this suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HARNESS = Path(__file__).resolve().parents[2] / "benchmarks" / "inference_bench.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inference_bench", _HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestInferNumClasses:
    def test_classification_is_the_raw_output_width(self) -> None:
        bench = _load()
        probe = {"output_names": ["output"], "output_shapes": [(1, 82)]}
        assert bench._infer_num_classes("classification", probe) == 82

    def test_semantic_segmentation_subtracts_the_background_channel(self) -> None:
        bench = _load()
        probe = {"output_names": ["output"], "output_shapes": [(1, 81, 512, 512)]}
        assert bench._infer_num_classes("semantic_segmentation", probe) == 80

    def test_semantic_segmentation_single_class_has_no_background_channel(self) -> None:
        bench = _load()
        probe = {"output_names": ["output"], "output_shapes": [(1, 1, 512, 512)]}
        assert bench._infer_num_classes("semantic_segmentation", probe) == 1

    def test_single_output_detection_head_subtracts_box_plus_objectness(self) -> None:
        bench = _load()
        # YOLOX-style: (batch, anchors, 4 box + 1 obj + classes)
        probe = {"output_names": ["output"], "output_shapes": [(1, 8400, 85)]}
        assert bench._infer_num_classes("object_detection", probe) == 80

    def test_multi_output_detection_head_subtracts_the_background_slot(self) -> None:
        bench = _load()
        # RF-DETR-style: separate box + label logits outputs.
        probe = {
            "output_names": ["dets", "labels"],
            "output_shapes": [(1, 300, 4), (1, 300, 81)],
        }
        assert bench._infer_num_classes("object_detection", probe) == 80

    def test_unrecognized_shape_raises_rather_than_guessing(self) -> None:
        bench = _load()
        probe = {"output_names": ["mystery"], "output_shapes": [(1, 4)]}
        import pytest

        with pytest.raises(ValueError, match="Could not infer a class count"):
            bench._infer_num_classes("object_detection", probe)


class TestConfigNameForProviders:
    def test_bare_cpu(self) -> None:
        bench = _load()
        assert bench._config_name_for_providers(["CPUExecutionProvider"]) == "onnx-cpu"

    def test_coreml_neuralnetwork(self) -> None:
        bench = _load()
        providers = [
            ("CoreMLExecutionProvider", {"ModelFormat": "NeuralNetwork"}),
            "CPUExecutionProvider",
        ]
        assert bench._config_name_for_providers(providers) == "onnx-coreml-neuralnetwork"

    def test_coreml_mlprogram(self) -> None:
        bench = _load()
        providers = [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
            "CPUExecutionProvider",
        ]
        assert bench._config_name_for_providers(providers) == "onnx-coreml-mlprogram"

    def test_cuda(self) -> None:
        bench = _load()
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        assert bench._config_name_for_providers(providers) == "onnx-cuda"

    def test_tensorrt(self) -> None:
        bench = _load()
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        assert bench._config_name_for_providers(providers) == "onnx-tensorrt"


class TestReportVisibility:
    def test_skip_row_is_visually_distinct_from_an_ok_row(self) -> None:
        bench = _load()
        rows = [
            bench.BenchRow(
                model="classifier", config="onnx-cpu", status="ok", device="cpu", p50_ms=1.2
            ),
            bench._skip_row("classifier", "onnx-cuda", "CUDAExecutionProvider not available"),
        ]
        table = bench.format_table(rows)
        assert "SKIP" in table
        assert "CUDAExecutionProvider not available" in table
        # The skip reason must be printed even though no numeric columns apply.
        skip_line_idx = next(i for i, line in enumerate(table.splitlines()) if "SKIP" in line)
        assert "->" in table.splitlines()[skip_line_idx + 1]

    def test_error_row_carries_its_reason_too(self) -> None:
        bench = _load()
        rows = [bench._error_row("yolox", "torch-cpu", "RuntimeError: boom")]
        table = bench.format_table(rows)
        assert "ERROR" in table
        assert "RuntimeError: boom" in table


class TestRegistry:
    def test_model_labels_are_unique(self) -> None:
        bench = _load()
        labels = [s.label for s in bench.MODEL_SPECS]
        assert len(labels) == len(set(labels))

    def test_runtime_config_names_are_unique(self) -> None:
        bench = _load()
        names = [c.name for c in bench.RUNTIME_CONFIGS]
        assert len(names) == len(set(names))

    def test_required_runtime_configs_are_present(self) -> None:
        bench = _load()
        names = {c.name for c in bench.RUNTIME_CONFIGS}
        assert names == {
            "onnx-cpu",
            "onnx-coreml-neuralnetwork",
            "onnx-coreml-mlprogram",
            "onnx-cuda",
            "onnx-tensorrt",
            "torch-cpu",
            "torch-mps",
            "torch-cuda",
        }

    def test_rf_detr_has_no_pth_on_record(self) -> None:
        """rf-detr is documented as ONNX-only on disk; torch rows must SKIP, not crash."""
        bench = _load()
        rfdetr = next(s for s in bench.MODEL_SPECS if s.label == "rf-detr")
        assert rfdetr.pth_stem is None


class TestParseArgs:
    def test_quick_flag_is_visible_and_defaults_are_none(self) -> None:
        bench = _load()
        args = bench.parse_args([])
        assert args.quick is False
        assert args.warmup is None
        assert args.iters is None

    def test_rejects_unknown_model_label_in_main(self) -> None:
        bench = _load()
        rc = bench.main(["--only-models", "not-a-real-model"])
        assert rc == 2

    def test_rejects_unknown_config_name_in_main(self) -> None:
        bench = _load()
        rc = bench.main(["--only-configs", "not-a-real-config"])
        assert rc == 2
