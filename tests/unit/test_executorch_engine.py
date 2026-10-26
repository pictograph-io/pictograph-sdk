"""The ExecuTorch loader: delegate reporting, and what a ``.pte`` cannot do.

The behaviour worth pinning here is the one that surprises callers: **a ``.pte``
cannot be re-targeted at load time.** Its delegate backend was chosen when the
artifact was BUILT, so ``device=`` has nothing to select and a CoreML-lowered
program does not become an XNNPACK one because the caller asked for CPU.

The delegate scan and the device mapping need no ExecuTorch installed - they read
the artifact - so most of this runs anywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pictograph.inference._executorch import (
    _delegates,
    _device_for,
    build_executorch_engine,
)


def _fake_pte(tmp_path: Path, *delegates: str) -> Path:
    """A file carrying delegate ids where the flatbuffer would put them.

    The scan is a bounded byte search for known backend id strings, which is exactly
    how ExecuTorch serializes them, so a file containing those bytes exercises the
    real code path without needing a 40 MB program.
    """
    path = tmp_path / "prog.pte"
    body = b"\x00\x01ET12" + b"".join(d.encode("ascii") + b"\x00" for d in delegates)
    path.write_bytes(body + b"\x00" * 512)
    return path


class TestDelegateReporting:
    """``.providers`` and ``.device`` must say what the program was lowered to."""

    def test_xnnpack_reports_cpu(self, tmp_path: Path) -> None:
        found = _delegates(_fake_pte(tmp_path, "XnnpackBackend"))
        assert found == ["XnnpackBackend"]
        assert _device_for(found) == "cpu"

    def test_coreml_reports_coreml(self, tmp_path: Path) -> None:
        found = _delegates(_fake_pte(tmp_path, "CoreMLBackend"))
        assert _device_for(found) == "coreml"

    def test_a_mixed_program_reports_the_accelerator_not_the_cpu_remainder(
        self, tmp_path: Path
    ) -> None:
        """A CoreML-partitioned graph still runs its unpartitioned ops on CPU.
        Reporting `cpu` there would hide the accelerator that is doing the work."""
        found = _delegates(_fake_pte(tmp_path, "XnnpackBackend", "CoreMLBackend"))
        assert set(found) == {"XnnpackBackend", "CoreMLBackend"}
        assert _device_for(found) == "coreml"

    def test_an_undelegated_program_is_portable_not_an_error(self, tmp_path: Path) -> None:
        """A fully-portable program has no delegate at all; that is normal."""
        assert _delegates(_fake_pte(tmp_path)) == []
        assert _device_for([]) == "cpu"

    def test_an_unreadable_file_degrades_rather_than_raising(self, tmp_path: Path) -> None:
        """The scan is informational - a miss must not stop the model running."""
        assert _delegates(tmp_path / "does-not-exist.pte") == []


class TestNoLoadTimeRetargeting:
    """The single most surprising property of a ``.pte``."""

    def test_device_cpu_refuses_an_accelerated_program(self, tmp_path: Path) -> None:
        """`device='cpu'` is a REPRODUCIBILITY request (CI, parity work). Quietly
        handing back a CoreML program would defeat the reason it was asked for, so it
        raises and says how to actually get CPU: load the portable .pte."""
        program = _fake_pte(tmp_path, "CoreMLBackend")
        with pytest.raises(ValueError, match="portable"):
            build_executorch_engine(
                weights=program,
                model_type="classification",
                architecture="resnet18",
                classes=["a"],
                input_shape=(224, 224),
                confidence=0.5,
                device="cpu",
            )

    def test_the_refusal_names_the_delegate_and_the_device(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            build_executorch_engine(
                weights=_fake_pte(tmp_path, "CoreMLBackend"),
                model_type="classification",
                architecture="resnet18",
                classes=["a"],
                input_shape=(224, 224),
                confidence=0.5,
                device="cpu",
            )
        message = str(excinfo.value)
        assert "CoreMLBackend" in message
        assert "cannot be changed at" in message

    def test_a_matching_device_request_is_honoured_not_refused(self) -> None:
        """`device=` is a CHECK here. Asking for the hardware the program was
        actually lowered to must pass - and `mps` must satisfy a CoreML program,
        since they name the same silicon."""
        from pictograph.inference.runtime import check_artifact_device

        for asked in ("auto", "mps", "coreml"):
            check_artifact_device(asked, "coreml", artifact="'a.pte' which", remedy="-")
        check_artifact_device("cpu", "cpu", artifact="'a.pte' which", remedy="-")

    def test_cuda_is_refused_because_no_cuda_pte_is_published(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cuda"):
            build_executorch_engine(
                weights=_fake_pte(tmp_path, "XnnpackBackend"),
                model_type="classification",
                architecture="resnet18",
                classes=["a"],
                input_shape=(224, 224),
                confidence=0.5,
                device="cuda",
            )


class TestInstallHint:
    def test_it_names_the_exact_command_and_the_torch_constraint(self) -> None:
        """ExecuTorch's prebuilt runtime is compiled against a specific torch ABI; a
        mismatch imports fine and then fails to dlopen with a C++ symbol error, which
        is a genuinely awful way to find out. The hint has to say so."""
        from pictograph.inference import _executorch

        # The extra is named WITH [inference], because that is the whole install:
        # ExecuTorch substitutes only the forward pass and reuses the shared
        # wrappers' pre/postprocess, which import onnxruntime at module scope.
        assert 'pip install "pictograph[inference,executorch]"' in _executorch._INSTALL_HINT
        assert "torch" in _executorch._INSTALL_HINT
        # …and it may not send the reader off to install a third-party package.
        assert "pip install executorch" not in _executorch._INSTALL_HINT


class TestLoadedProgram:
    """The parts that need the real runtime. Full numerical parity across all five
    task families lives in ``test_multiruntime_parity.py``; this pins the loader's
    own reporting contract."""

    def test_a_real_program_reports_executorch_and_its_delegate(self, tmp_path: Path) -> None:
        pytest.importorskip("executorch", reason="needs the [executorch] extra")
        pytest.importorskip("onnxruntime", reason="needs the [inference] extra")
        torch = pytest.importorskip("torch")

        import warnings

        from pictograph import load_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            from executorch.exir import to_edge_transform_and_lower

            class _Tiny(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.pool = torch.nn.AdaptiveAvgPool2d(1)
                    self.fc = torch.nn.Linear(3, 2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.fc(self.pool(x / 255.0).flatten(1))

            module = _Tiny().eval()
            exported = torch.export.export(module, (torch.randn(1, 3, 32, 32),))
            lowered = to_edge_transform_and_lower(
                exported, partitioner=[XnnpackPartitioner()]
            ).to_executorch()

        program = tmp_path / "xnnpack-fp32.pte"
        program.write_bytes(lowered.buffer)

        config = {
            "_pictograph": {
                "model_type": "classification",
                "architecture": "resnet18",
                "class_mapping": {"classes": ["a", "b"]},
                "input_shape": [32, 32],
            }
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(program, config, task="classification")
        try:
            assert model.backend == "executorch"
            assert model.providers == ["XnnpackBackend"]
            assert model.device == "cpu"
            assert model.classes == ["a", "b"]
        finally:
            model.close()

    def test_closing_releases_the_program_and_refuses_further_predicts(
        self, tmp_path: Path
    ) -> None:
        """An ExecuTorch program memory-plans at load and holds that arena; `close()`
        has to actually drop it, and a closed model must say so rather than crash."""
        pytest.importorskip("executorch", reason="needs the [executorch] extra")
        pytest.importorskip("onnxruntime", reason="needs the [inference] extra")
        torch = pytest.importorskip("torch")

        import warnings

        import numpy as np

        from pictograph import load_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            from executorch.exir import to_edge_transform_and_lower

            class _Tiny(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.pool = torch.nn.AdaptiveAvgPool2d(1)
                    self.fc = torch.nn.Linear(3, 2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.fc(self.pool(x / 255.0).flatten(1))

            exported = torch.export.export(_Tiny().eval(), (torch.randn(1, 3, 32, 32),))
            lowered = to_edge_transform_and_lower(
                exported, partitioner=[XnnpackPartitioner()]
            ).to_executorch()

        program = tmp_path / "xnnpack-fp32.pte"
        program.write_bytes(lowered.buffer)
        config = {
            "_pictograph": {
                "model_type": "classification",
                "architecture": "resnet18",
                "class_mapping": {"classes": ["a", "b"]},
                "input_shape": [32, 32],
            }
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(program, config, task="classification")
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            assert model.predict(image).top.name in {"a", "b"}
            model.close()
            with pytest.raises(RuntimeError, match="closed"):
                model.predict(image)


def test_logging_is_scoped_to_the_sdk_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Everything this module says goes to `pictograph.inference`, so an application
    can silence or route it as one unit."""
    from pictograph.inference import _executorch

    with caplog.at_level(logging.DEBUG, logger="pictograph.inference"):
        _executorch._delegates(Path("/nonexistent/definitely-not-here.pte"))
    assert all(r.name == "pictograph.inference" for r in caplog.records)
