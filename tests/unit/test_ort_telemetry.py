"""ONNX Runtime's telemetry to Microsoft stays off for everything the SDK runs (B645).

ONNX Runtime 1.30+ reports by default: importing it writes a device id and an event store on the user's machine and
queues events for ``mobile.events.data.microsoft.com``. The switch (``ORT_DISABLE_TELEMETRY``) is read only when the
runtime is IMPORTED, so the property that matters is ORDER: it must be set before any SDK module imports the runtime.
Each process-level check therefore runs in a FRESH interpreter - in this one the runtime may already be imported, and an
in-process check would pass no matter what.

The last test runs a real model through ``load_model(...).predict()`` on the real runtime, with HOME / TMPDIR /
XDG_CACHE_HOME pointed into a temp dir, and proves first, with a control run, that this runtime DOES write its store when
it is imported ahead of Pictograph - so "nothing written" means switched off, never a test that cannot see the store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from pictograph.inference import _ort_telemetry

if TYPE_CHECKING:
    from pathlib import Path

ENV = _ort_telemetry.ENV

# What the runtime writes on its first import (measured on 1.31.0): the event store and its device id
# (~/.cache/... on Linux, ~/Library/Application Support/... on macOS) and the session file ($TMPDIR/.ses).
TELEMETRY_FILES = frozenset({".onnxruntime", "onnxruntime.db", "deviceid", ".ses"})


def _fresh(code: str, *, env: dict[str, str] | None = None) -> str:
    """Run ``code`` in a NEW interpreter with ``ORT_DISABLE_TELEMETRY`` removed from its environment."""
    base = {k: v for k, v in os.environ.items() if k != ENV}
    base.update(env or {})
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=base,
        check=False,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


# --- the switch's own rules ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", "1", "true", "yes", "TRUE", "2", "on"])
def test_keep_off_sets_the_switch_unless_the_user_opted_in(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Unset, blank and any value that is not an explicit opt-in end as "1" - "2" included, which the runtime itself
    reads as ON."""
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    _ort_telemetry.keep_off()
    assert os.environ[ENV] == "1"


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off", "Off"])
def test_an_explicit_opt_in_is_left_alone(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    assert _ort_telemetry.opted_in()
    _ort_telemetry.keep_off()
    assert os.environ[ENV] == value


class _FakeRuntime:
    def __init__(self, *, fails: bool = False) -> None:
        self.calls = 0
        self.fails = fails

    def disable_telemetry_events(self) -> None:
        self.calls += 1
        if self.fails:
            raise RuntimeError("an odd build")


@pytest.fixture
def events_not_yet_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ort_telemetry, "_events_disabled", False)
    monkeypatch.delenv(ENV, raising=False)


@pytest.mark.usefixtures("events_not_yet_disabled")
def test_disable_events_calls_the_runtimes_switch_once_per_process() -> None:
    ort = _FakeRuntime()
    _ort_telemetry.disable_events(ort)
    _ort_telemetry.disable_events(ort)
    assert ort.calls == 1


@pytest.mark.usefixtures("events_not_yet_disabled")
def test_disable_events_does_nothing_for_a_user_who_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "0")
    ort = _FakeRuntime()
    _ort_telemetry.disable_events(ort)
    assert ort.calls == 0


@pytest.mark.usefixtures("events_not_yet_disabled")
def test_disable_events_never_breaks_inference() -> None:
    """A build whose switch raises, or that has none, is tolerated - and a failed call is retried next time."""
    failing = _FakeRuntime(fails=True)
    _ort_telemetry.disable_events(failing)
    _ort_telemetry.disable_events(failing)
    assert failing.calls == 2
    _ort_telemetry.disable_events(object())  # no disable_telemetry_events at all


# --- order: the switch is set before the runtime is imported ---------------------------------------------------------

_FAKE_RUNTIME = f"""
import os

SEEN_AT_IMPORT = os.environ.get({ENV!r})
CALLS = 0


def disable_telemetry_events():
    global CALLS
    CALLS += 1


def get_available_providers():
    return ["CPUExecutionProvider"]


class _Meta(type):
    def __getattr__(cls, name):
        return cls


class _Any(metaclass=_Meta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


def __getattr__(name):  # SessionOptions, InferenceSession, ...: the wrappers name them at class definition
    return _Any
"""


def _fake_runtime(tmp_path: Path) -> dict[str, str]:
    """A stand-in ``onnxruntime`` on PYTHONPATH that records the switch at the moment it is imported, and counts calls to
    its off switch."""
    pkg = tmp_path / "fake-ort" / "onnxruntime"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_FAKE_RUNTIME)
    path = os.pathsep.join([str(tmp_path / "fake-ort"), os.environ.get("PYTHONPATH", "")])
    return {"PYTHONPATH": path.rstrip(os.pathsep)}


def test_the_fake_runtime_double_records_what_it_saw(tmp_path: Path) -> None:
    """Anti-vacuity for the order tests: imported with no switch set, the double reports None and no call - so "1"
    below means the SDK set it, not a double that always says "1"."""
    out = _fresh(
        "import onnxruntime; print(onnxruntime.SEEN_AT_IMPORT, onnxruntime.CALLS)",
        env=_fake_runtime(tmp_path),
    )
    assert out == "None 0"


def test_importing_pictograph_sets_the_switch_and_loads_no_runtime() -> None:
    out = _fresh(
        f"import os, sys, pictograph; print(os.environ.get({ENV!r}), 'onnxruntime' in sys.modules)"
    )
    assert out == "1 False"


def test_every_inference_module_that_imports_the_runtime_sees_the_switch_already_set(
    tmp_path: Path,
) -> None:
    """A vendored wrapper imported FIRST (it imports the runtime at its top; Python runs the package's init before it),
    then every other module of pictograph.inference but the switch module itself. The runtime's first import, wherever
    it came from, must have found the switch set."""
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    out = _fresh(
        """
        import importlib, pkgutil, sys
        import pictograph.inference._wrappers.classifier_wrapper
        import pictograph.inference as inference
        for info in pkgutil.walk_packages(inference.__path__, prefix="pictograph.inference."):
            if info.name == "pictograph.inference._ort_telemetry":
                continue
            try:
                importlib.import_module(info.name)
            except ImportError:
                pass  # an optional dependency (torch, executorch, tensorrt) this env lacks
        ort = sys.modules.get("onnxruntime")
        print(ort is not None, getattr(ort, "SEEN_AT_IMPORT", None))
        """,
        env=_fake_runtime(tmp_path),
    )
    assert out == "True 1", (
        f"the runtime was imported without the switch set (or never imported): {out}"
    )


@pytest.mark.parametrize(
    "call",
    [
        "from pictograph.inference.runtime import session_options; session_options()",
        "from pictograph.inference.runtime import _available; _available()",
        # dispatch refuses the made-up type AFTER the builder reached the runtime: the call is what is under test
        "from pictograph.inference._engine import build_wrapper_with_session\n"
        "try:\n"
        "    build_wrapper_with_session(object(), model_type='not-a-type', architecture='x')\n"
        "except Exception:\n"
        "    pass",
    ],
    ids=["session_options", "available_providers", "build_wrapper_with_session"],
)
def test_each_sdk_entry_to_the_runtime_calls_its_off_switch(tmp_path: Path, call: str) -> None:
    """The runtime's own off switch (Windows reports through it; a runtime imported before Pictograph needs it) is
    called by every SDK path that reaches the runtime."""
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    out = _fresh(
        f"{call}\nimport onnxruntime\nprint(onnxruntime.CALLS)", env=_fake_runtime(tmp_path)
    )
    assert out == "1"


def test_the_sdk_way_in_sets_the_switch_first_and_calls_the_off_switch_once(tmp_path: Path) -> None:
    out = _fresh(
        """
        from pictograph.inference._ort_telemetry import import_onnxruntime
        ort = import_onnxruntime()
        import_onnxruntime()
        print(ort.SEEN_AT_IMPORT, ort.CALLS)
        """,
        env=_fake_runtime(tmp_path),
    )
    assert out == "1 1"


def test_a_user_opt_in_reaches_the_runtime_untouched(tmp_path: Path) -> None:
    env = _fake_runtime(tmp_path)
    env[ENV] = "0"
    out = _fresh(
        """
        import pictograph
        from pictograph.inference._ort_telemetry import import_onnxruntime
        ort = import_onnxruntime()
        print(ort.SEEN_AT_IMPORT, ort.CALLS)
        """,
        env=env,
    )
    assert out == "0 0"


# --- the real runtime, through load_model().predict() ----------------------------------------------------------------


def _tiny_classifier(root: Path) -> tuple[Path, Path, Path]:
    """A real two-class classifier graph (mean-pool, then a 3x2 linear), its config.json and an image."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from PIL import Image

    weights = numpy_helper.from_array(np.array([[1.0, -1.0]] * 3, dtype=np.float32), "W")
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("MatMul", ["flat", "W"], ["logits"]),
        ],
        "tiny_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        initializer=[weights],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="b645"
    )
    model.ir_version = 8
    root.mkdir(parents=True)
    graph_path = root / "model.onnx"
    onnx.save(model, str(graph_path))
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "classification",
                "architecture": "resnet18",
                "class_mapping": {"classes": ["bright", "dark"]},
                "input_shape": [32, 32],
            }
        )
    )
    image = root / "photo.png"
    Image.new("RGB", (32, 32), (250, 250, 250)).save(image)
    return graph_path, config, image


def _sandbox(root: Path) -> dict[str, str]:
    """Every place the runtime could write, inside ``root``."""
    env: dict[str, str] = {}
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "LOCALAPPDATA", "APPDATA", "USERPROFILE"):
        path = root / name.lower()
        path.mkdir(parents=True)
        env[name] = str(path)
    return env


def _telemetry_written(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.name in TELEMETRY_FILES)


_PREDICT = """
{first}
from pictograph import load_model

result = load_model({weights!r}, {config!r}, task="classification", device="cpu").predict({image!r})
import onnxruntime
print(result.top.name, onnxruntime.__version__)
"""


def test_a_real_model_run_through_the_sdk_writes_no_telemetry(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")
    weights, config, image = _tiny_classifier(tmp_path / "model")
    paths = {"weights": str(weights), "config": str(config), "image": str(image)}

    # Control: the runtime imported BEFORE pictograph (the switch comes too late) must write its store here - or this
    # runtime carries no telemetry and the assertion below could not fail.
    control = tmp_path / "control"
    top, version = _fresh(
        _PREDICT.format(first="import onnxruntime", **paths), env=_sandbox(control)
    ).split()
    assert top == "bright", "the tiny model did not run"
    if not _telemetry_written(control):
        pytest.skip(f"onnxruntime {version} writes no telemetry store here: nothing to switch off")

    # As a user runs it: pictograph first, the runtime only through it.
    sdk = tmp_path / "sdk"
    top, _ = _fresh(_PREDICT.format(first="", **paths), env=_sandbox(sdk)).split()
    assert top == "bright"
    assert _telemetry_written(sdk) == [], (
        f"onnxruntime {version} wrote telemetry: {_telemetry_written(sdk)}"
    )
