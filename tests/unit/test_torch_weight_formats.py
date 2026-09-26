"""The family x weight-FORMAT matrix for the native-PyTorch engine.

``tests/unit/test_torch_safetensors_fallback.py`` pins the DOWNLOAD side of the
story - which artifact ``format=`` asks for, and that the other one is never
substituted for it. This module pins the other half: once the bytes are in hand,
does every family actually rebuild from BOTH containers?

That is not one question, it is ten, because the container shape and the family
builder are independent axes:

* ``safetensors.torch.load_file`` ALWAYS returns a bare ``{name: tensor}``
  mapping - there is nowhere in the format to nest anything.
* a ``.pth`` is a pickled object the pipeline chose the shape of, and the
  pipelines disagree: YOLOX and RF-DETR nest the weights under ``"model"``
  alongside optimizer/args state, while the classification and semantic-seg
  pipelines save ``module.state_dict()`` directly, so their ``.pth`` IS bare.

A builder that reads only one of those shapes silently works for whichever
format its own pipeline happens to publish today and breaks the moment the other
one arrives. Both directions are real: ``rfdetr_keypoint`` publishes ONLY
safetensors (no ``.pth`` is ever written), and every other pipeline publishes
only a ``.pth`` - so for any given family, one half of this matrix is the half
that has never run in production.

Everything here is pure-python: the frameworks
(``segmentation_models_pytorch``, ``torch``) are faked through
``_torch._require`` / ``_torch._require_torch``, and the VENDORED YOLOX builder
(``pictograph.inference._yolox.build_yolox`` - no longer an optional import at
all) is patched directly, so the whole matrix runs on the base CI gate with no
``[inference]`` extra, no GPU, no real weights and no network. The one test that
needs real ``torch`` says so with ``importorskip``.

Measured against the real published weights for all five families (see the
2026-07-30 verification), each family loads byte-identical parameters from both
containers and predicts identically; these tests are the CI-runnable pin of
that result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pictograph.exceptions import ConflictError
from pictograph.inference import _torch, _yolox
from pictograph.inference._torch import (
    _bare_state_dict,
    _cache_stem,
    _classifier_hidden_units,
    _kp_counts_from_active_mask,
    _resolve_family,
    _state_for_load,
)


class _T:
    """Minimal stand-in for a torch tensor - shape, plus a summable row count."""

    def __init__(self, shape: tuple[int, ...], rows: list[int] | None = None) -> None:
        self.shape = shape
        self._rows = rows

    def sum(self, dim: int) -> _T:  # noqa: ARG002 - mirrors torch's kwarg
        return _T((self.shape[0],), rows=self._rows)

    def tolist(self) -> list[int]:
        return list(self._rows or [])


def _bare(**tensors: _T) -> dict[str, Any]:
    """The shape ``safetensors.torch.load_file`` always returns."""
    return dict(tensors)


def _wrapped(state: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The shape a ``.pth`` from YOLOX / RF-DETR arrives in."""
    return {"model": state, "optimizer": {"lr": 0.01}, "start_epoch": 7, **extra}


# One representative state dict per family, keyed the way that family's real
# checkpoint is keyed (verified against the published weights).
_STATES: dict[str, dict[str, Any]] = {
    "yolox": _bare(
        **{
            "backbone.backbone.stem.conv.conv.weight": _T((32, 12, 3, 3)),
            "head.cls_preds.0.weight": _T((80, 128, 1, 1)),
        }
    ),
    "smp": _bare(
        **{
            "encoder.conv1.weight": _T((64, 3, 7, 7)),
            "segmentation_head.0.weight": _T((81, 16, 3, 3)),
        }
    ),
    "torchvision": _bare(
        **{
            "conv1.weight": _T((64, 3, 7, 7)),
            "fc.1.weight": _T((256, 512)),
            "fc.4.weight": _T((82, 256)),
        }
    ),
    "rfdetr": _bare(
        **{
            "class_embed.weight": _T((81, 256)),
            "_kp_active_mask": _T((6, 1), rows=[1, 1, 1, 1, 1, 1]),
        }
    ),
}


# ───────────────────── the container-shape axis, per builder ─────────────────


class _RecordingModel:
    """Records what a builder strict-loads into it."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None
        self.strict: bool | None = None
        self.evaled = False
        self.head = type("_Head", (), {"decode_in_inference": False})()

    def load_state_dict(self, state: dict[str, Any], strict: bool = False) -> None:
        self.loaded = state
        self.strict = strict

    def eval(self) -> None:
        self.evaled = True


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> _RecordingModel:
    """Fake every optional framework so the builders run with no deps installed."""
    model = _RecordingModel()

    # YOLOX is no longer reached through `_require` - the architecture is
    # VENDORED into the wheel (`pictograph.inference._yolox`), so there is no
    # optional import to fake. Patch the vendored builder itself; what this
    # module is testing is the CONTAINER normalisation around it, not the
    # module construction (`test_yolox_vendored.py` owns that, against the real
    # upstream source).
    monkeypatch.setattr(_yolox, "build_yolox", lambda *_a, **_k: model)

    smp = type(
        "_Smp",
        (),
        {name: staticmethod(lambda **_k: model) for name in ("Unet", "UnetPlusPlus", "Segformer")},
    )
    fakes = {"segmentation_models_pytorch": smp}

    def fake_require(module: str, hint: str) -> Any:
        if module in fakes:
            return fakes[module]
        raise AssertionError(f"builder reached for an unexpected module: {module}")

    monkeypatch.setattr(_torch, "_require", fake_require)
    return model


class TestBuildersAcceptBothContainers:
    """Each builder must reach the SAME bare mapping from either container.

    ``_build_yolox`` always did (it normalises through ``_bare_state_dict``).
    ``_build_smp`` and ``_build_torchvision`` used to hand ``ckpt`` straight to
    ``load_state_dict``, which is only correct for the bare shape.
    """

    def test_yolox_from_a_pth_container(self, built: _RecordingModel) -> None:
        state = _STATES["yolox"]
        _torch._build_yolox(_wrapped(state), {"model_size": "s"}, "YOLOX-S", 80)
        assert built.loaded is state
        assert built.strict is True
        assert built.evaled is True
        assert built.head.decode_in_inference is True

    def test_yolox_from_a_bare_safetensors_mapping(self, built: _RecordingModel) -> None:
        state = _STATES["yolox"]
        _torch._build_yolox(state, {"model_size": "s"}, "YOLOX-S", 80)
        assert built.loaded is state
        assert built.strict is True

    def test_smp_from_a_bare_safetensors_mapping(self, built: _RecordingModel) -> None:
        """The shape smp's own pipeline publishes, and what safetensors returns."""
        state = _STATES["smp"]
        _torch._build_smp(state, {"architecture": "unetplusplus", "encoder": "resnet34"}, "", ["a"])
        assert built.loaded is state
        assert built.strict is True

    def test_smp_from_a_pth_container(self, built: _RecordingModel) -> None:
        """A wrapped checkpoint must not be handed to ``load_state_dict`` whole -
        every key would be ``model.…`` and the strict load would fail."""
        state = _STATES["smp"]
        _torch._build_smp(
            _wrapped(state), {"architecture": "unetplusplus", "encoder": "resnet34"}, "", ["a"]
        )
        assert built.loaded is state

    def test_an_unrecognised_container_is_handed_through_untouched(
        self, built: _RecordingModel
    ) -> None:
        """Normalising a container must not become a validation step: anything
        that is not identifiably a tensor mapping still reaches
        ``load_state_dict``, which raises its own precise error naming the keys
        that are actually missing."""
        odd = {"args": {}, "epoch": 3}
        _torch._build_smp(odd, {"architecture": "unet"}, "", ["a"])
        assert built.loaded is odd


def test_state_for_load_normalises_both_containers() -> None:
    bare = _STATES["smp"]
    assert _state_for_load(bare) is bare
    assert _state_for_load(_wrapped(bare)) is bare


def test_state_for_load_passes_an_unrecognised_container_through() -> None:
    odd = {"epoch": 1}
    assert _state_for_load(odd) is odd


class TestTorchvisionBuilder:
    """``_build_torchvision`` needs real ``torch.nn`` to assemble its head, so
    the end-to-end build is gated; the container normalisation it depends on is
    covered dependency-free above."""

    @pytest.mark.parametrize("wrap", [False, True], ids=["safetensors-bare", "pth-container"])
    def test_both_containers_build_the_same_head_and_load(self, wrap: bool) -> None:
        torch = pytest.importorskip("torch")
        tv_models = pytest.importorskip("torchvision.models")
        import torch.nn as nn

        # A real trained-shape checkpoint: the classification pipeline's head is
        # Dropout, Linear(in, hidden), ReLU, Dropout, Linear(hidden, n).
        reference = tv_models.resnet18(weights=None)
        reference.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 82),
        )
        state = reference.state_dict()
        ckpt = _wrapped(dict(state)) if wrap else state

        model = _torch._build_torchvision(ckpt, {"backbone": "resnet18"}, "resnet18", 82)

        # The head width came from the CHECKPOINT (fc.1.weight is (256, 512)),
        # not from the config default - from either container.
        assert model.fc[1].out_features == 256
        assert model.fc[4].out_features == 82
        # …and the tensors actually LANDED, which is the thing a wrapped
        # container silently got wrong.
        assert torch.equal(model.fc[4].weight, reference.fc[4].weight)
        assert torch.equal(model.conv1.weight, reference.conv1.weight)

    def test_the_head_width_probe_reads_through_both_containers(self) -> None:
        """``_classifier_hidden_units`` is what makes "artifact beats config"
        work; reading it off the wrong depth silently falls back to the config's
        stale 256 and fails the strict load."""
        state = _STATES["torchvision"]
        assert _classifier_hidden_units(state, "fc") == 256
        assert _classifier_hidden_units(_bare_state_dict(_wrapped(state)), "fc") == 256


# ───────────────────── the keypoint-arity axis ─────────────────


class TestKeypointArityFromEitherContainer:
    """``_kp_active_mask`` is the ONLY source of per-class arity for RF-DETR.

    Measured on the published RF-DETR checkpoints: ``args`` carries no
    ``num_keypoints_per_class`` at all, so this tensor route is not a fallback,
    it is the route. Reading it at the wrong depth returns ``None`` and the
    engine reports ``num_keypoints_per_class == []`` - joints then come back as
    ``point_0..point_N`` with no skeleton, from a model that loaded perfectly.
    """

    _MASK = _T((6, 1), rows=[1, 1, 1, 1, 1, 1])

    def test_bare_safetensors_mapping(self) -> None:
        assert _kp_counts_from_active_mask({"_kp_active_mask": self._MASK}, 6) == [1] * 6

    def test_pth_container_nests_the_mask_under_model(self) -> None:
        ckpt = _wrapped({"_kp_active_mask": self._MASK, "class_embed.weight": _T((7, 256))})
        assert _kp_counts_from_active_mask(ckpt, 6) == [1] * 6

    def test_both_containers_agree(self) -> None:
        mask = _T((3, 4), rows=[2, 3, 1])
        bare = {"_kp_active_mask": mask}
        assert _kp_counts_from_active_mask(bare, 3) == _kp_counts_from_active_mask(
            _wrapped(bare), 3
        )

    def test_args_still_win_over_the_tensor_when_present(self) -> None:
        ckpt = _wrapped(
            {"_kp_active_mask": self._MASK}, args={"num_keypoints_per_class": [4, 4, 4, 4, 4, 4]}
        )
        assert _torch._rfdetr_ckpt_num_keypoints_per_class(ckpt, 6) == [4] * 6

    def test_a_detection_checkpoints_empty_mask_is_rejected_not_mis_read(self) -> None:
        """A non-keypoint RF-DETR ships a degenerate ``(0, 0)`` mask; it must not
        be read as an arity schema."""
        ckpt = _wrapped({"_kp_active_mask": _T((0, 0), rows=[])})
        assert _kp_counts_from_active_mask(ckpt, 80) is None


# ───────────────────── the family-resolution axis ─────────────────


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected"),
    [
        ("classification", "resnet18", "torchvision"),
        ("semantic_segmentation", "UnetPlusPlus", "segmentation_models_pytorch"),
        ("object_detection", "YOLOX-S", "yolox"),
        ("object_detection", "RF-DETR Medium", "rfdetr"),
        ("instance_segmentation", "RF-DETR Seg Medium", "rfdetr"),
        ("keypoint_detection", "RF-DETR Keypoint", "rfdetr"),
    ],
)
def test_family_resolution_is_container_independent(
    model_type: str, architecture: str, expected: str
) -> None:
    """The same model must resolve to the same builder whichever container it
    arrived in - otherwise a format change silently reroutes it to another
    family's rebuild recipe."""
    bare = _STATES["rfdetr"] if "RF-DETR" in architecture else _STATES["yolox"]
    assert _resolve_family(model_type, architecture, bare) == expected
    assert _resolve_family(model_type, architecture, _wrapped(bare)) == expected


# ───────────────────── the download-order axis, per family ─────────────────


class _StrictModels:
    """A ``client.models`` stand-in whose signatures MIRROR the real resource.

    Deliberately keyword-strict and ``name``-first, exactly like
    :meth:`pictograph.resources.models.Models.download_file`: a positional call
    would bind the model id to ``name`` and omit the required ``file_name``, and
    ``filename=`` would be an unexpected kwarg. Both are silent-breakage shapes
    this pins against.
    """

    def __init__(self, *, has_pth: bool, has_safetensors: bool = True) -> None:
        self.has_pth = has_pth
        self.has_safetensors = has_safetensors
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def download(
        self,
        name: str | None = None,
        *,
        output_path: Path,
        model_id: str | None = None,
        format: str = "onnx",
    ) -> Path:
        assert name is None, "the loader must address the model by model_id=, not positionally"
        self.calls.append(
            ("download", {"model_id": model_id, "format": format, "output_path": output_path})
        )
        # BOTH native containers are now fetched through the `format=`
        # route. `download_file(file_name="model.safetensors")` was only ever
        # possible because every model published under that one literal name,
        # which is the same fact that made it a cache-collision hazard.
        if format == "safetensors":
            if not self.has_safetensors:
                raise ConflictError("this model publishes no safetensors", status_code=409)
            Path(output_path).write_bytes(b"safetensors")
            return Path(output_path)
        if not self.has_pth:
            raise ConflictError("this model publishes no .pth", status_code=409)
        Path(output_path).write_bytes(b"pth")
        return Path(output_path)

    def download_file(
        self,
        name: str | None = None,
        *,
        model_id: str | None = None,
        file_name: str,
        version: str | int | None = None,  # noqa: ARG002 - mirrors the real signature
        output_path: Path,
    ) -> Path:
        assert name is None, "the loader must address the model by model_id=, not positionally"
        self.calls.append(
            (
                "download_file",
                {"model_id": model_id, "file_name": file_name, "output_path": output_path},
            )
        )
        if not self.has_safetensors:
            raise ConflictError("this model publishes no safetensors", status_code=409)
        Path(output_path).write_bytes(b"safetensors")
        return Path(output_path)


class _Rec:
    """The subset of ``Model`` ``_fetch_native_weights`` reads."""

    def __init__(self, model_type: str) -> None:
        self.id = f"{model_type}-id"
        self.name = f"{model_type} fixture"
        self.status = "ready"
        self.model_type = model_type
        self.architecture = ""
        self.current_version_id = "v1"
        self.class_mapping = {"classes": ["a"]}
        self.training_config: dict[str, Any] = {}


@pytest.fixture
def loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the two on-disk loaders so no real torch/safetensors is needed."""
    monkeypatch.setattr(_torch, "_load_checkpoint", lambda _p: {"__from__": "pth"})
    monkeypatch.setattr(_torch, "_load_safetensors", lambda _p: {"__from__": "safetensors"})


_FAMILIES = [
    "classification",
    "semantic_segmentation",
    "object_detection",
    "instance_segmentation",
    "keypoint_detection",
]


@pytest.mark.usefixtures("loaders")
@pytest.mark.parametrize("model_type", _FAMILIES)
def test_every_family_fetches_safetensors_with_the_exact_kwargs(
    model_type: str, tmp_path: Path
) -> None:
    """The GATED artifact, for every family.

    Safetensors ships only after a publish-blocking parity gate against that
    version's ONNX; the ``.pth`` has never been gated, and `rfdetr_detection`
    shipped one its ONNX was never exported from. This is also not keypoint-only
    plumbing - the kwargs are load-bearing for every family: ``format=`` and the
    REQUIRED ``output_path=``. A later change moved this off ``download_file(file_name=
    "model.safetensors")``: artifacts are now named after their model, so the
    literal name no longer exists and only the format route resolves.
    """
    model = _Rec(model_type)
    models = _StrictModels(has_pth=True)

    path, ckpt = _torch._fetch_native_weights(
        model, models=models, cache_dir=tmp_path, weight_format="safetensors"
    )

    assert [c[0] for c in models.calls] == ["download"]
    assert models.calls[0][1] == {
        "model_id": model.id,
        "format": "safetensors",
        "output_path": tmp_path / f"{_cache_stem(model)}.safetensors",
    }
    assert path == tmp_path / f"{_cache_stem(model)}.safetensors"
    assert ckpt["__from__"] == "safetensors"


@pytest.mark.usefixtures("loaders")
@pytest.mark.parametrize("model_type", _FAMILIES)
def test_every_family_can_be_asked_for_the_pth_instead(model_type: str, tmp_path: Path) -> None:
    """A version whose safetensors gate FAILED publishes none; the ``.pth`` is
    then the only native form and must still load, for every family - and it is
    reached by NAMING it, not by a fallback from the other container."""
    model = _Rec(model_type)
    models = _StrictModels(has_pth=True, has_safetensors=False)

    path, ckpt = _torch._fetch_native_weights(
        model, models=models, cache_dir=tmp_path, weight_format="pytorch"
    )

    assert [c[0] for c in models.calls] == ["download"]
    assert models.calls[0][1]["format"] == "pytorch"
    assert path == tmp_path / f"{_cache_stem(model)}.pth"
    assert ckpt["__from__"] == "pth"


@pytest.mark.usefixtures("loaders")
@pytest.mark.parametrize("model_type", _FAMILIES)
def test_no_family_silently_substitutes_the_other_container(
    model_type: str, tmp_path: Path
) -> None:
    """The rule is family-independent: the container asked for is the container
    fetched, and a model without it is refused rather than served the other."""
    model = _Rec(model_type)
    models = _StrictModels(has_pth=True, has_safetensors=False)

    with pytest.raises(ConflictError):
        _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format="safetensors"
        )

    assert [c[0] for c in models.calls] == ["download"], (
        "a refused safetensors must not become a .pth download"
    )
    assert models.calls[0][1]["format"] == "safetensors"


@pytest.mark.usefixtures("loaders")
@pytest.mark.parametrize("model_type", _FAMILIES)
def test_both_artifacts_cache_under_the_same_version_aware_stem(
    model_type: str, tmp_path: Path
) -> None:
    """Whichever container a caller names, a retrain must re-download it."""
    model = _Rec(model_type)
    for weight_format in ("safetensors", "pytorch"):
        models = _StrictModels(has_pth=True, has_safetensors=True)
        path, _ = _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format=weight_format
        )
        assert path.stem == _cache_stem(model)
        path.unlink()


# ───────────── the record disagreeing with the artifact ─────────────


class _PosEmbed:
    """A stand-in for the backbone's position-embedding tensor: ``(1, 1+S*S, D)``."""

    def __init__(self, side: int, dim: int = 384) -> None:
        self.shape = (1, side * side + 1, dim)


def _state_with_grid(side: int) -> dict[str, Any]:
    return {
        "backbone.0.encoder.encoder.embeddings.cls_token": _PosEmbed(0),
        "backbone.0.encoder.encoder.embeddings.position_embeddings": _PosEmbed(side),
        "class_embed.weight": _T((3, 256)),
    }


class TestPosEmbedGrid:
    """``_rfdetr_pos_embed_tokens`` reads the grid the ARTIFACT was trained on.

    This is the only artifact-side record of resolution a safetensors file has -
    there is no ``args``, no ``model_config``, nothing else - so the whole
    resolution rests on it.
    """

    @pytest.mark.parametrize(("side", "tokens"), [(26, 676), (48, 2304), (16, 256)])
    def test_reads_the_grid_area(self, side: int, tokens: int) -> None:
        assert _torch._rfdetr_pos_embed_tokens(_state_with_grid(side)) == tokens

    def test_matches_the_real_published_artifacts(self) -> None:
        """The two shapes actually measured on published fixture weights:
        rfdetr_segmentation is (1, 677, 384) at 312px and rfdetr_keypoint is
        (1, 2305, 384) at 576px - both 12px patches."""
        assert _torch._rfdetr_pos_embed_tokens(_state_with_grid(26)) == 26 * 26
        assert _torch._rfdetr_pos_embed_tokens(_state_with_grid(48)) == 48 * 48

    def test_a_non_square_grid_declines_rather_than_guessing(self) -> None:
        state = {"backbone.0.encoder.encoder.embeddings.position_embeddings": _PosEmbed(0)}
        state["backbone.0.encoder.encoder.embeddings.position_embeddings"].shape = (1, 100, 384)
        assert _torch._rfdetr_pos_embed_tokens(state) is None

    @pytest.mark.parametrize(
        "state",
        [
            {},
            None,
            "not a mapping",
            {"unrelated.weight": _T((3, 3))},
        ],
    )
    def test_absent_or_unreadable_declines(self, state: Any) -> None:
        assert _torch._rfdetr_pos_embed_tokens(state) is None


class _FakeRFDETR:
    """An rfdetr wrapper: ``.model.resolution`` plus a state dict, like the real one."""

    def __init__(self, resolution: int, side: int) -> None:
        self.model = type("_Inner", (), {"resolution": resolution, "model": self})()
        self._side = side

    def state_dict(self) -> dict[str, Any]:
        return _state_with_grid(self._side)


class TestArtifactResolutionWins:
    """A record that disagrees with the weights must not silently change the model.

    rfdetr INTERPOLATES the trained position embeddings to whatever resolution the
    synthesized container asks for, so a wrong ``training_config.resolution``
    loads, predicts, and returns a different answer with no exception. Measured on
    the published fixtures: a record claiming 576 for weights trained at 312 gave
    3 detections where the ``.pth`` container gave 1, and a record claiming 432
    for a 576 keypoint model gave 6 predictions and an extra class.
    """

    def _spec(self, resolution: int | None) -> _torch.NativeSpec:
        return _torch.NativeSpec(
            model_type="instance_segmentation",
            architecture="RF-DETR Seg",
            training_config={"resolution": resolution} if resolution else {},
            classes=["a", "b"],
            name="fixture",
        )

    def test_agreement_does_not_rebuild(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The common case must cost nothing - no second container, no second build."""
        built: list[Any] = []
        monkeypatch.setattr(_torch, "_build_rfdetr", built.append)
        module = _FakeRFDETR(resolution=312, side=26)
        out = _torch._rebuild_rfdetr_at_artifact_resolution(
            module,
            _state_with_grid(26),
            Path("w.safetensors"),
            self._spec(312),
            ["a"],
            Path("cache"),
        )
        assert out is module
        assert built == []

    def test_a_wrong_record_is_rebuilt_at_the_artifacts_resolution(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """312px weights + a record claiming 576 -> rebuilt at 312, and said so.

        The patch size is NOT hardcoded: the built module reports 576px over a
        48x48 grid, so 12px/patch, so the artifact's 26x26 grid is 312px. Change
        either number and the derived resolution follows.
        """
        seen: dict[str, Any] = {}

        def _container(w: Any, c: Any, spec: Any, cls: Any, cache: Any) -> Path:
            seen["spec"] = spec
            return Path("c.pth")

        rebuilt = _FakeRFDETR(resolution=312, side=26)
        monkeypatch.setattr(_torch, "_rfdetr_container", _container)
        monkeypatch.setattr(_torch, "_build_rfdetr", lambda _p: rebuilt)
        monkeypatch.setattr(_torch, "_verify_rfdetr_load", lambda *_a, **_k: None)

        with caplog.at_level("WARNING"):
            out = _torch._rebuild_rfdetr_at_artifact_resolution(
                _FakeRFDETR(resolution=576, side=48),
                _state_with_grid(26),
                Path("w.safetensors"),
                self._spec(576),
                ["a"],
                Path("cache"),
            )
        assert out is rebuilt
        assert seen["spec"].training_config["resolution"] == 312
        assert "trained at 312x312" in caplog.text
        assert "asked for 576x576" in caplog.text

    def test_the_patch_size_comes_from_the_built_module_not_a_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 16px-patch model derives 16px geometry, with no code change.

        Pinning this is the point: hardcoding 12 would have passed every test
        above, because both real fixtures happen to be 12px models.
        """
        seen: dict[str, Any] = {}

        def _container(w: Any, c: Any, spec: Any, cls: Any, cache: Any) -> Path:
            seen["spec"] = spec
            return Path("c.pth")

        monkeypatch.setattr(_torch, "_rfdetr_container", _container)
        monkeypatch.setattr(_torch, "_build_rfdetr", lambda _p: _FakeRFDETR(320, 20))
        monkeypatch.setattr(_torch, "_verify_rfdetr_load", lambda *_a, **_k: None)

        # built: 640px over a 40x40 grid -> 16px patches. artifact grid 20x20 -> 320px.
        _torch._rebuild_rfdetr_at_artifact_resolution(
            _FakeRFDETR(resolution=640, side=40),
            _state_with_grid(20),
            Path("w.safetensors"),
            self._spec(640),
            ["a"],
            Path("cache"),
        )
        assert seen["spec"].training_config["resolution"] == 320

    @pytest.mark.parametrize(
        ("artifact_state", "module"),
        [
            ({"nothing": 1}, _FakeRFDETR(576, 48)),  # artifact grid unreadable
            (_state_with_grid(26), None),  # module unreadable
        ],
    )
    def test_it_declines_rather_than_guessing(
        self, artifact_state: Any, module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better the record's resolution than an invented one."""
        monkeypatch.setattr(
            _torch,
            "_build_rfdetr",
            lambda _p: pytest.fail("must not rebuild on unknowns"),
        )
        sentinel = module if module is not None else object()
        out = _torch._rebuild_rfdetr_at_artifact_resolution(
            sentinel, artifact_state, Path("w.safetensors"), self._spec(576), ["a"], Path("cache")
        )
        assert out is sentinel
