"""The two native containers - ``format="pytorch"`` (``.pth``) and
``format="safetensors"`` - and the rule that one is never served for the other.

They hold the same tensors and rebuild the same module, so it is tempting to
treat them as one thing with a fallback. They are not: ``model.safetensors`` is
published only after a publish-BLOCKING parity gate against that version's ONNX,
while the ``.pth`` is the ungated training checkpoint, and not every pipeline
writes both. ``rfdetr_keypoint`` finds no
``checkpoint_best_{total,regular,ema}.pth`` / ``checkpoint.pth`` in its
``output_dir`` and publishes ONNX + ``model.safetensors`` only, so
``model_versions.gcs_pytorch_weights_path`` is NULL and
``download(format="pytorch")`` answers 409.

So the caller names the container and gets THAT container or a refusal that says
which formats the model actually has. These tests pin exactly that: which
artifact is asked for, with which exact kwargs, what the cache does on a second
load, and what the refusal says when the requested one does not exist.

Deliberately pure-python: the download/caching wiring is exercised with the two
on-disk loaders monkeypatched out, so the base CI gate (no torch, no
safetensors, no ``[inference]`` extra) still covers it. The few tests that need
real tensors ``importorskip`` their dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pictograph.exceptions import AuthError, ConflictError, NotFoundError
from pictograph.inference import _torch
from pictograph.inference._torch import (
    _bare_state_dict,
    _cache_stem,
    _kp_counts_from_active_mask,
    _resolve_family,
    _rfdetr_variant,
    _verify_rfdetr_load,
)

_CLASSES = ["head", "torso", "l_hand", "r_hand", "l_foot", "r_foot"]


class _Model:
    """The subset of ``pictograph.models.model.Model`` the loader reads."""

    def __init__(
        self,
        *,
        model_type: str = "keypoint_detection",
        architecture: str = "RF-DETR Keypoint",
        version: str | None = "v1",
        training_config: dict[str, Any] | None = None,
    ) -> None:
        self.id = "0bcd0bcd-1111-2222-3333-444455556666"
        self.name = "Keypoint Demo"
        self.status = "ready"
        self.model_type = model_type
        self.architecture = architecture
        self.current_version_id = version
        self.class_mapping = {"classes": list(_CLASSES)}
        self.training_config = training_config if training_config is not None else {}


class _Models:
    """A stand-in for ``client.models`` that records calls instead of doing I/O.

    ``pth_error`` / ``safetensors_error``, when set, are raised instead of
    "downloading"; otherwise the target file is created so cache-hit behaviour is
    observable.
    """

    def __init__(
        self,
        *,
        pth_error: Exception | None = None,
        safetensors_error: Exception | None = None,
        manifest: list[str] | None = None,
    ) -> None:
        self.pth_error = pth_error
        self.safetensors_error = safetensors_error
        #: Wire-vocabulary `format` tokens the files manifest reports, or None to
        #: simulate a manifest lookup that itself fails.
        self.manifest = manifest
        self.download_calls: list[dict[str, Any]] = []
        self.download_file_calls: list[dict[str, Any]] = []

    def download(self, *, model_id: str, output_path: Path, format: str) -> Path:
        self.download_calls.append(
            {"model_id": model_id, "output_path": Path(output_path), "format": format}
        )
        # Both native containers come through `format=` now, so the stub
        # has to be able to fail them independently: a shared error here would
        # make "asked for one, refused the other" untestable.
        if format == "safetensors":
            if self.safetensors_error is not None:
                raise self.safetensors_error
            Path(output_path).write_bytes(b"safetensors")
            return Path(output_path)
        if self.pth_error is not None:
            raise self.pth_error
        Path(output_path).write_bytes(b"pth")
        return Path(output_path)

    def download_file(self, *, model_id: str, file_name: str, output_path: Path) -> Path:
        self.download_file_calls.append(
            {"model_id": model_id, "file_name": file_name, "output_path": Path(output_path)}
        )
        if self.safetensors_error is not None:
            raise self.safetensors_error
        Path(output_path).write_bytes(b"safetensors")
        return Path(output_path)

    def files(self, *, model_id: str) -> Any:  # noqa: ARG002 - mirrors the real kwarg
        """The manifest the refusal reads to name what the model DOES publish."""
        if self.manifest is None:
            raise RuntimeError("manifest unavailable")
        entries = [
            SimpleNamespace(version_id="v1", format=fmt, kind="weights") for fmt in self.manifest
        ]
        return SimpleNamespace(files=entries)


@pytest.fixture
def stub_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two on-disk loaders so no real torch/safetensors is needed."""
    monkeypatch.setattr(_torch, "_load_checkpoint", lambda p: {"__from__": "pth", "path": p})
    monkeypatch.setattr(
        _torch, "_load_safetensors", lambda p: {"__from__": "safetensors", "path": p}
    )


@pytest.mark.usefixtures("stub_loaders")
class TestFetchNativeWeights:
    """``weight_format`` selects the container. It is a selection, not a label."""

    def test_safetensors_asks_for_the_gated_artifact_and_never_the_pth(
        self, tmp_path: Path
    ) -> None:
        """Why the gated artifact is the one to reach for.

        Safetensors is published only after a publish-blocking parity gate
        against that version's ONNX; the ``.pth`` has never been gated. For a
        while ``rfdetr_detection`` published a ``.pth`` its ONNX was never
        exported from (measured 0.63-0.95 max|delta| across 24-37 surviving
        boxes on a real published pair), so on those models it is
        ``format="safetensors"`` that reproduces the graph.
        """
        model, models = _Model(), _Models()
        path, ckpt = _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format="safetensors"
        )

        expected = tmp_path / f"{_cache_stem(model)}.safetensors"
        assert path == expected
        assert ckpt["__from__"] == "safetensors"
        # Resolved by FORMAT, never by the literal name
        # `model.safetensors`. Artifacts are named after their model now, so the
        # by-name fetch would 404 on everything trained from 2026-07-31 on,
        # while the format route reads `gcs_safetensors_path` off the version
        # row and is therefore correct for BOTH naming eras.
        assert models.download_calls == [
            {
                "model_id": model.id,
                "format": "safetensors",
                "output_path": expected,
            }
        ]
        assert models.download_file_calls == []

    def test_pytorch_asks_for_the_pth_and_never_the_safetensors(self, tmp_path: Path) -> None:
        model, models = _Model(), _Models()
        path, ckpt = _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format="pytorch"
        )

        assert path == tmp_path / f"{_cache_stem(model)}.pth"
        assert ckpt["__from__"] == "pth"
        assert [c["format"] for c in models.download_calls] == ["pytorch"]
        assert models.download_file_calls == []

    @pytest.mark.parametrize(
        "missing",
        [
            ConflictError("no such artifact", status_code=409),
            NotFoundError("no such file", status_code=404),
        ],
    )
    def test_a_missing_container_is_refused_not_substituted(
        self, tmp_path: Path, missing: Exception
    ) -> None:
        """The rule the whole ``format=`` argument rests on: asking for one
        container and receiving the other would make the argument a label."""
        models = _Models(pth_error=missing, manifest=["onnx", "safetensors"])
        with pytest.raises(ConflictError):
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="pytorch"
            )
        assert models.download_file_calls == [], "it must not fall back to the other container"

    def test_the_refusal_names_the_formats_that_do_exist(self, tmp_path: Path) -> None:
        """ "This model has no .pth" is a dead end; "it publishes onnx,
        safetensors" is the next call."""
        models = _Models(
            pth_error=ConflictError("gcs_pytorch_weights_path is NULL", status_code=409),
            manifest=["onnx", "safetensors"],
        )
        with pytest.raises(ConflictError) as excinfo:
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="pytorch"
            )

        message = str(excinfo.value)
        assert "no 'pytorch' weights" in message
        assert "safetensors" in message and "onnx" in message
        # The underlying reason survives, so the caller isn't left guessing.
        assert "gcs_pytorch_weights_path is NULL" in message
        assert excinfo.value.status_code == 409
        assert "never substituted" in (excinfo.value.fix or "")

    def test_a_single_alternative_is_offered_as_the_exact_next_call(self, tmp_path: Path) -> None:
        models = _Models(
            safetensors_error=ConflictError("parity gate failed", status_code=409),
            manifest=["pytorch"],
        )
        with pytest.raises(ConflictError, match="format='pytorch'"):
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="safetensors"
            )

    def test_a_manifest_lookup_that_fails_omits_the_list_rather_than_inventing_one(
        self, tmp_path: Path
    ) -> None:
        """A wrong list is worse than none, and this path is already failing -
        it must never turn a clear 409 into a traceback."""
        models = _Models(pth_error=ConflictError("nothing there", status_code=409), manifest=None)
        with pytest.raises(ConflictError) as excinfo:
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="pytorch"
            )
        assert "This model publishes" not in str(excinfo.value)

    def test_the_alternatives_are_scoped_to_the_served_version(self, tmp_path: Path) -> None:
        """ "This model has a .pth" is misleading when that .pth belongs to a
        version the model no longer serves."""
        models = _Models(
            safetensors_error=ConflictError("none", status_code=409),
            manifest=["onnx", "pytorch"],
        )
        stale = SimpleNamespace(version_id="v0", format="engine", kind="weights")
        real_files = models.files
        models.files = lambda **kw: SimpleNamespace(  # type: ignore[method-assign]
            files=[stale, *real_files(**kw).files]
        )
        with pytest.raises(ConflictError) as excinfo:
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="safetensors"
            )
        assert "tensorrt_engine" not in str(excinfo.value)

    def test_an_unrelated_api_error_is_not_swallowed_into_a_409(self, tmp_path: Path) -> None:
        """Only "this model has no artifact in that format" becomes a
        ConflictError. A 401 must surface as a 401."""
        models = _Models(safetensors_error=AuthError("bad key", status_code=401))
        with pytest.raises(AuthError):
            _torch._fetch_native_weights(
                _Model(), models=models, cache_dir=tmp_path, weight_format="safetensors"
            )
        # The one attempt that was made was the one that was asked for - the
        # 401 did not become a second, `.pth`-shaped try.
        assert [c["format"] for c in models.download_calls] == ["safetensors"]

    def test_a_cached_container_short_circuits_every_download(self, tmp_path: Path) -> None:
        """A second load must not re-pay the round trip NOR re-download 163 MB."""
        model, models = _Model(), _Models()
        (tmp_path / f"{_cache_stem(model)}.safetensors").write_bytes(b"cached")

        path, ckpt = _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format="safetensors"
        )

        assert path.suffix == ".safetensors"
        assert ckpt["__from__"] == "safetensors"
        assert models.download_calls == []
        assert models.download_file_calls == []

    def test_a_cached_other_container_is_not_used_for_the_one_asked_for(
        self, tmp_path: Path
    ) -> None:
        """The cache must not reintroduce the substitution the fetch refuses:
        a machine holding an older ``.pth`` must still get the safetensors."""
        model, models = _Model(), _Models()
        (tmp_path / f"{_cache_stem(model)}.pth").write_bytes(b"cached-pth")

        path, ckpt = _torch._fetch_native_weights(
            model, models=models, cache_dir=tmp_path, weight_format="safetensors"
        )

        assert path.suffix == ".safetensors"
        assert ckpt["__from__"] == "safetensors"
        assert [c["format"] for c in models.download_calls] == ["safetensors"]

    def test_a_retrain_re_downloads_because_the_stem_is_version_aware(self, tmp_path: Path) -> None:
        """Both containers cache under the SAME version-keyed stem, so a new
        version does not keep predicting with the old tensors forever."""
        first = _Model(version="v1")
        (tmp_path / f"{_cache_stem(first)}.safetensors").write_bytes(b"cached")

        retrained = _Model(version="v2")
        models = _Models()
        path, _ckpt = _torch._fetch_native_weights(
            retrained, models=models, cache_dir=tmp_path, weight_format="safetensors"
        )

        assert path == tmp_path / f"{_cache_stem(retrained)}.safetensors"
        assert path != tmp_path / f"{_cache_stem(first)}.safetensors"
        assert [c["format"] for c in models.download_calls] == ["safetensors"]


class TestRfdetrVariantResolution:
    """The variant class name has to be supplied for a safetensors rebuild - the
    bare tensors carry no ``model_name``. LOCKSTEP with each RF-DETR pipeline's
    MODEL_CLASSES and with rfdetr's own checkpoint→class map."""

    def test_keypoint_resolves_to_the_single_preview_variant(self) -> None:
        name, weights_hint = _rfdetr_variant(_Model(model_type="keypoint_detection"))
        assert name == "RFDETRKeypointPreview"
        assert "keypoint-preview" in weights_hint

    def test_segmentation_size_comes_from_the_architecture_label(self) -> None:
        model = _Model(model_type="instance_segmentation", architecture="RF-DETR Seg Medium")
        name, weights_hint = _rfdetr_variant(model)
        assert name == "RFDETRSegMedium"
        assert weights_hint == "rf-detr-seg-medium.pth"

    def test_detection_size_comes_from_the_architecture_label(self) -> None:
        model = _Model(model_type="object_detection", architecture="RF-DETR Nano")
        assert _rfdetr_variant(model)[0] == "RFDETRNano"

    def test_training_config_model_size_is_the_fallback_source(self) -> None:
        model = _Model(
            model_type="object_detection",
            architecture="RF-DETR",
            training_config={"model_size": "large"},
        )
        assert _rfdetr_variant(model)[0] == "RFDETRLarge"


class TestVerifyRfdetrLoad:
    """rfdetr loads with ``strict=False`` and only WARNS on a partial load, so a
    wrong variant would otherwise hand back a half-initialised model that
    predicts confident nonsense."""

    class _Inner:
        def __init__(self, shapes: dict[str, tuple[int, ...]]) -> None:
            self._shapes = shapes

        def state_dict(self) -> dict[str, Any]:
            return {k: _Tensor(v) for k, v in self._shapes.items()}

    class _Wrapper:
        def __init__(self, inner: Any) -> None:
            self.model = type("M", (), {"model": inner})()

    def test_a_full_match_passes(self) -> None:
        shapes = {f"w{i}": (4, 4) for i in range(10)}
        module = self._Wrapper(self._Inner(shapes))
        _verify_rfdetr_load(module, {k: _Tensor(v) for k, v in shapes.items()}, Path("x.pth"))

    def test_a_wrong_variant_is_an_error_not_a_warning(self) -> None:
        artifact = {f"w{i}": _Tensor((4, 4)) for i in range(10)}
        # Same names, different widths - what a wrong size variant looks like.
        module = self._Wrapper(self._Inner({f"w{i}": (8, 8) for i in range(10)}))
        with pytest.raises(ValueError, match="0/10 tensors"):
            _verify_rfdetr_load(module, artifact, Path("model-v1.rfdetr.pth"))

    def test_a_small_remap_is_tolerated(self) -> None:
        """rfdetr legitimately renames a few keys during load
        (``remap_projector_to_cross_attn``), so the guard is a threshold."""
        artifact = {f"w{i}": _Tensor((4, 4)) for i in range(20)}
        built = {f"w{i}": (4, 4) for i in range(19)}  # one key remapped away
        _verify_rfdetr_load(self._Wrapper(self._Inner(built)), artifact, Path("x.pth"))


class _Tensor:
    """Minimal stand-in for a torch tensor - shape plus a summable row count."""

    def __init__(self, shape: tuple[int, ...], rows: list[int] | None = None) -> None:
        self.shape = shape
        self._rows = rows

    def sum(self, dim: int) -> _Tensor:  # noqa: ARG002 - mirrors torch's kwarg
        return _Tensor((self.shape[0],), rows=self._rows)

    def tolist(self) -> list[int]:
        return list(self._rows or [])


class TestBareStateDictHandling:
    def test_a_pth_container_yields_its_nested_model_mapping(self) -> None:
        inner = {"a": _Tensor((1,))}
        assert _bare_state_dict({"model": inner, "args": {}}) is inner

    def test_a_safetensors_mapping_is_already_the_state_dict(self) -> None:
        bare = {"a": _Tensor((1,)), "b": _Tensor((2,))}
        assert _bare_state_dict(bare) is bare

    def test_a_non_tensor_mapping_is_not_mistaken_for_weights(self) -> None:
        assert _bare_state_dict({"args": {}, "epoch": 3}) is None


class TestKeypointArityFromTensors:
    """A bare safetensors state dict has no ``args`` to read the schema from,
    but it carries ``_kp_active_mask`` - the same buffer rfdetr infers
    ``num_keypoints_per_class`` from."""

    def test_active_mask_gives_the_per_class_arity(self) -> None:
        mask = _Tensor((3, 4), rows=[2, 3, 1])
        assert _kp_counts_from_active_mask({"_kp_active_mask": mask}, 3) == [2, 3, 1]

    def test_a_row_count_that_is_not_the_class_count_is_rejected(self) -> None:
        mask = _Tensor((2, 4), rows=[2, 3])
        assert _kp_counts_from_active_mask({"_kp_active_mask": mask}, 6) is None

    def test_an_inactive_class_row_is_rejected_rather_than_mis_indexing_arity(self) -> None:
        mask = _Tensor((3, 4), rows=[2, 0, 1])
        assert _kp_counts_from_active_mask({"_kp_active_mask": mask}, 3) is None

    def test_no_mask_is_not_an_error(self) -> None:
        assert _kp_counts_from_active_mask({"other": _Tensor((1,))}, 3) is None


def test_keypoint_models_resolve_to_the_rfdetr_family_without_sniffing() -> None:
    """A safetensors artifact is a BARE state dict with no ``args``/``model``
    keys to sniff, so the keypoint family must be resolved from the model_type
    the way instance_segmentation already is."""
    assert _resolve_family("keypoint_detection", "", {"backbone.0.weight": object()}) == "rfdetr"
    assert _resolve_family("keypoint_detection", "RF-DETR Keypoint", {}) == "rfdetr"


class TestRfdetrContainer:
    """The synthesised ``.pth`` container ``RFDETR.from_checkpoint`` reads."""

    def test_a_pth_is_passed_through_untouched(self, tmp_path: Path) -> None:
        pth = tmp_path / "weights.pth"
        pth.write_bytes(b"x")
        assert _torch._rfdetr_container(pth, {}, _Model(), _CLASSES, tmp_path) is pth

    def test_safetensors_is_wrapped_once_and_reused(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")

        artifact = tmp_path / "abc-v1.safetensors"
        artifact.write_bytes(b"x")
        state = {"class_embed.weight": torch.zeros(7, 3)}
        model = _Model(training_config={"resolution": 576})

        container = _torch._rfdetr_container(artifact, state, model, _CLASSES, tmp_path)
        assert container.parent == tmp_path
        assert container.name.startswith("abc-v1.")
        assert container.name.endswith(".rfdetr.pth")
        # The basename must NOT look like an rfdetr registry asset - rfdetr's
        # own loader calls download_pretrain_weights() on whatever path it is
        # handed, and a registry-matching name would re-download over our file.
        assert "rf-detr-" not in container.name

        payload = torch.load(container, map_location="cpu", weights_only=False)
        assert set(payload["model"]) == {"class_embed.weight"}
        assert payload["model_name"] == "RFDETRKeypointPreview"
        assert payload["args"]["class_names"] == _CLASSES
        assert payload["args"]["num_classes"] == len(_CLASSES)
        assert payload["model_config"] == {"resolution": 576}

        # A second call reuses the container rather than re-serialising 163 MB.
        before = container.stat().st_mtime_ns
        again = _torch._rfdetr_container(artifact, state, model, _CLASSES, tmp_path)
        assert again == container
        assert container.stat().st_mtime_ns == before

    def test_no_resolution_in_the_record_means_no_model_config_override(
        self, tmp_path: Path
    ) -> None:
        torch = pytest.importorskip("torch")

        artifact = tmp_path / "abc-v1.safetensors"
        artifact.write_bytes(b"x")
        container = _torch._rfdetr_container(
            artifact, {"w": torch.zeros(2)}, _Model(), _CLASSES, tmp_path
        )
        payload = torch.load(container, map_location="cpu", weights_only=False)
        assert "model_config" not in payload

    def test_two_models_both_called_model_safetensors_get_their_own_container(
        self, tmp_path: Path
    ) -> None:
        """The OFFLINE case, which is the only one where the filenames collide.

        ``load_model`` is handed the file the user downloaded, and every Pictograph
        model publishes its native weights as ``model.safetensors`` - so keying the
        container on ``weights.stem`` gave every model in an organization the one
        ``model.rfdetr.pth``. The SECOND model loaded in a session found that file
        already present, returned it, and rebuilt itself from the FIRST model's
        architecture, class list and resolution.

        Two models, not one, because a single-model test passes either way: the
        collision only exists once something else has written the cache entry. The
        payloads are read back rather than just the paths compared, because the
        failure mode is reading the wrong CONTENT, not writing the wrong name.
        """
        torch = pytest.importorskip("torch")
        cache = tmp_path / "cache"
        cache.mkdir()

        first_dir, second_dir = tmp_path / "seg", tmp_path / "kp"
        loaded: list[tuple[Path, dict[str, Any]]] = []
        for directory, model, classes, state in (
            (
                first_dir,
                _Model(
                    model_type="instance_segmentation",
                    architecture="RF-DETR Seg Nano",
                    training_config={"resolution": 432},
                ),
                ["pallet", "forklift"],
                {"class_embed.weight": torch.zeros(3, 3)},
            ),
            (
                second_dir,
                _Model(training_config={"resolution": 576}),
                _CLASSES,
                {"class_embed.weight": torch.zeros(7, 3), "_kp_active_mask": torch.ones(7, 4)},
            ),
        ):
            directory.mkdir()
            artifact = directory / "model.safetensors"
            artifact.write_bytes(b"x" * (16 if directory is first_dir else 32))
            container = _torch._rfdetr_container(artifact, state, model, classes, cache)
            loaded.append(
                (container, torch.load(container, map_location="cpu", weights_only=False))
            )

        (seg_path, seg), (kp_path, kp) = loaded
        assert seg_path != kp_path

        assert seg["model_name"] == "RFDETRSegNano"
        assert seg["args"]["class_names"] == ["pallet", "forklift"]
        assert seg["model_config"] == {"resolution": 432}

        assert kp["model_name"] == "RFDETRKeypointPreview"
        assert kp["args"]["class_names"] == _CLASSES
        assert kp["model_config"] == {"resolution": 576}
        assert set(kp["model"]) == {"class_embed.weight", "_kp_active_mask"}

    def test_an_edited_config_beside_unchanged_weights_is_a_different_container(
        self, tmp_path: Path
    ) -> None:
        """The weights file alone cannot be the key.

        ``classes`` and the trained resolution come from ``config.json``, which the
        caller can change without touching a byte of ``model.safetensors`` - and both
        are baked into the container that rfdetr rebuilds from.
        """
        torch = pytest.importorskip("torch")

        artifact = tmp_path / "model.safetensors"
        artifact.write_bytes(b"x")
        state = {"class_embed.weight": torch.zeros(7, 3)}

        first = _torch._rfdetr_container(
            artifact, state, _Model(training_config={"resolution": 576}), _CLASSES, tmp_path
        )
        relabelled = _torch._rfdetr_container(
            artifact, state, _Model(training_config={"resolution": 576}), ["nose", "tail"], tmp_path
        )
        rescaled = _torch._rfdetr_container(
            artifact, state, _Model(training_config={"resolution": 432}), _CLASSES, tmp_path
        )
        assert len({first, relabelled, rescaled}) == 3
        assert torch.load(relabelled, map_location="cpu", weights_only=False)["args"][
            "class_names"
        ] == ["nose", "tail"]


def test_load_safetensors_round_trips_a_real_artifact(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file

    path = tmp_path / "m.safetensors"
    save_file({"a": torch.arange(6).reshape(2, 3).float()}, str(path))

    loaded = _torch._load_safetensors(path)
    assert set(loaded) == {"a"}
    assert tuple(loaded["a"].shape) == (2, 3)
