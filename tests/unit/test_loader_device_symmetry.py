"""The two loaders must be TWINS on hardware selection - signature and behaviour.

``get_model`` and ``load_model`` are the same call with one difference: where the
weights come from. Everything else about them is supposed to be identical, and this
module asserts that mechanically rather than by review - which is how the drift this
change fixes went unnoticed. Before it, ``get_model`` took THREE overlapping hardware
arguments (``device`` + ``accelerate`` + ``providers``) and ``load_model`` took two
of the three, missing the one that names hardware. A caller with a local ``.pth`` had
no way to say which device to put it on.

Three properties are pinned here:

1. **Signature parity.** The shared arguments are the same names with the same
   defaults, and neither loader carries a hardware argument the other lacks. The
   asymmetries that REMAIN are enumerated with the reason each one is real, so a new
   one cannot be added silently.
2. **``load_model`` loads all five formats.** Refusing the two native containers was
   the deepest asymmetry of all - ``format=`` is documented as one vocabulary, and
   two of its five values used to work on only one loader.
3. **A named device is honoured or raises**, on both, with the same message shape.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from pictograph import get_model, load_model
from pictograph.inference.runtime import DEVICES, WEIGHT_FORMATS

# Arguments that legitimately belong to ONE loader, with the reason. Anything else
# appearing on one and not the other is a defect this test exists to catch.
_JUSTIFIED_ASYMMETRIES = {
    # get_model fetches, so it needs to be told WHAT to fetch and with whose key.
    "name": "get_model addresses a model by NAME; load_model takes local paths",
    "api_key": "get_model calls the API; load_model is offline and reads no key",
    "client": "same - an existing authenticated client to fetch with",
    "precision": "selects WHICH artifact to download; load_model is given the file",
    "target": "selects WHICH binding to download; load_model is given the file",
    # load_model is handed the artifact, so it needs the artifact.
    "weights": "the file itself, which get_model derives from the model record",
    "config": "the config.json, which get_model reads off the model record",
}


def _params(fn: Any) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(fn).parameters)


class TestSignatureParity:
    def test_neither_loader_has_an_unjustified_extra_argument(self) -> None:
        online, offline = _params(get_model), _params(load_model)
        unexplained = (set(online) ^ set(offline)) - set(_JUSTIFIED_ASYMMETRIES)
        assert not unexplained, (
            f"{sorted(unexplained)} is on one loader and not the other with no stated "
            f"reason. The loaders are twins: either add it to both, or record why it "
            f"cannot exist on one in _JUSTIFIED_ASYMMETRIES."
        )

    def test_the_shared_arguments_have_identical_defaults(self) -> None:
        """A shared name that means the same thing must also DEFAULT the same way -
        otherwise the twins diverge for anyone who does not pass it."""
        online, offline = _params(get_model), _params(load_model)
        shared = (set(online) & set(offline)) - {"format"}  # format's default is its own test
        mismatched = {
            name: (online[name].default, offline[name].default)
            for name in shared
            if online[name].default != offline[name].default
        }
        assert not mismatched, f"shared arguments defaulting differently: {mismatched}"

    def test_device_is_present_on_both_and_defaults_to_auto(self) -> None:
        for loader in (get_model, load_model):
            device = _params(loader).get("device")
            assert device is not None, f"{loader.__name__} has no device= argument"
            assert device.default == "auto", f"{loader.__name__}'s device must default to auto"
            assert device.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_three_old_hardware_arguments_are_gone_from_both(self) -> None:
        """`accelerate` named a latency tradeoff and `providers` named an ORT
        internal; neither is hardware, and having them beside `device` is what made
        one choice look like three."""
        for loader in (get_model, load_model):
            params = _params(loader)
            assert "accelerate" not in params, f"{loader.__name__} still takes accelerate="
            assert "providers" not in params, f"{loader.__name__} still takes providers="

    def test_format_defaults_differ_deliberately(self) -> None:
        """The ONE shared argument that defaults differently, and it is correct:
        an artifact on disk says what it is, so `load_model` reads the suffix rather
        than assuming. `get_model` has no file to ask, so it names the default."""
        assert _params(get_model)["format"].default == "onnx"
        assert _params(load_model)["format"].default is None


class TestDeviceVocabularyIsOne:
    def test_the_taught_vocabulary_is_the_hardware_and_auto(self) -> None:
        assert DEVICES == ("auto", "cpu", "cuda", "mps")

    def test_every_device_is_accepted_by_both_loaders_signatures(self) -> None:
        """Both annotate the same type, so a value legal on one is legal on the
        other - the property that makes 'identical values' checkable at all."""
        online = _params(get_model)["device"].annotation
        offline = _params(load_model)["device"].annotation
        assert online == offline == "Device"


def _config(tmp_path: Path, **over: Any) -> Path:
    payload: dict[str, Any] = {
        "model_type": "classification",
        "architecture": "resnet18",
        "class_mapping": {"classes": ["a", "b"]},
        "input_shape": [224, 224],
        "export_name": "Test Model",
    }
    payload.update(over)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadModelAcceptsEveryFormat:
    """``format=`` is one vocabulary, so ``load_model`` must speak all of it.

    It used to refuse ``pytorch`` and ``safetensors`` outright, on the grounds that
    rebuilding a checkpoint needs the model record. It does not: it needs the task,
    the architecture and the training config, and a pipeline writes all three into
    the ``config.json`` that ships beside the weights.
    """

    @pytest.mark.parametrize("fmt", WEIGHT_FORMATS)
    def test_no_format_is_rejected_for_being_a_native_checkpoint(
        self, fmt: str, tmp_path: Path
    ) -> None:
        from pictograph.inference.runtime import suffix_for_format

        weights = tmp_path / f"weights{suffix_for_format(fmt)}"  # type: ignore[arg-type]
        weights.write_bytes(b"not-a-real-artifact")
        # Every format must get PAST argument validation and fail on the bytes (or on
        # a missing optional runtime) - never on "this loader does not do that format".
        with pytest.raises(Exception) as exc:
            load_model(weights, _config(tmp_path), task="classification")
        message = str(exc.value)
        assert "use get_model" not in message, f"{fmt} is still refused by load_model"
        assert "runs a compiled graph" not in message, f"{fmt} is still refused by load_model"

    def test_a_pth_reaches_the_torch_builder_rather_than_a_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the owner hit: a local checkpoint, pinned to a device."""
        pytest.importorskip("torch")
        import pictograph.inference._torch as torch_mod

        seen: dict[str, Any] = {}

        def _spy(weights: Path, **kwargs: Any) -> Any:
            seen["weights"] = weights
            seen.update(kwargs)
            raise RuntimeError("stop here - the routing is what is under test")

        monkeypatch.setattr(torch_mod, "build_local_torch_engine", _spy)
        weights = tmp_path / "checkpoint_best_042.pth"
        weights.write_bytes(b"x")
        with pytest.raises(RuntimeError, match="stop here"):
            load_model(weights, _config(tmp_path), task="classification", device="cpu")

        assert seen["weights"] == weights
        assert seen["device"] == "cpu", "the device request must reach the torch builder"
        assert seen["weight_format"] == "pytorch"
        assert seen["spec"].model_type == "classification"
        assert seen["spec"].architecture == "resnet18"
        assert seen["spec"].classes == ["a", "b"]

    def test_the_training_config_is_read_out_of_config_json(self) -> None:
        """The rebuild picks a model definition from these keys, so losing them
        offline would rebuild a DIFFERENT module from the same checkpoint."""
        from pictograph.inference import _training_config_of

        raw = {
            "_pictograph": {"model_type": "object_detection", "architecture": "YOLOX"},
            "config": {"model_size": "s", "encoder": "resnet34", "dropout_rate": 0.3},
        }
        config = _training_config_of(raw, (640, 640))
        assert config["model_size"] == "s"
        assert config["encoder"] == "resnet34"
        assert config["dropout_rate"] == 0.3

    def test_the_input_shape_backfills_the_height_and_width(self) -> None:
        """RF-DETR nano trains at 384, and defaulting to 640 mis-sizes every input -
        the same 'artifact beats config' rule the ONNX path applies.

        The envelope DECLARES the shape here. It previously did not, which made the
        fixture impossible: `_parse_model_config` only ever returns (384, 384) when
        the artifact said so, and substitutes (640, 640) when it said nothing. The
        old fixture therefore asserted the backfill using the one input for which
        the value is a guess - see the undeclared cases below.
        """
        from pictograph.inference import _training_config_of

        raw = {"_pictograph": {"model_type": "object_detection", "input_shape": [384, 384]}}
        config = _training_config_of(raw, (384, 384))
        assert (config["image_height"], config["image_width"]) == (384, 384)

    def test_an_undeclared_input_shape_is_not_backfilled(self) -> None:
        """A guess must not be written under the keys a KNOWN value uses.

        `_parse_model_config` substitutes (640, 640) when the artifact declares no
        `input_shape`. Backfilling that made every downstream reader treat it as
        fact: RF-DETR rebuilt the module at 640 and its DINOv2 backbone asserts the
        input be divisible by 24, so `load_model(format="safetensors")` raised
        outright for any model whose config.json carries `input_shape: null` -
        reproduced on three published fixtures. Leaving the keys unset hands each
        family its own default, which is valid by construction.
        """
        from pictograph.inference import _training_config_of

        config = _training_config_of(
            {"_pictograph": {"model_type": "object_detection"}}, (640, 640)
        )
        assert "image_height" not in config
        assert "image_width" not in config

    def test_a_malformed_input_shape_is_treated_as_undeclared(self) -> None:
        """`_parse_model_config` falls back to 640 on a malformed shape, and a
        fallback is a guess whichever key it came from."""
        from pictograph.inference import _training_config_of

        for bad in ([], "512x512", [0, 0], ["a", "b"], [512]):
            config = _training_config_of(
                {"_pictograph": {"model_type": "object_detection", "input_shape": bad}},
                (640, 640),
            )
            assert "image_height" not in config, f"{bad!r} must not be treated as declared"

    def test_a_declared_shape_still_wins_for_a_family_with_its_own_default(self) -> None:
        """The fix must not stop a REAL declaration from overriding the family default:
        a classification model defaults to 224, but if the artifact says 320, use 320."""
        from pictograph.inference import _training_config_of

        raw = {"_pictograph": {"model_type": "classification", "input_shape": [320, 320]}}
        config = _training_config_of(raw, (320, 320))
        assert (config["image_height"], config["image_width"]) == (320, 320)

    def test_an_explicit_height_in_the_config_is_not_overwritten(self) -> None:
        from pictograph.inference import _training_config_of

        config = _training_config_of(
            {"config": {"image_height": 512, "image_width": 512}}, (640, 640)
        )
        assert (config["image_height"], config["image_width"]) == (512, 512)


class TestBothLoadersRefuseAnImpossibleDeviceIdentically:
    """Same refusal, same wording, whichever door the caller came through."""

    def test_an_unknown_device_is_rejected_before_anything_else_happens(
        self, tmp_path: Path
    ) -> None:
        """No network call, no file read - a device typo must not first cost an API
        round trip on one loader and nothing on the other."""
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a device this SDK knows"):
            load_model(weights, _config(tmp_path), task="classification", device="gpu")
        with pytest.raises(ValueError, match="not a device this SDK knows"):
            get_model("anything", task="classification", device="gpu", api_key="pk_live_x" * 4)

    def test_a_tensorrt_plan_on_the_cpu_is_refused_by_both(self, tmp_path: Path) -> None:
        weights = tmp_path / "sm75-trt10.13.3.9-fp32.engine"
        weights.write_bytes(b"x")
        with pytest.raises(ValueError, match="cannot use device='cpu'"):
            load_model(weights, _config(tmp_path), task="classification", device="cpu")
        with pytest.raises(ValueError, match="cannot use device='cpu'"):
            get_model(
                "anything",
                task="classification",
                format="tensorrt_engine",
                device="cpu",
                api_key="pk_live_x" * 4,
            )


class TestCheckpointContainerShapes:
    """``load_model`` now takes a checkpoint off a user's disk, so it must recognise
    every container our own pipelines write - the filename does not distinguish them.

    The classification pipeline saves the PUBLISHED ``checkpoint_best_*.pth`` as a
    bare state dict and its RESUMABLE checkpoint nested under ``model_state_dict``.
    Handing over the wrong one used to fail the strict load with a wall of
    "Missing key(s)" naming every layer in the backbone - a message that says
    nothing about the container being the problem.
    """

    @pytest.mark.parametrize("container", ["bare", "model", "model_state_dict", "state_dict"])
    def test_every_container_our_pipelines_write_is_unwrapped(self, container: str) -> None:
        torch = pytest.importorskip("torch")
        from pictograph.inference._torch import _bare_state_dict

        tensors = {"fc.weight": torch.zeros(2, 2), "fc.bias": torch.zeros(2)}
        ckpt = tensors if container == "bare" else {container: tensors}
        assert _bare_state_dict(ckpt) == tensors

    def test_a_non_tensor_mapping_is_handed_through_untouched(self) -> None:
        """`load_state_dict`'s own error is already precise; this normalises a
        container, it does not add a validation step."""
        from pictograph.inference._torch import _bare_state_dict

        assert _bare_state_dict({"epoch": 3, "loss": 0.1}) is None
        assert _bare_state_dict("not a dict") is None
