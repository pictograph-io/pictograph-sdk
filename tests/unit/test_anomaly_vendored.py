"""The anomaly models' vendored inference definitions (``pictograph.inference._anomaly``).

Without anomalib: they build from a model's metadata, load a state strictly, keep the contract's
shapes, export to ONNX and answer there as in torch. With anomalib installed (the training
image's pin - a developer's venv; the SDK's CI does not carry it): each answers exactly what
anomalib's own model answers for the same trained state, which is the gate every training run
also passes on real images (ROADMAP Program N).
"""

from __future__ import annotations

import inspect
import io
import math
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pictograph.inference._anomaly import (  # noqa: E402
    anomaly_meta_of,
    build_anomaly_module,
    load_anomaly_module,
)
from pictograph.inference._anomaly.efficient_ad import (  # noqa: E402
    EfficientAdInference,
    efficientad_state_from_anomalib,
)
from pictograph.inference._anomaly.patchcore import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    PatchcoreInference,
    feature_dim,
    gaussian_kernel,
    patchcore_state_from_anomalib,
)

PATCHCORE_META = {
    "family": "patchcore",
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "memory_bank_size": 64,
    "num_neighbors": 9,
}


def _images(n: int, size: int = 64, seed: int = 0) -> torch.Tensor:
    """Structured images in [0, 1]: smooth gradients plus noise, so features are not degenerate."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = torch.stack([xx, yy, (xx + yy) / 2])[None].repeat(n, 1, 1, 1)
    return (0.7 * base + 0.3 * torch.rand(n, 3, size, size, generator=g)).clamp(0, 1)


def _filled_patchcore(meta: dict = PATCHCORE_META, seed: int = 1) -> PatchcoreInference:
    torch.manual_seed(seed)
    module = build_anomaly_module(meta).eval()
    with torch.no_grad():
        module.memory_bank.copy_(torch.rand_like(module.memory_bank))
    return module


def test_meta_is_read_from_the_artifact_or_a_bare_envelope() -> None:
    meta = {"family": "efficientad", "model_size": "small"}
    assert anomaly_meta_of({"_pictograph": {"anomaly": meta}, "training": {}}) == meta
    assert anomaly_meta_of({"anomaly": meta}) == meta
    assert anomaly_meta_of({"_pictograph": {"name": "x"}}) is None
    assert anomaly_meta_of(None) is None


def test_unknown_family_and_variants_refuse() -> None:
    with pytest.raises(ValueError, match="unknown anomaly family"):
        build_anomaly_module({"family": "padim"})
    with pytest.raises(ValueError, match="unknown PatchCore backbone"):
        build_anomaly_module({**PATCHCORE_META, "backbone": "vit_b_16"})
    with pytest.raises(ValueError, match="unknown EfficientAD size"):
        build_anomaly_module({"family": "efficientad", "model_size": "large"})


def test_patchcore_shapes_and_memory_bank_width() -> None:
    module = _filled_patchcore()
    assert module.memory_bank.shape == (64, feature_dim("wide_resnet50_2", ("layer2", "layer3")))
    assert feature_dim("wide_resnet50_2", ("layer2", "layer3")) == 1536
    assert feature_dim("resnet18", ("layer2", "layer3")) == 384
    with torch.no_grad():
        amap, score = module(_images(2, 64))
    assert amap.shape == (2, 1, 64, 64) and score.shape == (2, 1)
    assert torch.isfinite(amap).all() and torch.isfinite(score).all()


def test_patchcore_trunk_stops_at_its_deepest_layer() -> None:
    names = {
        k.split(".")[1]
        for k in build_anomaly_module(PATCHCORE_META).state_dict()
        if k.startswith("trunk.")
    }
    assert names == {"conv1", "bn1", "layer1", "layer2", "layer3"}


def test_efficientad_shapes() -> None:
    for size in ("small", "medium"):
        module = build_anomaly_module({"family": "efficientad", "model_size": size}).eval()
        with torch.no_grad():
            amap, score = module(_images(1, 256))
        assert amap.shape == (1, 1, 256, 256) and score.shape == (1, 1)


def test_load_is_strict() -> None:
    module = _filled_patchcore()
    state = module.state_dict()
    assert load_anomaly_module(PATCHCORE_META, state).training is False
    missing = {k: v for k, v in state.items() if k != "memory_bank"}
    with pytest.raises(RuntimeError, match="memory_bank"):
        load_anomaly_module(PATCHCORE_META, missing)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_anomaly_module(PATCHCORE_META, {**state, "extra.weight": torch.zeros(1)})


def test_gaussian_kernel_is_anomalibs() -> None:
    """33 taps for sigma 4, symmetric, summing to 1 - and, where kornia is installed, its values."""
    k = gaussian_kernel(4.0)
    assert k.shape == (1, 1, 33, 33)
    assert math.isclose(float(k.sum()), 1.0, rel_tol=1e-6)
    assert torch.allclose(k, k.flip(-1)) and torch.allclose(k, k.transpose(-1, -2))
    kornia = pytest.importorskip("kornia")
    from kornia.filters.kernels import normalize_kernel2d

    ref = normalize_kernel2d(kornia.filters.get_gaussian_kernel2d((33, 33), (4.0, 4.0)))
    assert torch.allclose(k.view(33, 33), ref.view(33, 33), rtol=0, atol=1e-8)


def test_normalization_runs_inside_the_patchcore_module() -> None:
    module = build_anomaly_module(PATCHCORE_META)
    assert torch.allclose(module.pixel_mean.view(3), torch.tensor(IMAGENET_MEAN))
    assert torch.allclose(module.pixel_std.view(3), torch.tensor(IMAGENET_STD))


# The export traces Python arithmetic on the input's size (EfficientAD's decoder sizes, as
# anomalib's own export does): a graph is specialized to its model's input size by contract.
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
# torch's exporter notes a slice it does not constant-fold (an optimizer remark, not a defect).
@pytest.mark.filterwarnings("ignore:Constant folding:UserWarning")
@pytest.mark.parametrize("family", ["patchcore", "efficientad"])
def test_onnx_export_answers_as_torch(family: str) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    if family == "patchcore":
        module, size = _filled_patchcore(), 64
    else:
        torch.manual_seed(2)
        module, size = EfficientAdInference(model_size="small").eval(), 256
    x = _images(2, size, seed=5)
    buf = io.BytesIO()
    # dynamo=False: torch 2.9 made the dynamo exporter the default (it needs onnxscript); the
    # platform exports with TorchScript's (test_multiruntime_parity.py says why). The filter is
    # scoped to the call: the suite escalates DeprecationWarning, and this one is torch's.
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            x[:1],
            buf,
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
    session = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    amap, score = session.run(None, {"input": x.numpy()})
    with torch.no_grad():
        t_amap, t_score = module(x)
    assert amap.shape == tuple(t_amap.shape) and score.shape == tuple(t_score.shape)
    assert torch.allclose(torch.from_numpy(amap), t_amap, rtol=1e-4, atol=1e-5)
    assert torch.allclose(torch.from_numpy(score), t_score, rtol=1e-4, atol=1e-5)


# ── against anomalib itself (its pin installed: the training image, a developer venv) ──

#: anomalib's own deprecation notices (timm's old import path inside it, its dinov2 package,
#: ``subsample_embedding``'s retired argument - raised however it is called, as anomalib's own
#: Lightning module calls it), not this module's.
anomalib_notices = pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::DeprecationWarning")


def _anomalib() -> object:
    anomalib = pytest.importorskip("anomalib")
    if not hasattr(torch.compiler, "is_exporting"):
        # PatchCore's search asks torch.compiler.is_exporting (added after torch 2.5, the
        # training image's torch): it only decides whether to chunk the search, so False is
        # its answer outside an export.
        torch.compiler.is_exporting = lambda: False
    return anomalib


@anomalib_notices
def test_patchcore_answers_exactly_as_anomalib() -> None:
    _anomalib()
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    torch.manual_seed(3)
    ref = PatchcoreModel(layers=["layer2", "layer3"], backbone="wide_resnet50_2", pre_trained=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    train = _images(6, 64, seed=10)
    ref.train()
    with torch.no_grad():
        ref((train - mean) / std)
    torch.manual_seed(4)
    ref.subsample_embedding(0.25)
    ref.eval()
    state = patchcore_state_from_anomalib(ref.state_dict())
    meta = {**PATCHCORE_META, "memory_bank_size": int(ref.memory_bank.shape[0])}
    ours = load_anomaly_module(meta, state)
    test = _images(3, 64, seed=11)
    with torch.no_grad():
        expected = ref((test - mean) / std)
        amap, score = ours(test)
    assert torch.allclose(amap, expected.anomaly_map, rtol=1e-5, atol=1e-6)
    assert torch.allclose(score.view(-1), expected.pred_score.view(-1), rtol=1e-5, atol=1e-6)


@anomalib_notices
def test_patchcore_state_needs_a_fitted_memory_bank() -> None:
    _anomalib()
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    ref = PatchcoreModel(layers=["layer2", "layer3"], backbone="wide_resnet50_2", pre_trained=False)
    with pytest.raises(ValueError, match="no memory bank"):
        patchcore_state_from_anomalib(ref.state_dict())


@anomalib_notices
@pytest.mark.parametrize("model_size", ["small", "medium"])
def test_efficientad_answers_exactly_as_anomalib(model_size: str) -> None:
    _anomalib()
    from anomalib.models.image.efficient_ad.torch_model import (
        EfficientAdModel,
        EfficientAdModelSize,
    )

    torch.manual_seed(5)
    ref = EfficientAdModel(model_size=EfficientAdModelSize(model_size)).eval()
    with torch.no_grad():
        ref.mean_std["mean"].copy_(torch.rand_like(ref.mean_std["mean"]))
        ref.mean_std["std"].copy_(torch.rand_like(ref.mean_std["std"]) + 0.5)
        for key, value in {"qa_st": 0.2, "qb_st": 1.7, "qa_ae": 0.1, "qb_ae": 0.9}.items():
            ref.quantiles[key].fill_(value)
    state = efficientad_state_from_anomalib(ref.state_dict())
    ours = load_anomaly_module({"family": "efficientad", "model_size": model_size}, state)
    x = _images(2, 256, seed=12)
    with torch.no_grad():
        expected = ref(x)
        amap, score = ours(x)
    assert torch.allclose(amap, expected.anomaly_map, rtol=1e-5, atol=1e-6)
    assert torch.allclose(score, expected.pred_score, rtol=1e-5, atol=1e-6)


@anomalib_notices
def test_efficientad_state_needs_its_statistics() -> None:
    _anomalib()
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel

    with pytest.raises(ValueError, match="never set"):
        efficientad_state_from_anomalib(EfficientAdModel().state_dict())


def test_imagenet_norm_batch_is_anomalibs_at_float32_and_keeps_a_half_input_half() -> None:
    """ROADMAP Program N: an FP16 graph halves EfficientAD's networks; the normalization at their entry must not promote
    their input back to float32 (a float input against half weights is refused), and at float32 it is anomalib's."""
    from pictograph.inference._anomaly.efficient_ad import imagenet_norm_batch

    x = torch.rand(2, 3, 8, 8)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    assert torch.equal(imagenet_norm_batch(x), (x - mean) / std), (
        "bit for bit anomalib's at float32"
    )
    half = imagenet_norm_batch(x.half())
    assert half.dtype == torch.float16
    assert torch.allclose(half.float(), (x - mean) / std, atol=1e-2)
