"""Anomaly detection (PatchCore, EfficientAD) - the canonical pre- and post-processing.

An anomaly model's ONNX graph takes ``(B, 3, S, S)`` RGB in [0, 1] - the image stretched to the
model's square input by ``resize_linear_u8`` (x86 OpenCV's INTER_LINEAR, bit for bit, on every
host), exactly as the training pipeline read its images (a resize mismatch reads as anomaly on
every normal image) - and answers RAW:
``anomaly_map`` ``(B, 1, S, S)`` and ``pred_score`` ``(B, 1)``. The model's ``config.json``
carries anomalib's post-processing numbers (``_pictograph.anomaly``: ``image_threshold``,
``pixel_threshold``, ``image_min`` / ``image_max``, ``pixel_min`` / ``pixel_max``), and THIS module
applies anomalib 2.7.1's ``PostProcessor._normalize``, number for number:

    normalized = clamp(((raw - threshold) / (max - min)) + 0.5, 0, 1)

(a NaN threshold falls back to the range's midpoint; a zero range answers 1 / 0 / 0.5 above /
below / at the threshold; a range never measured leaves the raw value). A pixel threshold the
training run could not measure is the image threshold (anomalib's threshold matching).

What a prediction is (ROADMAP Program N, design 9): the image's LABEL - the model's anomaly
class when the normalized score is OVER ``confidence`` (anomalib's ``preds > threshold``, its
default 0.5 the calibrated threshold), else its normal class - and, on an anomalous image, its
anomalous REGIONS: the pixels over ``confidence`` of the normalized map, resized to the contour
resolution, split into connected components (``mask_to_instance_polygons``), each a polygon
named the anomaly class whose confidence is its highest normalized score.

LOCKSTEP: pictograph-prt ``decode_anomaly`` (src/core/decode.cpp) ports this arithmetic and is
held to it by the parity fixtures; the SDK's ``_wrappers`` is a synced copy.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .mask_to_polygon import bounded_mask_size, mask_to_instance_polygons

#: The smallest anomalous region kept, as a share of the contour image (defects are small: the
#: semantic decode's 0.2% floor would drop a thin scratch). LOCKSTEP pictograph-prt
#: ``kAnomalyMinRegionRatio``.
ANOMALY_MIN_REGION_RATIO = 0.0005

#: The encodings a caller may ask the map in (``anomaly_map=``): ``png`` - the NORMALIZED map
#: as an 8-bit grayscale PNG (value = round(normalized x 255)), for drawing; ``float16`` - the
#: RAW map, little-endian IEEE half floats row-major, for scoring. Both at the model's input
#: resolution, base64.
MAP_ENCODINGS = ("png", "float16")


def _linear_taps(
    src: int, dst: int, horizontal: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One axis of ``cv::resize`` INTER_LINEAR on uint8: each output index's two source indices and
    their 11-bit weights (``INTER_RESIZE_COEF_SCALE`` 2048, each rounded half to even on its own).
    The source coordinate is ``(d + 0.5) * scale - 0.5`` in double, then float; the horizontal axis
    clamps index and weight at the borders, the vertical clamps only the rows. LOCKSTEP
    pictograph-prt ``cv_x_frac`` / ``cv_tap_x_u8`` / ``cv_tap_y_u8`` (include/prt/core/pixel.hpp)."""
    scale = 1.0 / (float(dst) / float(src))
    fx = ((np.arange(dst, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(fx).astype(np.int64)
    f = (fx - s.astype(np.float32)).astype(np.float32)
    if horizontal:
        low = s < 0
        f[low], s[low] = np.float32(0.0), 0
        high = s >= src - 1
        f[high], s[high] = np.float32(0.0), src - 1
        s0, s1 = s, np.where(s + 1 < src, s + 1, src - 1)
    else:
        s0, s1 = np.clip(s, 0, src - 1), np.clip(s + 1, 0, src - 1)
    w0 = np.rint((np.float32(1.0) - f) * np.float32(2048.0)).astype(np.int32)
    w1 = np.rint(f * np.float32(2048.0)).astype(np.int32)
    return s0, s1, w0, w1


def resize_linear_u8(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)`` on a uint8 image of
    any channel count, as x86 OpenCV computes it - on EVERY host. OpenCV itself does not: its
    INTER_LINEAR is not bit-exact across builds (an arm64 or 5.x build differs from the x86 4.10
    the models train on by one level in ~17% of a picture's values, measured 2026-11-25), and an
    anomaly model scores that difference - its calibration images, resized on a Mac, crossed the
    threshold they had set. The integer arithmetic of x86's SIMD path: a horizontal pass
    ``p0 * w0 + p1 * w1``, then the vertical ``(((w0 * (h0 >> 4)) >> 16) + ((w1 * (h1 >> 4)) >> 16)
    + 2) >> 2``, saturated (an exact halving is the 2 x 2 rounded average, INTER_AREA's, as
    OpenCV's own dispatch makes it). LOCKSTEP pictograph-prt ``cv_resize_linear_u8``
    (src/core/resize.cpp), which every PRT host runs; both are held to x86 cv2 by
    pictograph-prt/tests/data/parity/preprocess_cases.json."""
    src = np.asarray(image)
    if src.dtype != np.uint8:
        raise TypeError(f"resize_linear_u8 takes a uint8 image, not {src.dtype}")
    planar = src.ndim == 2
    if planar:
        src = src[:, :, None]
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    width, height = int(width), int(height)
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"cannot resize a {src_w}x{src_h} image to {width}x{height}")
    xs0, xs1, xw0, xw1 = _linear_taps(src_w, width, horizontal=True)
    ys0, ys1, yw0, yw1 = _linear_taps(src_h, height, horizontal=False)

    def hpass(rows: np.ndarray) -> np.ndarray:  # (dst_h,) source rows -> (dst_h, dst_w, C) int32
        a = src[rows[:, None], xs0[None, :]].astype(np.int32)
        b = src[rows[:, None], xs1[None, :]].astype(np.int32)
        return a * xw0[None, :, None] + b * xw1[None, :, None]

    h0, h1 = hpass(ys0), hpass(ys1)
    v = (
        ((yw0[:, None, None] * (h0 >> 4)) >> 16) + ((yw1[:, None, None] * (h1 >> 4)) >> 16) + 2
    ) >> 2
    out = np.clip(v, 0, 255).astype(np.uint8)
    return out[:, :, 0] if planar else out


def preprocess_bgr(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """A BGR uint8 image -> ``(3, height, width)`` float32 RGB in [0, 1]: the stretch resize
    (``resize_linear_u8`` on the uint8 image) every engine feeds an anomaly model, and the one the
    training pipeline reads its images with (``pipelines/anomaly``) - one function, so the
    features a model was fitted on are the features it is served, whatever the host."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = resize_linear_u8(rgb, int(width), int(height))
    return np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1))


def _f32(value: Any) -> float:
    """A config number as float32, the precision anomalib holds its post-processing buffers in: NaN
    when absent or not a JSON number (a string or a boolean is none), +-inf past float32's range.
    LOCKSTEP pictograph-prt ``config_f32`` (src/core/model_spec.cpp)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    try:
        d = float(value)
    except OverflowError:  # an integer past every float
        d = math.inf if value > 0 else -math.inf
    with np.errstate(over="ignore"):
        return float(np.float32(d))


def anomaly_normalize(values: Any, threshold: float, lo: float, hi: float) -> np.ndarray:
    """anomalib 2.7.1 ``PostProcessor._normalize`` in numpy float32, element for element."""
    x = np.asarray(values, dtype=np.float32)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return x
    t = np.float32(
        threshold if math.isfinite(threshold) else (np.float32(hi) + np.float32(lo)) / np.float32(2)
    )
    rng = np.float32(hi) - np.float32(lo)
    if rng == 0:
        return np.where(
            x > t, np.float32(1.0), np.where(x < t, np.float32(0.0), np.float32(0.5))
        ).astype(np.float32)
    return np.clip(((x - t) / rng) + np.float32(0.5), np.float32(0.0), np.float32(1.0)).astype(
        np.float32
    )


@dataclass(frozen=True)
class AnomalyCalibration:
    """A model's post-processing numbers (``_pictograph.anomaly``), float32 as anomalib holds them."""

    image_threshold: float
    pixel_threshold: float
    image_min: float
    image_max: float
    pixel_min: float
    pixel_max: float

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> AnomalyCalibration:
        meta = meta or {}
        image_t = _f32(meta.get("image_threshold"))
        pixel_t = _f32(meta.get("pixel_threshold"))
        # anomalib's threshold matching: each threshold falls back to the other.
        if not math.isfinite(image_t):
            image_t = pixel_t
        if not math.isfinite(pixel_t):
            pixel_t = image_t
        return cls(
            image_t,
            pixel_t,
            _f32(meta.get("image_min")),
            _f32(meta.get("image_max")),
            _f32(meta.get("pixel_min")),
            _f32(meta.get("pixel_max")),
        )

    def score(self, raw: Any) -> np.ndarray:
        return anomaly_normalize(raw, self.image_threshold, self.image_min, self.image_max)

    def pixels(self, raw: Any) -> np.ndarray:
        return anomaly_normalize(raw, self.pixel_threshold, self.pixel_min, self.pixel_max)


def split_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(anomaly_map (B, 1, H, W), pred_score (B,))`` from the graph's outputs in ANY order, by
    shape: the rank-4 tensor is the map, the other the score (``(B, 1)`` or ``(B,)``)."""
    amap = score = None
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 4:
            amap = arr
        elif arr.ndim in (1, 2):
            score = arr
    if amap is None or score is None:
        shapes = [tuple(np.asarray(o).shape) for o in outputs]
        raise ValueError(
            f"an anomaly graph answers a (B, 1, H, W) map and a (B, 1) score; got {shapes}"
        )
    return amap.astype(np.float32, copy=False), score.reshape(score.shape[0], -1)[:, 0].astype(
        np.float32
    )


def encode_map(amap: np.ndarray, encoding: str, calibration: AnomalyCalibration) -> dict[str, Any]:
    """One image's map (``(H, W)`` raw, the model's resolution) as the response carries it."""
    h, w = int(amap.shape[0]), int(amap.shape[1])
    if encoding == "png":
        norm = calibration.pixels(amap)
        q = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", q)
        if not ok:
            raise RuntimeError("could not encode the anomaly map")
        data = base64.b64encode(buf.tobytes()).decode("ascii")
    elif encoding == "float16":
        data = base64.b64encode(np.ascontiguousarray(amap, dtype="<f2").tobytes()).decode("ascii")
    else:
        raise ValueError(f"anomaly_map encoding {encoding!r} is not one of {MAP_ENCODINGS}")
    return {"width": w, "height": h, "encoding": encoding, "data": data}


def decode_map(payload: dict[str, Any]) -> np.ndarray:
    """The inverse of :func:`encode_map`: ``(H, W)`` float32 - the raw map for ``float16``, the
    normalized map (k / 255) for ``png``."""
    raw = base64.b64decode(str(payload["data"]))
    h, w = int(payload["height"]), int(payload["width"])
    if payload.get("encoding") == "float16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(h, w)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (h, w):
        raise ValueError("the anomaly map PNG does not decode to its declared size")
    return img.astype(np.float32) / np.float32(255.0)


@dataclass
class AnomalyPrediction:
    """One image's answer, before it is shaped into a response."""

    raw_score: float
    score: float  # normalized, [0, 1]
    is_anomalous: bool
    raw_map: np.ndarray  # (S, S) float32, the model's resolution
    regions: list[dict[str, Any]] = field(default_factory=list)  # polygon bodies + confidence


class AnomalyDetector:
    """An anomaly model's ONNX graph with its calibration. ``classes`` is the model's class list,
    ``[normal_label, anomaly_label]`` (index 1 is the anomalous class, anomalib's label 1)."""

    DEFAULT_PROVIDERS: tuple[str, ...] = (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        input_shape: tuple[int, int],
        anomaly: dict[str, Any] | None = None,
        providers: list[str] | None = None,
        sess_options: ort.SessionOptions | None = None,
        mask_max_side: int | None = None,
    ):
        if len(classes) != 2:
            raise ValueError(
                f"an anomaly model has two classes, [normal, anomaly]; got {list(classes)}"
            )
        if not isinstance(anomaly, dict) or not anomaly:
            # Without its calibration a score means nothing (LOCKSTEP pictograph-prt: refused at load).
            raise ValueError(
                "an anomaly model needs its calibration - config.json `_pictograph.anomaly` "
                "(anomalib's thresholds and ranges) - and this model has none"
            )
        self.model_path = model_path
        self.normal_label, self.anomaly_label = str(classes[0]), str(classes[1])
        self.height, self.width = int(input_shape[0]), int(input_shape[1])
        self.calibration = AnomalyCalibration.from_meta(anomaly)
        self.mask_max_side = mask_max_side
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            providers=list(providers) if providers is not None else list(self.DEFAULT_PROVIDERS),
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """A BGR uint8 image -> ``(3, S, S)`` float32 RGB in [0, 1] (:func:`preprocess_bgr`)."""
        return preprocess_bgr(image_bgr, self.width, self.height)

    def run(self, tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(B, 3, S, S)`` -> (raw maps ``(B, S, S)``, raw scores ``(B,)``)."""
        amap, score = split_outputs(
            self.session.run(None, {self.input_name: tensors.astype(np.float32)})
        )
        return amap[:, 0], score

    def decide(
        self, raw_map: np.ndarray, raw_score: float, frame_hw: tuple[int, int], confidence: float
    ) -> AnomalyPrediction:
        """One image's raw outputs -> its label and regions at ``confidence`` (see the module)."""
        score = float(self.calibration.score(np.float32(raw_score)))
        anomalous = score > float(confidence)
        pred = AnomalyPrediction(
            raw_score=float(raw_score), score=score, is_anomalous=anomalous, raw_map=raw_map
        )
        if not anomalous:
            return pred
        orig_h, orig_w = frame_hw
        tw, th = bounded_mask_size(orig_w, orig_h, self.mask_max_side)
        resized = cv2.resize(raw_map.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
        norm = self.calibration.pixels(resized)
        mask = (norm > np.float32(confidence)).astype(np.uint8)
        scale = (
            (orig_w / float(tw), orig_h / float(th)) if (tw, th) != (orig_w, orig_h) else (1.0, 1.0)
        )
        pred.regions = mask_to_instance_polygons(
            mask, min_area_ratio=ANOMALY_MIN_REGION_RATIO, scale=scale, score_map=norm
        )
        return pred

    def predict(self, image_bgr: np.ndarray, confidence: float = 0.5) -> AnomalyPrediction:
        raw_maps, raw_scores = self.run(self.preprocess(image_bgr)[None])
        return self.decide(raw_maps[0], float(raw_scores[0]), image_bgr.shape[:2], confidence)

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector(model='{self.model_path}', classes=[{self.normal_label!r}, "
            f"{self.anomaly_label!r}], input=({self.height}, {self.width}))"
        )
