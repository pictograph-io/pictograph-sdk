"""Augmentation operations - one Pillow image transform + its geometry remap.

Every op is a small, deterministic-given-a-seed callable. Geometric ops
(:class:`HorizontalFlip`, :class:`Rotate`, :class:`Crop`, …) transform the image
*and* remap annotation geometry consistently, using the same output frame Pillow
actually produced (so the annotation space never drifts from the pixels).
Photometric ops (:class:`Brightness`, :class:`Blur`, …) change pixels only and
pass geometry through unchanged.

Built on **Pillow alone** (the SDK's base dependency) - no numpy, no OpenCV - so
``pictograph.augment`` works on a bare ``pip install pictograph``.

Parameter magnitudes accept either a fixed ``float`` or a ``(low, high)`` range
that is sampled uniformly per application from the engine's seeded RNG, matching
how tools like Roboflow express augmentation strength. ``p`` is the probability
the op fires at all (otherwise the input passes through untouched).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from pictograph.augment._geometry import remap_annotations

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

    from PIL.Image import Image as PILImage

    from pictograph.models.annotation import Annotation


def _sample(value: float | tuple[float, float], rng: Random) -> float:
    """Return ``value`` if scalar, else a uniform draw from the ``(lo, hi)`` range."""
    if isinstance(value, tuple):
        lo, hi = value
        return rng.uniform(lo, hi)
    return float(value)


class Augmentation:
    """Base class for an augmentation op.

    Subclasses implement :meth:`apply`. The engine calls ops in sequence,
    threading the (image, annotations) pair and a seeded ``rng``.
    """

    #: Human-readable op name (used in reports / ``repr``).
    name: str = "augmentation"

    def apply(
        self,
        image: PILImage,
        annotations: Sequence[Annotation],
        rng: Random,
    ) -> tuple[PILImage, list[Annotation]]:  # pragma: no cover - abstract
        """Return a transformed ``(image, annotations)`` pair."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}()"


# ───────────────────────── geometric ops ─────────────────────────


class HorizontalFlip(Augmentation):
    """Mirror left↔right. ``p`` is the probability of applying it."""

    name = "horizontal_flip"

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        w, _h = image.size
        out = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        nw, nh = out.size
        return out, remap_annotations(annotations, lambda x, y: (w - x, y), nw, nh)

    def __repr__(self) -> str:
        return f"HorizontalFlip(p={self.p})"


class VerticalFlip(Augmentation):
    """Mirror top↔bottom. ``p`` is the probability of applying it."""

    name = "vertical_flip"

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        _w, h = image.size
        out = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        nw, nh = out.size
        return out, remap_annotations(annotations, lambda x, y: (x, h - y), nw, nh)

    def __repr__(self) -> str:
        return f"VerticalFlip(p={self.p})"


# Lossless 90°-step transpose constants (counter-clockwise), keyed by k.
_ROT90 = {
    1: Image.Transpose.ROTATE_90,
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_270,
}


class Rotate90(Augmentation):
    """Lossless 90°·k counter-clockwise rotation.

    ``k`` is 1/2/3 (90/180/270°). Pass ``k=None`` to pick one at random per
    application (a common "orientation" augmentation for top-down imagery).
    """

    name = "rotate90"

    def __init__(self, k: int | None = 1, p: float = 1.0) -> None:
        if k is not None and k not in (1, 2, 3):
            raise ValueError("Rotate90 k must be 1, 2, 3, or None (random).")
        self.k = k
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        k = self.k if self.k is not None else rng.choice((1, 2, 3))
        w, h = image.size
        out = image.transpose(_ROT90[k])
        nw, nh = out.size
        if k == 1:  # 90° CCW: (x, y) -> (y, w - x)
            fn = lambda x, y: (y, w - x)  # noqa: E731
        elif k == 2:  # 180°: (x, y) -> (w - x, h - y)
            fn = lambda x, y: (w - x, h - y)  # noqa: E731
        else:  # 270° CCW: (x, y) -> (h - y, x)
            fn = lambda x, y: (h - y, x)  # noqa: E731
        return out, remap_annotations(annotations, fn, nw, nh)

    def __repr__(self) -> str:
        return f"Rotate90(k={self.k}, p={self.p})"


class Rotate(Augmentation):
    """Rotate by an arbitrary angle (degrees), expanding the canvas to fit.

    ``degrees`` is a fixed angle or a ``(low, high)`` range sampled per
    application (e.g. ``Rotate((-15, 15))``). The canvas expands so no content
    is cropped, so annotations are never dropped - bounding boxes grow to the
    axis-aligned enclosure of the rotated box (standard for box augmentation).
    """

    name = "rotate"

    def __init__(
        self,
        degrees: float | tuple[float, float] = (-15.0, 15.0),
        *,
        p: float = 1.0,
        fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.degrees = degrees
        self.p = p
        self.fill = fill

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        angle = _sample(self.degrees, rng)
        rgb = image.convert("RGB")
        out = rgb.rotate(angle, expand=True, fillcolor=self.fill)
        sw, sh = rgb.size
        nw, nh = out.size
        a = math.radians(angle)
        cos_a, sin_a = math.cos(a), math.sin(a)
        scx, scy = sw / 2.0, sh / 2.0
        dcx, dcy = nw / 2.0, nh / 2.0

        def fn(x: float, y: float) -> tuple[float, float]:
            rx, ry = x - scx, y - scy
            nx = rx * cos_a + ry * sin_a
            ny = -rx * sin_a + ry * cos_a
            return nx + dcx, ny + dcy

        return out, remap_annotations(annotations, fn, nw, nh)

    def __repr__(self) -> str:
        return f"Rotate(degrees={self.degrees}, p={self.p})"


class Resize(Augmentation):
    """Resize to a fixed ``(width, height)``; scales geometry proportionally."""

    name = "resize"

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Resize width/height must be positive.")
        self.width = width
        self.height = height

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]  # noqa: ARG002
        w, h = image.size
        out = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        sx, sy = self.width / w, self.height / h
        return out, remap_annotations(
            annotations, lambda x, y: (x * sx, y * sy), self.width, self.height
        )

    def __repr__(self) -> str:
        return f"Resize(width={self.width}, height={self.height})"


class Crop(Augmentation):
    """Random crop that keeps a ``scale`` fraction of each dimension.

    ``scale`` is a fixed fraction or a ``(low, high)`` range (e.g. keep
    80-100% of width/height). The crop window is placed at a random offset.
    Annotations are clipped to the crop and dropped when less than
    ``min_visibility`` of their area survives.
    """

    name = "crop"

    def __init__(
        self,
        scale: float | tuple[float, float] = (0.8, 1.0),
        *,
        p: float = 1.0,
        min_visibility: float = 0.1,
    ) -> None:
        self.scale = scale
        self.p = p
        self.min_visibility = min_visibility

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        w, h = image.size
        fw = max(0.05, min(1.0, _sample(self.scale, rng)))
        cw = max(1, round(w * fw))
        ch = max(1, round(h * fw))
        left = rng.randint(0, w - cw)
        top = rng.randint(0, h - ch)
        out = image.crop((left, top, left + cw, top + ch))
        nw, nh = out.size
        return out, remap_annotations(
            annotations,
            lambda x, y: (x - left, y - top),
            nw,
            nh,
            clip=True,
            min_visibility=self.min_visibility,
        )

    def __repr__(self) -> str:
        return f"Crop(scale={self.scale}, p={self.p})"


# ───────────────────────── photometric ops ─────────────────────────


class _PhotometricEnhance(Augmentation):
    """Shared base for the Pillow ``ImageEnhance`` ops (geometry unchanged)."""

    _enhancer: type

    def __init__(self, factor: float | tuple[float, float] = (0.8, 1.2), *, p: float = 1.0) -> None:
        self.factor = factor
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        factor = _sample(self.factor, rng)
        out = self._enhancer(image.convert("RGB")).enhance(factor)
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(factor={self.factor}, p={self.p})"


class Brightness(_PhotometricEnhance):
    """Scale brightness (1.0 = unchanged, <1 darker, >1 brighter)."""

    name = "brightness"
    _enhancer = ImageEnhance.Brightness


class Contrast(_PhotometricEnhance):
    """Scale contrast (1.0 = unchanged)."""

    name = "contrast"
    _enhancer = ImageEnhance.Contrast


class Saturation(_PhotometricEnhance):
    """Scale color saturation (1.0 = unchanged, 0 = grayscale)."""

    name = "saturation"
    _enhancer = ImageEnhance.Color


class Grayscale(Augmentation):
    """Convert to grayscale (kept as a 3-channel RGB image). Geometry unchanged."""

    name = "grayscale"

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        out = image.convert("L").convert("RGB")
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"Grayscale(p={self.p})"


class Blur(Augmentation):
    """Gaussian blur with a fixed or sampled ``radius`` (pixels). Geometry unchanged."""

    name = "blur"

    def __init__(self, radius: float | tuple[float, float] = (0.0, 2.0), *, p: float = 1.0) -> None:
        self.radius = radius
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        radius = max(0.0, _sample(self.radius, rng))
        out = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"Blur(radius={self.radius}, p={self.p})"


class Noise(Augmentation):
    """Additive zero-mean luminance noise. Geometry unchanged.

    ``amount`` (0-1) scales the noise standard deviation. The noise *magnitude*
    is reproducible under the engine seed; the per-pixel noise *pattern* comes
    from Pillow's internal RNG (``Image.effect_noise``), so it is not itself
    seed-reproducible - this never affects annotation geometry.
    """

    name = "noise"

    def __init__(
        self, amount: float | tuple[float, float] = (0.0, 0.08), *, p: float = 1.0
    ) -> None:
        self.amount = amount
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        amount = max(0.0, _sample(self.amount, rng))
        rgb = image.convert("RGB")
        if amount <= 0.0:
            return rgb, list(annotations)
        sigma = amount * 96.0  # up to ~1 std ≈ 96/255
        noise = Image.effect_noise(rgb.size, sigma)  # 'L', mean 128
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        # (image + noise) - 128  → additive zero-mean noise, clamped to [0, 255].
        out = ImageChops.add(rgb, noise_rgb, scale=1.0, offset=-128)
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"Noise(amount={self.amount}, p={self.p})"


class HueShift(Augmentation):
    """Rotate the hue channel by ``degrees`` (0-360). Geometry unchanged."""

    name = "hue_shift"

    def __init__(
        self, degrees: float | tuple[float, float] = (-20.0, 20.0), *, p: float = 1.0
    ) -> None:
        self.degrees = degrees
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        degrees = _sample(self.degrees, rng)
        hsv = image.convert("HSV")
        h, s, v = hsv.split()
        # Pillow's H channel is 0-255 (a full turn), so 360° maps to 256 steps.
        shift = round(degrees / 360.0 * 256.0) % 256
        h = h.point(lambda px, _s=shift: (px + _s) % 256)
        out = Image.merge("HSV", (h, s, v)).convert("RGB")
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"HueShift(degrees={self.degrees}, p={self.p})"


class CutOut(Augmentation):
    """Erase ``count`` random rectangles (regularization). Geometry unchanged.

    Each hole is a fraction of the image (``size`` of each side) filled with
    ``fill``. The annotations are intentionally left as-is - the object is
    occluded, not removed, which is the point of cutout/random-erasing.
    """

    name = "cutout"

    def __init__(
        self,
        size: float | tuple[float, float] = (0.1, 0.3),
        *,
        count: int = 1,
        fill: tuple[int, int, int] = (128, 128, 128),
        p: float = 1.0,
    ) -> None:
        self.size = size
        self.count = count
        self.fill = fill
        self.p = p

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        out = image.convert("RGB").copy()
        w, h = out.size
        draw = ImageDraw.Draw(out)
        for _ in range(max(0, self.count)):
            frac = max(0.0, min(1.0, _sample(self.size, rng)))
            cw = max(1, round(w * frac))
            ch = max(1, round(h * frac))
            left = rng.randint(0, max(0, w - cw))
            top = rng.randint(0, max(0, h - ch))
            draw.rectangle([left, top, left + cw, top + ch], fill=self.fill)
        return out, list(annotations)

    def __repr__(self) -> str:
        return f"CutOut(size={self.size}, count={self.count}, p={self.p})"


class Shear(Augmentation):
    """Horizontal shear by ``degrees`` (keeps the canvas; clips geometry).

    Each row is shifted horizontally in proportion to its ``y`` (``tan(degrees)``).
    The canvas size is preserved, so content that shears off-frame is clipped and
    annotations that fall (mostly) outside are dropped.
    """

    name = "shear"

    def __init__(
        self,
        degrees: float | tuple[float, float] = (-10.0, 10.0),
        *,
        p: float = 1.0,
        fill: tuple[int, int, int] = (0, 0, 0),
        min_visibility: float = 0.1,
    ) -> None:
        self.degrees = degrees
        self.p = p
        self.fill = fill
        self.min_visibility = min_visibility

    def apply(self, image, annotations, rng):  # type: ignore[no-untyped-def]
        if rng.random() >= self.p:
            return image, list(annotations)
        angle = _sample(self.degrees, rng)
        s = math.tan(math.radians(angle))
        rgb = image.convert("RGB")
        w, h = rgb.size
        # Pillow AFFINE samples output(x,y) from input(a*x+b*y+c, d*x+e*y+f); a
        # left-shear that shifts input point (px,py) to (px + s*py, py) needs b=-s.
        out = rgb.transform(
            (w, h), Image.Transform.AFFINE, (1, -s, 0, 0, 1, 0), fillcolor=self.fill
        )
        return out, remap_annotations(
            annotations,
            lambda x, y: (x + s * y, y),
            w,
            h,
            clip=True,
            min_visibility=self.min_visibility,
        )

    def __repr__(self) -> str:
        return f"Shear(degrees={self.degrees}, p={self.p})"
