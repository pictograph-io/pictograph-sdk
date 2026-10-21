"""The :class:`Augmenter` - chain augmentation ops into reproducible variants."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image as _PILImage
from pydantic import TypeAdapter

from pictograph.models.annotation import Annotation

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from PIL.Image import Image as PILImage

    from pictograph.augment._ops import Augmentation

# Validates a list of raw dicts OR existing annotation models into typed models.
_ANN_LIST_ADAPTER: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])

ImageInput = Union[str, Path, "PILImage"]
AnnotationInput = Union["Annotation", "dict[str, Any]"]


def _coerce_image(image: ImageInput) -> PILImage:
    if isinstance(image, (str, Path)):
        return _PILImage.open(image).convert("RGB")
    return image.convert("RGB")


class Augmenter:
    """Apply a pipeline of augmentation ops to an ``(image, annotations)`` pair.

    Compose ops once, then produce one or many augmented variants. Ops run in
    the order given; each transforms the image and remaps annotation geometry
    consistently. A seed makes the whole sequence of generated variants
    reproducible::

        from pictograph.augment import Augmenter, HorizontalFlip, Rotate, Brightness

        aug = Augmenter([HorizontalFlip(), Rotate((-15, 15)), Brightness((0.8, 1.2))], seed=42)
        img, anns = aug("photo.jpg", annotations)  # one variant
        variants = aug.generate("photo.jpg", annotations, 3)  # three distinct variants

    Args:
        ops: The augmentation ops to apply, in order.
        seed: Optional RNG seed. With a seed, the sequence of variants a given
            :class:`Augmenter` produces is deterministic (re-running yields the
            same variants); each successive variant still differs from the last.
            Without a seed, variants are non-deterministic.
    """

    def __init__(self, ops: Sequence[Augmentation], *, seed: int | None = None) -> None:
        self.ops = list(ops)
        self._seed = seed
        # Non-cryptographic randomness is intentional - this is image augmentation.
        self._rng = random.Random(seed)  # noqa: S311

    def reset(self) -> None:
        """Rewind the RNG to the initial seed (re-emit the same variant sequence)."""
        self._rng = random.Random(self._seed)  # noqa: S311

    def augment(
        self,
        image: ImageInput,
        annotations: Iterable[AnnotationInput] | None = None,
    ) -> tuple[PILImage, list[Annotation]]:
        """Produce a single augmented variant of ``(image, annotations)``.

        Args:
            image: A file path/str or an already-open Pillow image. Never
                mutated - a fresh image is returned.
            annotations: Annotation models or raw annotation dicts (validated
                into the typed discriminated union). Defaults to none.

        Returns:
            ``(augmented_image, augmented_annotations)`` - a new RGB Pillow image
            and the geometry-remapped annotations.
        """
        img = _coerce_image(image)
        anns: list[Annotation] = (
            _ANN_LIST_ADAPTER.validate_python(list(annotations)) if annotations else []
        )
        for op in self.ops:
            img, anns = op.apply(img, anns, self._rng)
        return img, anns

    __call__ = augment

    def generate(
        self,
        image: ImageInput,
        annotations: Iterable[AnnotationInput] | None = None,
        n: int = 1,
    ) -> list[tuple[PILImage, list[Annotation]]]:
        """Produce ``n`` distinct augmented variants of the same input.

        The source image is opened once and reused. Each variant draws fresh
        randomness from the seeded RNG, so the variants differ from one another
        yet the whole batch is reproducible under a fixed ``seed``.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        base = _coerce_image(image)
        anns_in = list(annotations) if annotations else None
        return [self.augment(base, anns_in) for _ in range(n)]
