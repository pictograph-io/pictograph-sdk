"""THE annotation look's configuration: every primitive's attributes in one object.

The same object Pictograph's live views and annotated videos are drawn with, and the app's
image test: widths and radii are pixels AT 1080p and follow the image (``u`` = its longer
side / 1920; a width is never thinner than ``min(2, its 1080p width)``); a label's size
follows the label rule times ``scale.label``; its corner radius is in em; opacities and
amounts run 0 - 1; colors are hex. A class's color is THE Pictograph palette (or what the
dataset stores) unless ``classes`` names one.

Every field has THE look's value, so a partial mapping is a complete config::

    draw_annotations(img, anns, look={"box": {"fill": 0}, "label": {"position": "tl"}})
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Anchor9 = Literal["tl", "tc", "tr", "cl", "cc", "cr", "bl", "bc", "br"]
Corner = Literal["tl", "tr", "bl", "br"]


class _Part(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Keyline(_Part):
    """A stroke one hairline wider each side, drawn under the color so a line holds on any ground.

    THE look draws none (``box.keyline`` and ``outline.keyline`` are ``None``); give one to hold a line on a busy
    ground: ``look={"box": {"keyline": {"alpha": 0.42}}, "outline": {"keyline": {"alpha": 0.3}}}`` (a box's
    keyline is also a point's, a polyline's and, with ``bone.keyline``, a bone's)."""

    color: Literal["ink", "paper"] = "ink"
    alpha: float = Field(default=0.42, ge=0.0, le=1.0)


class ScaleConfig(_Part):
    """``sublinear``: the label is ``max(12, 20 * u ** 0.6)`` px (20 at 1080p, 15.7 at 720p, 30.3 at 4K);
    ``linear``: ``max(12, 20 * u)``. ``label`` multiplies either."""

    rule: Literal["sublinear", "linear"] = "sublinear"
    label: float = Field(default=1.0, gt=0.0)


class BoxConfig(_Part):
    width: float = Field(default=2.0, ge=0.0)
    corners: Literal["square", "brackets"] = "square"
    keyline: Keyline | None = None
    #: The fill's opacity (its color lifted toward white by ``fill.frost``).
    fill: float = Field(default=0.2, ge=0.0, le=1.0)


class OutlineConfig(_Part):
    """A polygon's (instance mask's) rim."""

    width: float = Field(default=2.0, ge=0.0)
    keyline: Keyline | None = None


class FillConfig(_Part):
    """A polygon's fill: its opacity, and how far its color (and a box's fill) lifts toward white."""

    opacity: float = Field(default=0.28, ge=0.0, le=1.0)
    frost: float = Field(default=0.15, ge=0.0, le=1.0)


class RegionConfig(_Part):
    """A semantic region's fill opacity and edge width."""

    opacity: float = Field(default=0.34, ge=0.0, le=1.0)
    outline: float = Field(default=1.0, ge=0.0)


class PointConfig(_Part):
    #: px at 1080p, scaled with the image like every size (``Sizes.rad``): 3 px at a 640 px longer side, 6 at 1280, 9 at
    #: 1920 (4.5 until 2026-11-05: floored to 3 px up to 1280, so a 720p stream's dot was half a 640 image's). LOCKSTEP
    #: the engine's kPointRadius and the app's InferenceOverlay pointRadius.
    radius: float = Field(default=9.0, ge=0.0)
    style: Literal["solid", "paper", "glass", "ring", "dot"] = "solid"


class BoneConfig(_Part):
    width: float = Field(default=2.5, ge=0.0)
    #: A bone takes the box's (or the outline's) keyline when there is one.
    keyline: bool = True
    #: "class" (the default since 2026-11-05): each joint its own class's color, each bone its pose class's - as a
    #: tile and the engine draw a pose. "instance": one palette color per person.
    color: Literal["instance", "class"] = "class"


class PolylineConfig(_Part):
    width: float = Field(default=3.0, ge=0.0)
    ends: bool = True


class TailConfig(_Part):
    """A track's tail (the engine's; carried so one config holds everywhere)."""

    width: float = Field(default=3.0, ge=0.0)
    fade: bool = True


class LabelParts(_Part):
    """What a label says: its class name, a tracked detection's id, a prediction's score (off: THE look's label is its
    name and track id)."""

    name: bool = True
    track: bool = True
    confidence: bool = False


class LabelConfig(_Part):
    show: bool = True
    #: The glass's tone: frost (light, the app's paper) or smoke (dark, the app's ink).
    glass: Literal["frost", "smoke"] = "frost"
    #: How much of the class color is in the glass: 0 clear glass, 1 the class color itself.
    tint: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Opaque by default: nothing is blurred behind a label, so an overlapped label never shows through the one on top.
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    #: The words: the app's ink or inverse ink by contrast, or the class color's own deep (or light) shade.
    ink: Literal["contrast", "shade"] = "contrast"
    radius: float = Field(default=0.25, ge=0.0)
    #: Where a box's (a rotated box's, a polyline's, a keypoint's) label sits on it.
    position: Anchor9 = "tc"
    #: Where a polygon's label sits: ``tc`` over its own top band, ``cc`` at its center (the point inside it
    #: farthest from its edges), any other anchor on its box.
    mask_position: Anchor9 = "tc"
    style: Literal["chip", "tab", "inset", "split", "leader", "text"] = "chip"
    parts: LabelParts = LabelParts()
    typeface: Literal["sans", "pair", "mono"] = "sans"
    #: Where a label meets another: draw it anyway, move it to the next free place, or leave it out.
    overlap: Literal["draw", "shift", "hide"] = "draw"


class LineConfig(_Part):
    width: float = Field(default=3.0, ge=0.0)
    color: str = "#42a5f5"
    arrow: bool = True


class ZoneConfig(_Part):
    opacity: float = Field(default=0.12, ge=0.0, le=1.0)
    width: float = Field(default=2.0, ge=0.0)
    color: str = "#ab47bc"
    dash: bool = True


class BadgeConfig(_Part):
    """The classifier badges: their size against a label, and the image corner they stack in."""

    scale: float = Field(default=1.25, gt=0.0)
    corner: Corner = "tl"


class ClassStyle(_Part):
    """What one class may set for itself."""

    color: str | None = None
    #: x every width the class draws.
    width: float | None = Field(default=None, gt=0.0)
    #: Its box / polygon fill opacity.
    fill: float | None = Field(default=None, ge=0.0, le=1.0)
    label: bool | None = None
    position: Anchor9 | None = None


class Look(_Part):
    """THE annotation look (every live view and annotated video, the app's image test, ``draw_annotations``).

    Every field defaults to THE look ("Color frost"); give only what you change.
    """

    scale: ScaleConfig = ScaleConfig()
    box: BoxConfig = BoxConfig()
    outline: OutlineConfig = OutlineConfig()
    fill: FillConfig = FillConfig()
    region: RegionConfig = RegionConfig()
    point: PointConfig = PointConfig()
    bone: BoneConfig = BoneConfig()
    polyline: PolylineConfig = PolylineConfig()
    tail: TailConfig = TailConfig()
    label: LabelConfig = LabelConfig()
    line: LineConfig = LineConfig()
    zone: ZoneConfig = ZoneConfig()
    badge: BadgeConfig = BadgeConfig()
    classes: dict[str, ClassStyle] = Field(default_factory=dict)


_PAPER_KEY = Keyline(color="paper", alpha=0.7)

#: The directions the look was chosen from (``color`` is THE look), by name: ``look="smoke"``.
LOOK_PRESETS: dict[str, Look] = {
    "color": Look(),
    "frost": Look(
        box=BoxConfig(keyline=_PAPER_KEY, fill=0.0),
        outline=OutlineConfig(keyline=Keyline(color="paper", alpha=0.55)),
        fill=FillConfig(opacity=0.3, frost=0.35),
        region=RegionConfig(opacity=0.32),
        point=PointConfig(style="paper"),
        label=LabelConfig(tint=0.18, alpha=0.84, ink="shade"),
    ),
    "smoke": Look(
        box=BoxConfig(keyline=Keyline(color="ink", alpha=0.5), fill=0.0),
        outline=OutlineConfig(keyline=Keyline(color="ink", alpha=0.32)),
        fill=FillConfig(opacity=0.26, frost=0.0),
        region=RegionConfig(opacity=0.3),
        point=PointConfig(style="glass"),
        label=LabelConfig(glass="smoke", tint=0.22, alpha=0.8),
    ),
}
