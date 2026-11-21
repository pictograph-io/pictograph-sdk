"""THE annotation look, drawn with Pillow - pass for pass the reference renderer Pictograph's look was designed
on (the engine's and the app's draw the same passes):

1. the photo
2. ONE fill layer per opacity (regions, polygons, box fills), blended once
3. strokes - bones, region edges, polygon rims, boxes, rotated boxes, polylines, points - drawn at 2x and
   reduced by area (anti-aliased; an axis-aligned box stays pixel-crisp)
4. labels: laid out first (so one can make way for another), painted weakest first; classifier badges last

Nothing is blurred, for speed: the look is drawn only with what the engine and Pillow already have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Literal

from PIL import Image, ImageChops, ImageDraw, ImageFont

from pictograph._look.config import Anchor9, ClassStyle, Keyline, Look

if TYPE_CHECKING:
    from collections.abc import Callable

#: THE Pictograph palette, the one the app and the engine draw classes in.
PALETTE: tuple[str, ...] = (
    "#ed3e49",  # Product Red
    "#ec407a",  # Hot Pink
    "#ff5e24",  # Product Orange
    "#ff9800",  # Tangerine
    "#ffab91",  # Peach
    "#ffc540",  # Product Yellow
    "#ffd54f",  # Gold
    "#30A941",  # Green
    "#5AE29E",  # Mint Green
    "#26AD81",  # Teal
    "#64ffda",  # Aqua
    "#4dd0e1",  # Cyan Light
    "#26c6da",  # Turquoise
    "#00e5ff",  # Cyan
    "#42a5f5",  # Sky Blue
    "#3e62b0",  # Product Blue
    "#7986cb",  # Periwinkle
    "#722f8f",  # Product Purple
    "#ab47bc",  # Lavender
    "#ba68c8",  # Orchid
)

# The glass, in the app's own colors.
INK = "#21232b"
INK_INVERSE = "#f7f8f6"
PAPER = "#f4f3f0"
WHITE = "#ffffff"

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
Pt = tuple[float, float]


def class_color(name: str, stored: dict[str, str] | None = None) -> str:
    """The color class ``name`` is drawn in: its dataset's stored color, else THE palette color its name hashes to -
    the app's ``classColor`` and core's ``prediction_color``: ``h = h * 31 + c`` over the name's UTF-16 code units
    (JavaScript's ``charCodeAt``), modulo 2 ** 32."""
    if stored and stored.get(name):
        return stored[name]
    h = 0
    units = name.encode("utf-16-le")
    for i in range(0, len(units), 2):
        h = (h * 31 + int.from_bytes(units[i : i + 2], "little")) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


# ── Color ─────────────────────────────────────────────────────────────────────────────────────


def _hex(color: str) -> RGB:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"not a hex color: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _to_hex(rgb: RGB) -> str:
    return "#" + "".join(f"{v:02x}" for v in rgb)


def _jsround(v: float) -> int:
    """JavaScript's ``Math.round`` (half up), so the SDK sizes things exactly as the reference renderer."""
    return math.floor(v + 0.5)


def mix(a: str, b: str, t: float) -> str:
    ra, ga, ba = _hex(a)
    rb, gb, bb = _hex(b)
    return _to_hex(
        (_jsround(ra + (rb - ra) * t), _jsround(ga + (gb - ga) * t), _jsround(ba + (bb - ba) * t))
    )


def _rgba(color: str, alpha: float) -> RGBA:
    r, g, b = _hex(color)
    return r, g, b, max(0, min(255, _jsround(alpha * 255)))


def _rel_lum(color: str) -> float:
    def lin(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = _hex(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = _rel_lum(a), _rel_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@lru_cache(maxsize=1024)
def label_ink(glass: str) -> tuple[str, str]:
    """A label's color and its words: the app's ink or inverse ink, whichever has the higher contrast ratio on it
    (WCAG 2) - dark words on every palette color but Product Blue and Product Purple, at least 4.09:1 on every one.
    Returns ``(glass, ink)``; the color is never changed. The rule every renderer carries (the engine's
    ``contrast_text``, the app's ``labelInk``)."""
    return glass, INK if contrast(INK, glass) >= contrast(INK_INVERSE, glass) else INK_INVERSE


def shade_on(color: str, glass: str) -> str:
    """A class color's shade for words on a glass: deep on light glass, light on dark, at least 4.5:1."""
    toward = INK if _rel_lum(glass) > 0.3 else WHITE
    t = 0.45
    while t <= 1.0001:
        c = mix(color, toward, t)
        if contrast(c, glass) >= 4.5:
            return c
        t += 0.05
    return toward


def _key_color(k: Keyline) -> str:
    return INK if k.color == "ink" else PAPER


def _instance_color(instance: int) -> str:
    return PALETTE[(instance - 1) % len(PALETTE)]


# ── What is drawn ───────────────────────────────────────────────────────────────────────────


@dataclass
class Shape:
    """One annotation to draw, in image pixels."""

    kind: Literal["box", "obb", "polygon", "region", "polyline", "keypoint"]
    name: str
    #: Shown on the label when not None.
    confidence: float | None = None
    track: int | None = None
    #: The axis-aligned box (x, y, w, h): a box's own; a polygon's, a region's, a rotated box's enclosure.
    box: tuple[float, float, float, float] | None = None
    #: A rotated box's corners, clockwise.
    corners: list[Pt] | None = None
    #: A polygon's or a region's rings (holes are rings inside rings: even-odd).
    rings: list[list[Pt]] | None = None
    path: list[Pt] | None = None
    point: Pt | None = None
    #: A keypoint's instance (a pose: one palette color per instance).
    instance: int | None = None


@dataclass
class Bone:
    a: Pt
    b: Pt
    name: str
    instance: int | None = None


@dataclass
class Badge:
    name: str
    confidence: float | None = None


# ── Sizes for one image ─────────────────────────────────────────────────────────────────────


@dataclass
class Sizes:
    W: int
    H: int
    u: float
    em: float
    badge_em: float
    gap: int
    inset: int
    hair: int
    #: An explicit stroke width (``draw_annotations(width=...)``): every width this many px, unscaled.
    stroke: int | None = None

    def px(self, v: float) -> int:
        """A width given at 1080p, in this image's whole pixels (never thinner than min(2, its 1080p value))."""
        if v <= 0:
            return 0
        if self.stroke is not None:
            return self.stroke
        return max(min(2, _jsround(v)), _jsround(v * self.u))

    def rad(self, v: float) -> float:
        """A radius given at 1080p, in this image's pixels (never under min(3, its 1080p value))."""
        if v <= 0:
            return 0.0
        return max(min(3.0, v), v * self.u)


def label_em(w: int, h: int, cfg: Look) -> float:
    u = max(w, h) / 1920
    base = 20 * u if cfg.scale.rule == "linear" else 20 * u**0.6
    return max(12.0, base) * cfg.scale.label


def sizes_for(
    w: int, h: int, cfg: Look, font_size: float | None = None, stroke: int | None = None
) -> Sizes:
    u = max(w, h) / 1920
    em = float(font_size) if font_size is not None else label_em(w, h, cfg)
    return Sizes(
        W=w,
        H=h,
        u=u,
        em=em,
        badge_em=round(em * cfg.badge.scale * 10) / 10,
        gap=max(2, _jsround(em * 0.18)),
        inset=max(8, _jsround(0.015 * min(w, h))),
        hair=max(1, _jsround(u)),
        stroke=stroke,
    )


# ── Fonts ─────────────────────────────────────────────────────────────────────────────────────

_FONT_FILES = {
    "sans": "OpenSans-SemiBold.ttf",
    "sans-regular": "OpenSans-Regular.ttf",
    "mono": "OxygenMono-Regular.ttf",
}


@lru_cache(maxsize=64)
def _font(face: str, size: float) -> ImageFont.FreeTypeFont:
    ref = resources.files("pictograph._look") / "fonts" / _FONT_FILES[face]
    with resources.as_file(ref) as path:
        return ImageFont.truetype(str(path), max(1.0, size))


@dataclass
class _Fonts:
    name: ImageFont.FreeTypeFont
    value: ImageFont.FreeTypeFont
    strong: ImageFont.FreeTypeFont
    cap: float


def _fonts(em: float, cfg: Look) -> _Fonts:
    face = cfg.label.typeface
    if face == "mono":
        return _Fonts(
            _font("mono", em), _font("mono", em * 0.92), _font("mono", em * 0.92), em * 0.7
        )
    if face == "pair":
        return _Fonts(
            _font("sans", em), _font("mono", em * 0.88), _font("mono", em * 0.88), em * 0.714
        )
    return _Fonts(
        _font("sans", em), _font("sans-regular", em * 0.92), _font("sans", em * 0.92), em * 0.714
    )


def _width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text)) if text else 0.0


# ── Geometry ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float


def _bounds(pts: list[Pt]) -> Rect:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _overlaps(a: Rect, b: Rect, pad: float = 1) -> bool:
    return (
        a.x < b.x + b.w + pad
        and b.x < a.x + a.w + pad
        and a.y < b.y + b.h + pad
        and b.y < a.y + a.h + pad
    )


def shape_center(rings: list[list[Pt]], precision: float = 1.0) -> Pt:
    """A shape's center for its label: the point inside it farthest from its edges (its pole of inaccessibility),
    by the cell search of Mapbox's polylabel - the reference renderer's ``shapeCenter``. Holes count (even-odd)."""
    pts = [p for r in rings for p in r]
    if not pts:
        return 0.0, 0.0
    b = _bounds(pts)
    if b.w <= 0 or b.h <= 0:
        return b.x, b.y

    def signed(x: float, y: float) -> float:
        inside = False
        best = math.inf
        for r in rings:
            n = len(r)
            j = n - 1
            for i in range(n):
                ax, ay = r[i]
                bx, by = r[j]
                if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
                    inside = not inside
                dx, dy = bx - ax, by - ay
                len2 = dx * dx + dy * dy
                t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / len2)) if len2 > 0 else 0.0
                ex, ey = ax + t * dx - x, ay + t * dy - y
                best = min(best, ex * ex + ey * ey)
                j = i
        return (1 if inside else -1) * math.sqrt(best)

    def cell(x: float, y: float, half: float) -> tuple[float, float, float, float, float]:
        d = signed(x, y)
        return x, y, half, d, d + half * math.sqrt(2)

    size = min(b.w, b.h)
    queue = []
    x = b.x
    while x < b.x + b.w:
        y = b.y
        while y < b.y + b.h:
            queue.append(cell(x + size / 2, y + size / 2, size / 2))
            y += size
        x += size
    best = cell(b.x + b.w / 2, b.y + b.h / 2, 0)
    while queue:
        k = max(range(len(queue)), key=lambda i: queue[i][4])
        c = queue[k]
        queue[k] = queue[-1]
        queue.pop()
        if c[3] > best[3]:
            best = c
        if c[4] - best[3] <= precision:
            continue
        half = c[2] / 2
        queue += [
            cell(c[0] - half, c[1] - half, half),
            cell(c[0] + half, c[1] - half, half),
            cell(c[0] - half, c[1] + half, half),
            cell(c[0] + half, c[1] + half, half),
        ]
    return best[0], best[1]


def mask_top(rings: list[list[Pt]], box: Rect) -> Rect:
    """Over a polygon's own top (its highest band, not its box's middle): a running person's label sits over the
    head, not over the arm the box reaches out to - the reference renderer's ``maskTop``."""
    pts = [p for r in rings for p in r]
    if not pts:
        return box
    min_y = min(p[1] for p in pts)
    band = min_y + max(4.0, box.h * 0.06)
    xs = [p[0] for p in pts if p[1] <= band]
    return Rect((min(xs) + max(xs)) / 2, box.y, 0, box.h)


# ── The renderer ────────────────────────────────────────────────────────────────────────────


@dataclass
class _Ctx:
    canvas: Image.Image
    photo: Image.Image
    cfg: Look
    s: Sizes
    stored: dict[str, str] | None
    ss: int
    layer: Image.Image | None = None

    def color(self, name: str) -> str:
        own = self.cfg.classes.get(name)
        if own is not None and own.color:
            return own.color
        return class_color(name, self.stored)

    def own(self, name: str) -> ClassStyle:
        return self.cfg.classes.get(name) or ClassStyle()

    def tone(self, color: str) -> str:
        return mix(color, WHITE, self.cfg.fill.frost) if self.cfg.fill.frost > 0 else color


def _ss_xy(pts: list[Pt], ss: int) -> list[Pt]:
    return [(x * ss, y * ss) for x, y in pts]


@dataclass
class _Mask:
    """A mask over its own bounds only: ``image`` (L) placed at ``x``, ``y`` (in the layer's pixels)."""

    image: Image.Image
    x: int
    y: int


def _local_mask(rings: list[list[Pt]], ss: int) -> _Mask | None:
    """An even-odd mask of a polygon's rings at ``ss``x (each ring flips what it covers), over the rings' bounds."""
    pts = [p for r in rings if len(r) >= 3 for p in r]
    if not pts:
        return None
    x0 = math.floor(min(p[0] for p in pts) * ss) - 1
    y0 = math.floor(min(p[1] for p in pts) * ss) - 1
    x1 = math.ceil(max(p[0] for p in pts) * ss) + 2
    y1 = math.ceil(max(p[1] for p in pts) * ss) + 2
    mask = Image.new("1", (x1 - x0, y1 - y0), 0)
    for ring in rings:
        if len(ring) < 3:
            continue
        one = Image.new("1", mask.size, 0)
        ImageDraw.Draw(one).polygon([(x * ss - x0, y * ss - y0) for x, y in ring], fill=1)
        mask = ImageChops.logical_xor(mask, one)
    return _Mask(mask.convert("L"), x0, y0)


def _paste(target: Image.Image, source: Image.Image | RGBA, m: _Mask) -> None:
    """``source`` (a color, or an image the mask's size) into ``target`` through ``m``, clipped to ``target``."""
    w, h = m.image.size
    cx0, cy0 = max(0, m.x), max(0, m.y)
    cx1, cy1 = min(target.width, m.x + w), min(target.height, m.y + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    crop = (cx0 - m.x, cy0 - m.y, cx1 - m.x, cy1 - m.y)
    mask = m.image.crop(crop)
    src: Image.Image | RGBA = source.crop(crop) if isinstance(source, Image.Image) else source
    target.paste(src, (cx0, cy0, cx1, cy1), mask)


def _reduce(layer: Image.Image, size: tuple[int, int]) -> Image.Image:
    """An ``ss``x RGBA layer reduced to ``size`` by area, on premultiplied alpha (edges anti-aliased; an
    axis-aligned line on the pixel grid stays exactly crisp)."""
    if layer.size == size:
        return layer
    return layer.convert("RGBa").resize(size, Image.Resampling.BOX).convert("RGBA")


def _composite(ctx: _Ctx, layer: Image.Image, opacity: float = 1.0) -> None:
    small = _reduce(layer, ctx.canvas.size)
    if opacity < 1:
        alpha = small.getchannel("A").point(lambda a: _jsround(a * opacity))
        small.putalpha(alpha)
    ctx.canvas.alpha_composite(small)


def _fill_groups(ctx: _Ctx, shapes: list[Shape]) -> None:
    cfg, ss = ctx.cfg, ctx.ss
    size = ctx.canvas.size
    regions = [sh for sh in shapes if sh.kind == "region" and sh.rings]
    if regions and cfg.region.opacity > 0:
        layer = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
        for sh in regions:
            m = _local_mask(sh.rings or [], ss)
            if m is not None:
                _paste(layer, _rgba(ctx.tone(ctx.color(sh.name)), 1), m)
        _composite(ctx, layer, cfg.region.opacity)
    groups: dict[float, list[Shape]] = {}
    for sh in shapes:
        if sh.kind == "polygon":
            op = ctx.own(sh.name).fill
            op = cfg.fill.opacity if op is None else op
        elif sh.kind in ("box", "obb"):
            op = ctx.own(sh.name).fill
            op = cfg.box.fill if op is None else op
        else:
            continue
        if op > 0:
            groups.setdefault(op, []).append(sh)
    for op, members in groups.items():
        layer = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for sh in members:
            fill = _rgba(ctx.tone(ctx.color(sh.name)), 1)
            if sh.kind == "polygon" and sh.rings:
                m = _local_mask(sh.rings, ss)
                if m is not None:
                    _paste(layer, fill, m)
            elif sh.kind == "obb" and sh.corners:
                d.polygon(_ss_xy(sh.corners, ss), fill=fill)
            elif sh.box is not None:
                x, y, w, h = sh.box
                d.rectangle([x * ss, y * ss, (x + w) * ss - 1, (y + h) * ss - 1], fill=fill)
        _composite(ctx, layer, op)


def _box_corners(sh: Shape) -> list[Pt]:
    if sh.corners:
        return sh.corners
    if sh.box is None:
        return []
    x, y, w, h = sh.box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _crisp(v: float, sw: int) -> float:
    """A crisp edge: an odd stroke centred on a pixel, an even one on a pixel boundary."""
    return math.floor(v) + 0.5 if sw % 2 == 1 else float(_jsround(v))


def _rect_ring(
    d: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sw: float,
    ss: int,
    fill: RGBA,
) -> None:
    """A rectangle's stroke ``sw`` px wide centred on its edges, at ``ss``x (exact, square corners)."""
    h = sw / 2
    ox1, oy1, ox2, oy2 = (x1 - h) * ss, (y1 - h) * ss, (x2 + h) * ss, (y2 + h) * ss
    t = sw * ss
    d.rectangle([ox1, oy1, ox2 - 1, oy1 + t - 1], fill=fill)
    d.rectangle([ox1, oy2 - t, ox2 - 1, oy2 - 1], fill=fill)
    d.rectangle([ox1, oy1 + t, ox1 + t - 1, oy2 - t - 1], fill=fill)
    d.rectangle([ox2 - t, oy1 + t, ox2 - 1, oy2 - t - 1], fill=fill)


def _draw_box(ctx: _Ctx, d: ImageDraw.ImageDraw, color: str, sw: int, sh: Shape) -> None:
    if sh.box is None:
        return
    s, ss, key = ctx.s, ctx.ss, ctx.cfg.box.keyline
    x, y, w, h = sh.box
    if ctx.cfg.box.corners == "brackets":
        hw = s.hair
        x1, y1, x2, y2 = _crisp(x, hw), _crisp(y, hw), _crisp(x + w, hw), _crisp(y + h, hw)
        _rect_ring(d, x1, y1, x2, y2, hw, ss, _rgba(color, 0.9))
        bw = sw + 1
        arm = _jsround(min(max(min(w, h) * 0.22, s.em * 0.5), s.em * 1.4))
        left, top = _jsround(x - bw / 2), _jsround(y - bw / 2)
        right, bottom = _jsround(x + w + bw / 2), _jsround(y + h + bw / 2)
        corners = [
            (left, top, 1, 1),
            (right, top, -1, 1),
            (left, bottom, 1, -1),
            (right, bottom, -1, -1),
        ]
        for pass_key in (True, False):
            if pass_key and key is None:
                continue
            fill = _rgba(_key_color(key), key.alpha) if pass_key and key else _rgba(color, 1)
            grow = s.hair if pass_key else 0
            for cx, cy, dx, dy in corners:
                rx = cx if dx > 0 else cx - arm
                ry = cy if dy > 0 else cy - bw
                d.rectangle(
                    [
                        (rx - grow) * ss,
                        (ry - grow) * ss,
                        (rx + arm + grow) * ss - 1,
                        (ry + bw + grow) * ss - 1,
                    ],
                    fill=fill,
                )
                rx = cx if dx > 0 else cx - bw
                ry = cy if dy > 0 else cy - arm
                d.rectangle(
                    [
                        (rx - grow) * ss,
                        (ry - grow) * ss,
                        (rx + bw + grow) * ss - 1,
                        (ry + arm + grow) * ss - 1,
                    ],
                    fill=fill,
                )
        return
    x1, y1, x2, y2 = _crisp(x, sw), _crisp(y, sw), _crisp(x + w, sw), _crisp(y + h, sw)
    if key is not None:
        _rect_ring(d, x1, y1, x2, y2, sw + 2 * s.hair, ss, _rgba(_key_color(key), key.alpha))
    _rect_ring(d, x1, y1, x2, y2, sw, ss, _rgba(color, 1))


def _offset_rect(corners: list[Pt], grow: float) -> list[Pt]:
    """A rotated rectangle's corners moved ``grow`` px out along its own axes (exact miter corners)."""
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    (ax, ay), (bx, by) = corners[0], corners[1]
    ux, uy = bx - ax, by - ay
    lu = math.hypot(ux, uy) or 1.0
    ux, uy = ux / lu, uy / lu
    vx, vy = -uy, ux
    out = []
    for px_, py_ in corners:
        dx, dy = px_ - cx, py_ - cy
        su = 1 if dx * ux + dy * uy > 0 else -1
        sv = 1 if dx * vx + dy * vy > 0 else -1
        out.append((px_ + (su * ux + sv * vx) * grow, py_ + (su * uy + sv * vy) * grow))
    return out


def _ring_stroke(ctx: _Ctx, corners: list[Pt], sw: float, fill: RGBA) -> None:
    """A rotated rectangle's stroke ``sw`` px wide centred on its edges: the band between two offsets (exact miter
    corners), over the stroke's own bounds."""
    if ctx.layer is None:
        return
    m = _local_mask([_offset_rect(corners, sw / 2), _offset_rect(corners, -sw / 2)], ctx.ss)
    if m is None:
        return
    solid = Image.new("RGBA", m.image.size, (*fill[:3], 255))
    solid.putalpha(m.image.point(lambda a: a * fill[3] // 255))
    w, h = m.image.size
    cx0, cy0 = max(0, m.x), max(0, m.y)
    cx1, cy1 = min(ctx.layer.width, m.x + w), min(ctx.layer.height, m.y + h)
    if cx1 > cx0 and cy1 > cy0:
        part = solid.crop((cx0 - m.x, cy0 - m.y, cx1 - m.x, cy1 - m.y))
        ctx.layer.alpha_composite(part, (cx0, cy0))


def _draw_obb(ctx: _Ctx, d: ImageDraw.ImageDraw, color: str, sw: int, corners: list[Pt]) -> None:
    s, key = ctx.s, ctx.cfg.box.keyline
    if ctx.cfg.box.corners == "brackets":
        _ring_stroke(ctx, corners, s.hair, _rgba(color, 0.9))
        for i in range(4):
            ax, ay = corners[i]
            px_, py_ = corners[(i + 3) % 4]
            nx, ny = corners[(i + 1) % 4]
            ln = math.hypot(nx - ax, ny - ay) or 1.0
            lp = math.hypot(px_ - ax, py_ - ay) or 1.0
            arm = min(max(min(ln, lp) * 0.22, s.em * 0.4), s.em * 1.2)
            pts = [
                (ax + (px_ - ax) / lp * arm, ay + (py_ - ay) / lp * arm),
                (ax, ay),
                (ax + (nx - ax) / ln * arm, ay + (ny - ay) / ln * arm),
            ]
            d.line(
                _ss_xy(pts, ctx.ss), fill=_rgba(color, 1), width=(sw + 1) * ctx.ss, joint="curve"
            )
        return
    if key is not None:
        _ring_stroke(ctx, corners, sw + 2 * s.hair, _rgba(_key_color(key), key.alpha))
    _ring_stroke(ctx, corners, sw, _rgba(color, 1))


def _stroke(
    d: ImageDraw.ImageDraw, pts: list[Pt], closed: bool, width: float, fill: RGBA, ss: int
) -> None:
    if len(pts) < 2 or width <= 0:
        return
    path = _ss_xy(pts + pts[:2] if closed else pts, ss)
    d.line(path, fill=fill, width=max(1, _jsround(width * ss)), joint="curve")
    if not closed:
        r = width * ss / 2
        for x, y in (path[0], path[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _draw_point(ctx: _Ctx, d: ImageDraw.ImageDraw, color: str, p: Pt, r: float) -> None:
    if r <= 0:
        return
    s, ss = ctx.s, ctx.ss
    key = ctx.cfg.box.keyline or ctx.cfg.outline.keyline
    ring = max(1.5, s.hair * 1.5)
    x, y = p[0] * ss, p[1] * ss

    def disk(fill: RGBA, rr: float) -> None:
        q = rr * ss
        d.ellipse([x - q, y - q, x + q, y + q], fill=fill)

    def circle(fill: RGBA, rr: float, w: float) -> None:
        q = rr * ss
        d.ellipse(
            [x - q - w * ss / 2, y - q - w * ss / 2, x + q + w * ss / 2, y + q + w * ss / 2],
            outline=fill,
            width=max(1, _jsround(w * ss)),
        )

    style = ctx.cfg.point.style
    if style == "paper":
        disk(_rgba(PAPER, 0.92), r)
        circle(_rgba(color, 1), r - ring / 2, ring)
    elif style == "glass":
        disk(_rgba(INK, 0.78), r)
        circle(_rgba(color, 1), r - ring / 2, ring)
    elif style == "solid":
        if key is not None:
            disk(_rgba(_key_color(key), key.alpha), r + s.hair)
        disk(_rgba(color, 1), r)
    elif style == "ring":
        if key is not None:
            circle(_rgba(_key_color(key), key.alpha), r - ring / 2, ring + 2 * s.hair)
        circle(_rgba(color, 1), r - ring / 2, ring)
    else:
        disk(_rgba(color, 1), r)


def _strokes(ctx: _Ctx, shapes: list[Shape], bones: list[Bone]) -> None:
    cfg, s, ss = ctx.cfg, ctx.s, ctx.ss
    size = ctx.canvas.size
    ctx.layer = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(ctx.layer, "RGBA")
    line_key = cfg.box.keyline or cfg.outline.keyline
    # Bones, under everything else.
    bw = s.px(cfg.bone.width)
    for bone in bones:
        width = s.px(cfg.bone.width * (ctx.own(bone.name).width or 1)) if bw else 0
        if width <= 0:
            continue
        col = (
            _instance_color(bone.instance)
            if cfg.bone.color == "instance" and bone.instance
            else ctx.color(bone.name)
        )
        if cfg.bone.keyline and line_key is not None:
            _stroke(
                d,
                [bone.a, bone.b],
                False,
                width + 2 * s.hair,
                _rgba(_key_color(line_key), line_key.alpha),
                ss,
            )
        _stroke(d, [bone.a, bone.b], False, width, _rgba(col, 1), ss)
    for sh in shapes:
        if sh.kind == "region" and sh.rings and cfg.region.outline > 0:
            for ring in sh.rings:
                _stroke(
                    d, ring, True, s.px(cfg.region.outline), _rgba(ctx.color(sh.name), 0.85), ss
                )
    for sh in shapes:
        if sh.kind == "polygon" and sh.rings and cfg.outline.width > 0:
            ow = s.px(cfg.outline.width * (ctx.own(sh.name).width or 1))
            for ring in sh.rings:
                if cfg.outline.keyline is not None:
                    _stroke(
                        d,
                        ring,
                        True,
                        ow + 2 * s.hair,
                        _rgba(_key_color(cfg.outline.keyline), cfg.outline.keyline.alpha),
                        ss,
                    )
                _stroke(d, ring, True, ow, _rgba(ctx.color(sh.name), 1), ss)
    for sh in shapes:
        if sh.kind == "box" and cfg.box.width > 0:
            _draw_box(
                ctx, d, ctx.color(sh.name), s.px(cfg.box.width * (ctx.own(sh.name).width or 1)), sh
            )
    for sh in shapes:
        if sh.kind == "obb" and sh.corners and cfg.box.width > 0:
            _draw_obb(
                ctx,
                d,
                ctx.color(sh.name),
                s.px(cfg.box.width * (ctx.own(sh.name).width or 1)),
                sh.corners,
            )
    for sh in shapes:
        if sh.kind == "polyline" and sh.path and len(sh.path) >= 2:
            pw = s.px(cfg.polyline.width * (ctx.own(sh.name).width or 1))
            col = ctx.color(sh.name)
            if line_key is not None:
                _stroke(
                    d,
                    sh.path,
                    False,
                    pw + 2 * s.hair,
                    _rgba(_key_color(line_key), line_key.alpha),
                    ss,
                )
            _stroke(d, sh.path, False, pw, _rgba(col, 1), ss)
            if cfg.polyline.ends:
                for p in (sh.path[0], sh.path[-1]):
                    _draw_point(ctx, d, col, p, s.rad(cfg.point.radius) * 0.85)
    for sh in shapes:
        if sh.kind == "keypoint" and sh.point is not None:
            col = (
                _instance_color(sh.instance)
                if cfg.bone.color == "instance" and sh.instance
                else ctx.color(sh.name)
            )
            _draw_point(ctx, d, col, sh.point, s.rad(cfg.point.radius))
    _composite(ctx, ctx.layer)
    ctx.layer = None


# ── Labels ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Paint:
    glass: str
    alpha: float
    name: str
    value: str


def _paint_for(cfg: Look, color: str) -> _Paint:
    lab = cfg.label
    base = mix(INK if lab.glass == "smoke" else PAPER, color, lab.tint)
    if lab.ink == "shade":
        name = shade_on(color, base)
        return _Paint(base, lab.alpha, name, mix(name, base, 0.3))
    glass, ink = label_ink(base)
    return _Paint(glass, lab.alpha, ink, mix(ink, glass, 0.3))


@lru_cache(maxsize=512)
def _chip_mask(w: int, h: int, r: int, corners: tuple[bool, bool, bool, bool]) -> Image.Image:
    """A label's rounded shape, anti-aliased (drawn at 4x, reduced by area)."""
    k = 4
    big = Image.new("L", (w * k, h * k), 0)
    ImageDraw.Draw(big).rounded_rectangle(
        [0, 0, w * k - 1, h * k - 1], radius=r * k, fill=255, corners=corners
    )
    return big.resize((w, h), Image.Resampling.BOX)


def _glass(
    ctx: _Ctx,
    paint: _Paint,
    x: int,
    y: int,
    w: int,
    h: int,
    r: int,
    corners: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> None:
    if w <= 0 or h <= 0:
        return
    mask = _chip_mask(w, h, r, corners)
    box = (x, y, x + w, y + h)
    tint = Image.new("RGBA", (w, h), _rgba(paint.glass, paint.alpha))
    ctx.canvas.paste(Image.alpha_composite(ctx.canvas.crop(box), tint), (x, y), mask)


Where = Anchor9 | Literal["corner"]


@dataclass
class _Words:
    name: str
    value: str


@dataclass
class _Layout:
    rect: Rect
    paint: Callable[[], None]


@dataclass
class _Job:
    priority: float
    make: Callable[[list[Rect]], _Layout | None]


def _candidates(ctx: _Ctx, a: Rect, w: int, h: int, where: Where, stroke: int) -> list[Rect]:
    s = ctx.s

    def clamp(x: float, y: float) -> Rect:
        return Rect(
            min(max(0, _jsround(x)), max(0, s.W - w)),
            min(max(0, _jsround(y)), max(0, s.H - h)),
            w,
            h,
        )

    if where == "corner":
        return [clamp(a.x, a.y)]
    vert, horiz = where[0], where[1]
    if vert == "c" and horiz == "l":
        x = a.x + stroke if a.x - s.gap - w < 0 else a.x - s.gap - w
    elif vert == "c" and horiz == "r":
        x = a.x + a.w - stroke - w if a.x + a.w + s.gap + w > s.W else a.x + a.w + s.gap
    else:
        x = a.x if horiz == "l" else a.x + a.w - w if horiz == "r" else a.x + a.w / 2 - w / 2
    style = ctx.cfg.label.style
    above = (
        a.y - h + stroke / 2
        if style == "tab"
        else a.y - s.gap - h - s.em
        if style == "leader"
        else a.y - s.gap - h
    )
    below = a.y + a.h - stroke / 2 if style == "tab" else a.y + a.h + s.gap
    inside_top = a.y + stroke
    inside_bottom = a.y + a.h - stroke - h
    if vert == "t":
        first = inside_top if style == "inset" or above < 0 else above
    elif vert == "b":
        first = inside_bottom if style == "inset" or below + h > s.H else below
    else:
        first = a.y + a.h / 2 - h / 2
    out = [clamp(x, first)]
    if ctx.cfg.label.overlap == "shift":
        out += [
            clamp(x, yy)
            for yy in (
                inside_top,
                below,
                above - (h + s.gap),
                inside_bottom,
                above - 2 * (h + s.gap),
            )
        ]
    return out


def _choose(ctx: _Ctx, cands: list[Rect], placed: list[Rect]) -> Rect | None:
    mode = ctx.cfg.label.overlap
    if mode == "draw":
        return cands[0]
    for r in cands:
        if not any(_overlaps(r, p) for p in placed):
            return r
    return cands[0] if mode == "shift" else None


def _label_job(
    ctx: _Ctx, color: str, words: _Words, a: Rect, where: Where, stroke: int, priority: float
) -> _Job:
    def make(placed: list[Rect]) -> _Layout | None:
        if not words.name and not words.value:
            return None
        cfg, lab = ctx.cfg, ctx.cfg.label
        em = ctx.s.badge_em if where == "corner" else ctx.s.em
        paint = _paint_for(cfg, color)
        h = _jsround(em * 1.5)
        pad = _jsround(em * 0.5)
        f = _fonts(em, cfg)
        w_name = _width(f.name, words.name)
        w_val = _width(f.value, words.value)
        gap_v = _jsround(em * 0.36) if words.name and words.value else 0
        split = lab.style == "split" and bool(words.name) and bool(words.value)
        name_w = _jsround(pad + w_name + pad * 0.8) if split else 0
        w = (
            _jsround(name_w + pad * 0.8 + w_val + pad)
            if split
            else _jsround(pad + w_name + gap_v + w_val + pad)
        )
        rect = _choose(ctx, _candidates(ctx, a, w, h, where, stroke), placed)
        if rect is None:
            return None
        x, y = int(rect.x), int(rect.y)
        base = y + _jsround(h / 2 + f.cap / 2)
        rad = _jsround(lab.radius * em)

        def paint_it() -> None:
            d = ImageDraw.Draw(ctx.canvas)
            if lab.style == "text" and where != "corner":
                tx = x + pad
                halo = max(2, _jsround(em * 0.11))
                if words.name:
                    d.text(
                        (tx, base),
                        words.name,
                        font=f.name,
                        fill=mix(color, WHITE, 0.28),
                        anchor="ls",
                        stroke_width=halo,
                        stroke_fill=_rgba(INK, 0.86),
                    )
                    tx += _jsround(w_name + gap_v)
                if words.value:
                    d.text(
                        (tx, base),
                        words.value,
                        font=f.value,
                        fill=INK_INVERSE,
                        anchor="ls",
                        stroke_width=halo,
                        stroke_fill=_rgba(INK, 0.86),
                    )
                return
            if lab.style == "leader" and where != "corner" and where[0] == "t":
                lx = _jsround(min(max(a.x + a.w / 2, x + rad + 1), x + w - rad - 1))
                d.line([(lx, y + h), (lx, a.y)], fill=color, width=ctx.s.hair)
                dot = max(2, ctx.s.hair * 2)
                d.ellipse([lx - dot, a.y - dot, lx + dot, a.y + dot], fill=color)
            if split:
                name_glass, name_ink = label_ink(mix(color, PAPER, 0.08))
                name_paint = _Paint(name_glass, 1.0, name_ink, "")
                _glass(ctx, name_paint, x, y, name_w, h, rad, (True, False, False, True))
                _glass(ctx, paint, x + name_w, y, w - name_w, h, rad, (False, True, True, False))
                d.text((x + pad, base), words.name, font=f.name, fill=name_paint.name, anchor="ls")
                d.text(
                    (x + name_w + _jsround(pad * 0.8), base),
                    words.value,
                    font=f.value,
                    fill=paint.name,
                    anchor="ls",
                )
                return
            tab_top = lab.style == "tab" and where != "corner" and where[0] == "t"
            tab_bottom = lab.style == "tab" and where != "corner" and where[0] == "b"
            corners = (
                (True, True, False, False)
                if tab_top
                else (False, False, True, True)
                if tab_bottom
                else (True, True, True, True)
            )
            _glass(ctx, paint, x, y, w, h, rad, corners)
            tx = x + pad
            if words.name:
                d.text((tx, base), words.name, font=f.name, fill=paint.name, anchor="ls")
                tx += _jsround(w_name + gap_v)
            if words.value:
                d.text((tx, base), words.value, font=f.value, fill=paint.value, anchor="ls")

        return _Layout(rect, paint_it)

    return _Job(priority, make)


def _words(cfg: Look, name: str, confidence: float | None, track: int | None) -> _Words:
    p = cfg.label.parts
    vals = []
    if p.track and track is not None:
        vals.append(str(track))
    if p.confidence and confidence is not None:
        vals.append(f"{_jsround(confidence * 100)}%")
    return _Words(name if p.name else "", " ".join(vals))


def _labels(ctx: _Ctx, shapes: list[Shape], badges: list[Badge]) -> None:
    cfg, s = ctx.cfg, ctx.s
    lab = cfg.label
    jobs: list[_Job] = []
    stroke = s.px(cfg.box.width)
    if lab.show:
        for sh in shapes:
            own = ctx.own(sh.name)
            if own.label is False:
                continue
            area = sh.kind in ("polygon", "region")
            at: Where = own.position or (lab.mask_position if area else lab.position)
            if area and sh.rings:
                b = Rect(*sh.box) if sh.box else _bounds([p for r in sh.rings for p in r])
                if at == "cc":
                    cx, cy = shape_center(sh.rings, 1.0)
                    anchor = Rect(cx, cy, 0, 0)
                elif at == "tc":
                    anchor = mask_top(sh.rings, b)
                else:
                    anchor = b
            elif sh.kind == "obb" and sh.corners:
                anchor = _bounds(sh.corners)
            elif sh.kind == "polyline" and sh.path and len(sh.path) >= 2:
                anchor = _bounds(sh.path)
            elif sh.kind == "keypoint" and sh.point is not None:
                r = s.rad(cfg.point.radius)
                anchor = Rect(sh.point[0] - r, sh.point[1] - r, 2 * r, 2 * r)
            elif sh.box is not None:
                anchor = Rect(*sh.box)
            else:
                continue
            color = (
                _instance_color(sh.instance)
                if sh.kind == "keypoint" and cfg.bone.color == "instance" and sh.instance
                else ctx.color(sh.name)
            )
            prio = sh.confidence if sh.confidence is not None else 0.5
            jobs.append(
                _label_job(
                    ctx,
                    color,
                    _words(cfg, sh.name, sh.confidence, sh.track),
                    anchor,
                    at,
                    stroke,
                    prio,
                )
            )
        # The classifier badges, stacked in their corner.
        if badges:
            badge_cfg = cfg.model_copy(
                update={
                    "label": lab.model_copy(
                        update={
                            "style": "text" if lab.style == "text" else "chip",
                            "overlap": "draw",
                        }
                    )
                }
            )
            bctx = _Ctx(ctx.canvas, ctx.photo, badge_cfg, s, ctx.stored, ctx.ss)
            em = s.badge_em
            step = _jsround(em * 1.5) + _jsround(em * 0.33)
            f = _fonts(em, cfg)
            for i, bd in enumerate(badges):
                words = _words(cfg, bd.name, bd.confidence, None)
                w = _jsround(
                    em
                    + _width(f.name, words.name)
                    + _jsround(em * 0.36)
                    + _width(f.value, words.value)
                )
                corner = cfg.badge.corner
                y = (
                    s.inset + i * step
                    if corner[0] == "t"
                    else s.H - s.inset - _jsround(em * 1.5) - i * step
                )
                x = s.inset if corner[1] == "l" else s.W - s.inset - w
                jobs.append(
                    _label_job(bctx, bctx.color(bd.name), words, Rect(x, y, 0, 0), "corner", 0, 3)
                )
    jobs.sort(key=lambda j: -j.priority)
    placed: list[Rect] = []
    layouts: list[_Layout] = []
    for job in jobs:
        lay = job.make(placed)
        if lay is None:
            continue
        placed.append(lay.rect)
        layouts.append(lay)
    for lay in reversed(layouts):
        lay.paint()


def render(
    image: Image.Image,
    shapes: list[Shape],
    *,
    cfg: Look,
    bones: list[Bone] | None = None,
    badges: list[Badge] | None = None,
    stored_colors: dict[str, str] | None = None,
    font_size: float | None = None,
    stroke: int | None = None,
) -> Image.Image:
    """Draw ``shapes`` (and a pose's ``bones``, a classifier's ``badges``) on a copy of ``image`` in THE look."""
    photo = image.convert("RGB")
    w, h = photo.size
    s = sizes_for(w, h, cfg, font_size, stroke)
    ss = 2 if max(w, h) <= 2560 else 1
    ctx = _Ctx(canvas=photo.convert("RGBA"), photo=photo, cfg=cfg, s=s, stored=stored_colors, ss=ss)
    _fill_groups(ctx, shapes)
    _strokes(ctx, shapes, bones or [])
    _labels(ctx, shapes, badges or [])
    return ctx.canvas.convert("RGB")
