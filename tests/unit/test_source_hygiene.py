"""Source-hygiene gate for the PUBLIC SDK tree.

The owner's requirement (2026-08-01): *"there should never be emojis in code ...
this is to be public facing in the sdk"*. A one-time scrub does not hold - nothing
stops the next emoji or the next leaked internal path. This test is that gate, and
it runs inside ``pytest`` so it fires on every change, next to ruff and mypy.

Two things are enforced over the installable package (``src/pictograph``) and the
published manifest (``pyproject.toml``):

1. **No emoji / pictographic characters.** Deliberately NARROW: the ASCII arrow
   ``->``, the middle dot, the arrow ``\\u2192``, the join glyph ``\\u22c8`` and
   ``!=`` (``\\u2260``) are typography, not emoji, and are NOT flagged.
   Over-blocking gets a gate switched off, so the ranges below are the
   emoji/pictograph blocks only.
2. **No internal repository paths** (``pictograph-core/`` / ``pictograph-app/`` /
   ``pictograph-backend/``). The published wheel is read by anyone; those paths
   leak our repository layout and infrastructure to the world. The SDK's own
   public names (``pictograph-sdk/``, ``pictograph.io``,
   the ``pictograph-cv`` skill) are NOT internal and must stay allowed.

The self-tests below prove the detector actually SEES a planted emoji and a planted
path, and that it ignores typography and public names - so a green scan means the
gate looked, not that it is blind. ``test_source_scan_covers_known_files`` guards
against an empty glob passing silently.
"""

from __future__ import annotations

from pathlib import Path

# Emoji / pictographic Unicode blocks. These EXCLUDE the arrows/math/punctuation
# below U+2600 (so ``->`` U+2192, ``.`` U+22C8, ``!=`` U+2260, em/en dashes and the
# ellipsis are all left alone) as well as plain ASCII ``->``.
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, symbols, cards
    (0x1F1E6, 0x1F1FF),  # regional indicators (flags)
    (0x2600, 0x26FF),  # misc symbols (warning, gear, weather, ...)
    (0x2700, 0x27BF),  # dingbats (check/ballot marks, decorative arrows, ...)
    (0x2B00, 0x2BFF),  # misc symbols & arrows (star, block arrows, ...)
    (0xFE00, 0xFE0F),  # variation selectors (emoji presentation)
)

# No trailing slash. The tokens used to carry one, which meant the bare
# spellings - "pictograph-backend/modal" written as Path(...) / "pictograph-backend"
# / "modal", or a prose "see pictograph-core" - walked straight past the gate.
# The bucket name "pictograph-app" has no slash either.
_INTERNAL_PATH_TOKENS: tuple[str, ...] = (
    "pictograph-core",
    "pictograph-app",
    "pictograph-backend",
)

# Binary / non-source extensions we must not try to decode as text.
_BINARY_EXTS = frozenset(
    {
        ".woff2",
        ".woff",
        ".ttf",
        ".otf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".gz",
        ".zip",
        ".pt",
        ".onnx",
        ".pyc",
        ".so",
        ".dylib",
    }
)
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".mypy_cache", ".ruff_cache"})


def _find_emoji(text: str) -> list[str]:
    """Return the pictographic characters in ``text`` (empty when clean)."""
    hits = []
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES):
            hits.append(ch)
    return hits


def _find_internal_paths(text: str) -> list[str]:
    """Return the internal-path tokens present in ``text`` (empty when clean)."""
    return [tok for tok in _INTERNAL_PATH_TOKENS if tok in text]


def _sdk_source_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "src" / "pictograph"
    assert root.is_dir(), f"SDK source root not found at {root}"
    return root


def _iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _BINARY_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield path, text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _gated_files():
    """The files this gate scans: the installable package plus the published
    manifest. ``pyproject.toml`` ships in the sdist and the public repo, and a
    manifest has no legitimate need for an emoji or an internal repo path."""
    yield from _iter_text_files(_sdk_source_root())
    pyproject = _repo_root() / "pyproject.toml"
    if pyproject.is_file():
        yield pyproject, pyproject.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Self-tests: prove the detector SEES, and that it does not over-block.
# Emoji are built with chr() so no literal emoji enters this file either.
# --------------------------------------------------------------------------- #


def test_detector_sees_emoji_and_ignores_typography() -> None:
    rocket = chr(0x1F680)  # rocket
    warning = chr(0x26A0) + chr(0xFE0F)  # warning sign + VS16
    check = chr(0x2713)  # check mark (dingbat)
    star = chr(0x2605)  # black star

    assert _find_emoji(f"launch {rocket} now") == [rocket]
    assert _find_emoji(f"heads up {warning}") == [chr(0x26A0), chr(0xFE0F)]
    assert _find_emoji(check) == [check]
    assert _find_emoji(star) == [star]

    # Typography that MUST NOT be flagged (this is what keeps the gate usable).
    # Built with chr() so no literal em/en dash enters the repo.
    typography = (
        "a -> b",  # ASCII arrow
        f"x {chr(0x00B7)} y",  # middle dot
        f"a {chr(0x2192)} b",  # rightwards arrow
        f"m {chr(0x22C8)} n",  # bowtie / join
        f"p {chr(0x2260)} q",  # not equal
        f"dash {chr(0x2014)} here",  # em dash
        f"en {chr(0x2013)} dash",  # en dash
        f"e{chr(0x2026)}",  # ellipsis
    )
    for ok in typography:
        assert _find_emoji(ok) == [], f"false positive on {ok!r}"


def test_detector_sees_internal_paths_and_ignores_public_names() -> None:
    assert _find_internal_paths("see pictograph-core/routes/x.py") == ["pictograph-core"]
    assert _find_internal_paths("gs://pictograph-app/models/y") == ["pictograph-app"]
    assert _find_internal_paths("in pictograph-backend/modal/z.py") == ["pictograph-backend"]

    # The bare spellings, with no trailing slash. These are what the tokens used
    # to miss: a path assembled a segment at a time, and plain prose. Nine test
    # modules reconstructed the private tree this way, past a green gate.
    assert _find_internal_paths('Path(x) / "pictograph-backend" / "modal"') == [
        "pictograph-backend"
    ]
    assert _find_internal_paths("documented in pictograph-core") == ["pictograph-core"]

    # Public / non-internal names MUST stay allowed.
    for ok in (
        "github.com/pictograph-io/pictograph-sdk",
        "https://pictograph.io/docs",
        "skills/pictograph-cv/SKILL.md",
        "import pictograph",
        "the pictograph package",
    ):
        assert _find_internal_paths(ok) == [], f"false positive on {ok!r}"


# --------------------------------------------------------------------------- #
# The real gate: scan the shipped SDK tree.
# --------------------------------------------------------------------------- #


def test_source_scan_covers_known_files() -> None:
    """Guard: a broken glob that scans nothing must not pass silently."""
    root = _repo_root()
    scanned = {p.relative_to(root).as_posix() for p, _ in _gated_files()}
    # Files that definitely exist in every build of the package + the manifest.
    for expected in (
        "src/pictograph/client.py",
        "src/pictograph/cli/_format.py",
        "src/pictograph/inference/_yolox/NOTICE",
        "pyproject.toml",
    ):
        assert expected in scanned, f"hygiene scan missed {expected}; glob is broken"
    assert len(scanned) > 50, f"only scanned {len(scanned)} files; glob is broken"


def test_no_emoji_in_shipped_sdk_source() -> None:
    root = _repo_root()
    offenders: list[str] = []
    for path, text in _gated_files():
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            chars = _find_emoji(line)
            if chars:
                codes = ", ".join(f"U+{ord(c):04X}" for c in chars)
                offenders.append(f"{rel}:{lineno} [{codes}] {line.strip()[:70]}")
    assert not offenders, "Emoji in the public SDK source:\n" + "\n".join(offenders)


def test_no_internal_paths_in_shipped_sdk_source() -> None:
    root = _repo_root()
    offenders: list[str] = []
    for path, text in _gated_files():
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            toks = _find_internal_paths(line)
            if toks:
                offenders.append(f"{rel}:{lineno} [{', '.join(toks)}] {line.strip()[:70]}")
    assert not offenders, "Internal repo paths in the public SDK source:\n" + "\n".join(offenders)
