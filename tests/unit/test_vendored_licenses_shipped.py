"""Every vendored package's LICENSE and NOTICE ships in the wheel AND the sdist.

The SDK vendors Apache-2.0 code (RF-DETR, YOLOX, anomalib's PatchCore and EfficientAD), and Apache-2.0 requires the
license and the notice to travel with it. Hatchling ships a package's ``.py`` files by its glob and nothing else, so
each LICENSE / NOTICE must be named in both ``artifacts`` lists - and the anomaly models' were not, until this test
(found before 1.69.217 was published, 2026-11-25).
"""

from __future__ import annotations

import sys
from pathlib import Path

# tomllib is stdlib in 3.11+; the SDK floor is 3.10, where tomli (the [cli] back-compat pin) stands in
# (test_rfdetr_vendored.py's pattern).
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the 3.10 back-compat path
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_every_vendored_license_and_notice_is_an_artifact_of_both_builds() -> None:
    targets = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "targets"
    ]
    wheel, sdist = set(targets["wheel"]["artifacts"]), set(targets["sdist"]["artifacts"])
    files = sorted(
        p.relative_to(ROOT).as_posix()
        for d in (ROOT / "src" / "pictograph" / "inference").iterdir()
        if d.is_dir() and d.name.startswith("_")
        for p in d.iterdir()
        if p.name in ("LICENSE", "NOTICE")
    )
    assert len(files) >= 6, f"found only {files} - the scan is wrong"
    missing = [
        (f, build)
        for f in files
        for build, listed in (("wheel", wheel), ("sdist", sdist))
        if f not in listed
    ]
    assert not missing, f"vendored license files the builds would not ship: {missing}"
