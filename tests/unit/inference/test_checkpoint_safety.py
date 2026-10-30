"""A checkpoint must not be able to execute code when it is loaded.

``torch.load(weights_only=False)`` runs the file's own ``__reduce__``. The loader
used to call it as an automatic fallback whenever the safe unpickler refused, and
logged that at ``debug`` level, which is off by default. So an untrusted
checkpoint executed silently.

That matters specifically because models can be FORKED between organizations, so
a ``.pth`` reaching this code is not necessarily one the caller produced.

These tests build a REAL malicious checkpoint whose payload writes a file, and
assert the file is never written. A test that only asserted "an exception was
raised" would pass against a loader that executed the payload and then failed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pictograph.exceptions import UnsafeCheckpointError  # noqa: E402
from pictograph.inference._torch import _load_checkpoint  # noqa: E402


class _Payload:
    """Pickles to a call that writes a marker file. This is what an attacker does."""

    def __reduce__(self) -> tuple:
        marker = Path(_Payload.marker)  # type: ignore[attr-defined]
        return (Path.write_text, (marker, "executed"))


def _write_malicious_checkpoint(path: Path, marker: Path) -> None:
    _Payload.marker = str(marker)  # type: ignore[attr-defined]
    with path.open("wb") as fh:
        pickle.dump({"model": _Payload()}, fh)


def test_a_malicious_checkpoint_does_not_execute(tmp_path: Path) -> None:
    """The payload must not run, and the loader must refuse rather than fall back."""
    weights = tmp_path / "evil.pth"
    marker = tmp_path / "pwned.txt"
    _write_malicious_checkpoint(weights, marker)

    with pytest.raises(UnsafeCheckpointError):
        _load_checkpoint(weights)

    assert not marker.exists(), (
        "the checkpoint's embedded code EXECUTED - the safe unpickler was bypassed"
    )


def test_the_unsafe_path_is_opt_in_only(tmp_path: Path) -> None:
    """allow_unsafe_pickle=True is the ONLY way to reach the executing loader.

    Asserted from the opposite direction: with the flag set, the payload DOES run.
    That is what proves the default is carrying the protection, rather than the
    payload being inert for some unrelated reason.
    """
    weights = tmp_path / "evil.pth"
    marker = tmp_path / "pwned-optin.txt"
    _write_malicious_checkpoint(weights, marker)

    try:
        _load_checkpoint(weights, allow_unsafe_pickle=True)
    except Exception:
        pass

    assert marker.exists(), (
        "the opt-in path did not reach the full unpickler, so the default-path "
        "assertion above proves nothing"
    )


def test_our_own_checkpoint_shape_loads_on_the_safe_path(tmp_path: Path) -> None:
    """An argparse.Namespace payload, which RF-DETR carries, must still load.

    This is the regression guard on the fix itself: allowlisting is only correct
    if it keeps real Pictograph checkpoints working. If this fails, the allowlist
    is wrong and someone will be tempted to reinstate the blanket fallback.
    """
    import argparse

    weights = tmp_path / "good.pth"
    torch.save(
        {"model": {"weight": torch.zeros(2)}, "args": argparse.Namespace(epochs=3)},
        weights,
    )

    loaded = _load_checkpoint(weights)
    assert loaded["args"].epochs == 3


def _rename_numpy_scalar(src: Path, dst: Path, old: bytes, new: bytes) -> None:
    """Re-zip a ``torch.save`` file with its pickle naming numpy's scalar ``new`` instead of ``old``.

    ``torch.save`` writes pickle protocol 2, where a global is the TEXT ``c<module>\\n<name>\\n``,
    so the module can be renamed by bytes - which is how a checkpoint saved under the OTHER numpy
    (1 vs 2) is produced here without installing it.
    """
    import zipfile

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith("data.pkl"):
                assert old in data, data
                data = data.replace(old, new)
            zout.writestr(info.filename, data)


@pytest.mark.parametrize("saved_by", ["numpy 2", "numpy 1"])
def test_a_checkpoint_carrying_numpy_numbers_loads_on_the_safe_path(
    tmp_path: Path, saved_by: str
) -> None:
    """B534: an older YOLOX checkpoint saved ``best_ap`` / ``curr_ap`` as numpy float64, so its pickle names
    numpy's scalar reconstructor, ``numpy.dtype`` and the dtype's class - and the safe loader refused every
    such ``.pth`` (the Use-Model snippet for those models raised ``UnsafeCheckpointError``, found by the
    snippet harness 2026-10-27). They build numbers and execute nothing, so they are allowlisted; the
    scalar's name depends on the numpy that SAVED the file, and our training images run numpy 1."""
    np = pytest.importorskip("numpy")

    weights = tmp_path / "yolox.pth"
    torch.save(
        {
            "start_epoch": 20,
            "model": {"w": torch.zeros(2)},
            "best_ap": np.float64(0.42),
            "curr_ap": np.float32(0.4),
        },
        weights,
    )
    if saved_by == "numpy 1":
        renamed = tmp_path / "yolox-numpy1.pth"
        _rename_numpy_scalar(
            weights, renamed, b"cnumpy._core.multiarray\nscalar", b"cnumpy.core.multiarray\nscalar"
        )
        weights = renamed

    loaded = _load_checkpoint(weights)
    assert float(loaded["best_ap"]) == pytest.approx(0.42)
    assert float(loaded["curr_ap"]) == pytest.approx(0.4)
    assert loaded["start_epoch"] == 20


def test_a_numpy_object_array_is_still_refused(tmp_path: Path) -> None:
    """Allowlisting numpy NUMBERS opened nothing else: an object array needs numpy's array reconstructor,
    which stays refused - and its payload never runs."""
    np = pytest.importorskip("numpy")

    weights = tmp_path / "objects.pth"
    marker = tmp_path / "pwned.txt"
    _Payload.marker = str(marker)  # type: ignore[attr-defined]
    payload = np.empty(1, dtype=object)
    payload[0] = _Payload()
    with weights.open("wb") as fh:
        pickle.dump({"model": payload}, fh)

    with pytest.raises(UnsafeCheckpointError):
        _load_checkpoint(weights)
    assert not marker.exists(), "an object array's payload EXECUTED"


def test_only_the_shared_loader_may_read_a_checkpoint() -> None:
    """Repo-wide: no `torch.load` outside `_safe_load` may skip the safe unpickler.

    This is the check that mattered. When it was scoped to one module it passed
    while FOUR unsafe loads sat in the vendored RF-DETR tree (builder.py,
    utilities/state_dict.py, models/weights.py x2), each reachable when loading a
    forked model. Vendored code is our code.
    """
    import ast

    import pictograph

    root = Path(pictograph.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "_safe_load.py":
            continue
        # Parse rather than grep: a docstring that DISCUSSES weights_only=False is
        # not a call, and a text match cannot tell the difference.
        for node in ast.walk(ast.parse(py.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "load"):
                continue
            if not (isinstance(fn.value, ast.Name) and fn.value.id == "torch"):
                continue
            safe = any(
                kw.arg == "weights_only" and getattr(kw.value, "value", None) is True
                for kw in node.keywords
            )
            if not safe:
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")

    assert offenders == [], "torch.load called outside the shared safe loader:\n  " + "\n  ".join(
        offenders
    )
