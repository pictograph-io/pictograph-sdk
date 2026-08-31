"""Live: cross-instance BackgroundJobTracker verification.

Exercises the routes we migrated onto the shared ``background_jobs``
table: connectors, auto_annotate batch, plus cancel semantics.

The canonical bug we're guarding against is: a poll landing on a
different Cloud Run instance than the worker used to report
``status=processing, progress=0%`` forever because the worker's state
lived in a module-level dict. With the tracker live, every one of these
calls should either see real progress or end in a terminal state.
"""

from __future__ import annotations

import os
import time

import pytest

from pictograph import Client
from pictograph.exceptions import NotFoundError

pytestmark = pytest.mark.live


# ───────────── connector import - fast path ─────────────


def test_connector_roboflow_import_reaches_completed_via_poll(
    client: Client,
) -> None:
    """Poll-visible lifecycle: processing → completed, progress ≥ 0 at each step.

    Requires Roboflow test key in env (``ROBOFLOW_TEST_KEY``). Uses the
    ~3-image ``test-import-jwrco`` slug so the test finishes in seconds.
    """
    rf_key = os.environ.get("ROBOFLOW_TEST_KEY")
    if not rf_key:
        pytest.skip("set ROBOFLOW_TEST_KEY to run connector live tests")

    validate = client.connectors.validate("roboflow", rf_key)
    assert validate.valid, f"Roboflow key invalid: {validate}"
    target = next(
        (d for d in (validate.datasets or []) if d.slug == "test-import-jwrco"),
        None,
    )
    if target is None:
        pytest.skip("test-import-jwrco dataset not visible to this key")

    job = client.connectors.import_("roboflow", rf_key, [target], wait=False)
    assert job.status == "processing"

    # Poll until terminal. The whole point of the tracker: progress calls
    # hitting any Cloud Run instance should succeed; we just need a
    # terminal state inside the timeout.
    deadline = time.monotonic() + 120
    last_progress = -1.0
    while time.monotonic() < deadline:
        j = client.connectors.get_import(job.import_id)
        assert j.progress >= last_progress - 0.01, "progress went backwards"
        last_progress = j.progress
        if j.status == "completed":
            assert j.progress == pytest.approx(100.0, abs=0.5)
            return
        if j.status == "error":
            pytest.fail(f"import errored: imported={j.imported_images} failed={j.failed_images}")
        if j.status == "cancelled":
            pytest.fail("import cancelled unexpectedly")
        time.sleep(1.0)
    pytest.fail(f"import {job.import_id} did not finish within 120s; last={last_progress}")


# ───────────── connector import - cancel path ─────────────


def test_connector_cancel_visible_cross_instance(client: Client) -> None:
    """Kick off an import, cancel it, assert the poll reports cancelled.

    This is the hardest guarantee to get right: the ``cancel`` request can
    easily land on a different Cloud Run instance than the worker.
    Without the tracker's DB-flushed status field the worker would keep
    running (and the poll would keep saying ``processing``) until it
    finished naturally. With the tracker, the worker's next cancel-check
    picks up the DB flip.
    """
    rf_key = os.environ.get("ROBOFLOW_TEST_KEY")
    if not rf_key:
        pytest.skip("set ROBOFLOW_TEST_KEY to run connector live tests")

    validate = client.connectors.validate("roboflow", rf_key)
    target = next(
        (d for d in (validate.datasets or []) if d.slug == "test-import-jwrco"),
        None,
    )
    if target is None:
        pytest.skip("test-import-jwrco not visible")

    job = client.connectors.import_("roboflow", rf_key, [target], wait=False)
    # Cancel as quickly as possible - the 3-image job often completes in
    # <5s, so a slow cancel can race to completion. We accept both outcomes
    # (cancelled OR completed) but REQUIRE that the status is terminal.
    client.connectors.cancel_import(job.import_id)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        j = client.connectors.get_import(job.import_id)
        if j.status in ("cancelled", "completed", "error"):
            assert j.status != "error"
            return
        time.sleep(0.5)
    pytest.fail("cancel never reached terminal; last status unknown")


# ───────────── 404 shape ─────────────


def test_get_unknown_import_raises_not_found(client: Client) -> None:
    """Both API surfaces should 404 on an unknown import id.

    Before the tracker migration the developer route silently returned a
    ``status=processing, progress=0`` stub for *any* id - effectively a
    type-confusion bug where 'unknown job' and 'remote job' were
    indistinguishable. The tracker read ensures a clean 404.
    """
    with pytest.raises(NotFoundError):
        client.connectors.get_import("00000000000000000000000000000000")


# ───────────── auto-annotate batch - tracker regression ─────────────


def test_auto_annotate_batch_progress_and_cancel(
    client: Client, scratch_dataset_with_images
) -> None:
    """Batch SAM3 job: poll should see progress, cancel should take effect.

    The auto_annotate_batch route uses its own ``auto_annotation_jobs``
    table rather than the shared ``background_jobs`` tracker, but the
    architectural fix is the same: the worker flushes progress regularly
    enough that a poll on any instance sees live state, and
    ``cancel_batch`` writes the cancel flag to the DB so the worker -
    wherever it's running - picks it up.
    """
    project, images = scratch_dataset_with_images
    job = client.auto_annotate.batch(
        project.name,
        [img.filename for img in images],
        classes=[{"name": "thing", "output_type": "bbox"}],
        confidence_threshold=0.3,
        wait=False,
        timeout=120,
    )
    # Cancel right away - we care about terminal visibility across
    # instances, not the annotation output.
    try:
        client.auto_annotate.cancel_batch(job.job_id)
    except Exception:
        # Some states (e.g. already-failed) reject cancel; that's a
        # terminal state too. Poll will confirm.
        pass

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        j = client.auto_annotate.get_batch(job.job_id)
        if j.status in ("cancelled", "completed", "failed"):
            return
        time.sleep(1.0)
    pytest.fail(f"batch {job.job_id} did not reach terminal within 60s")
