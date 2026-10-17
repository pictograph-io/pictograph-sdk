"""Live integration test config - hits the real production API.

Run as::

    PICTOGRAPH_TEST_KEY=pk_live_… pytest tests/live/ -m live

These burn real credits and create real server-side state, so running them is an
EXPLICIT opt-in twice over: ``pyproject.toml`` deselects the ``live`` marker by
default, and the key must be ``PICTOGRAPH_TEST_KEY`` specifically.

``PICTOGRAPH_API_KEY`` deliberately does NOT unlock them. It used to, which meant
anyone who had exported it for ordinary SDK use - the variable the README tells
every user to set - would spend their own credits against production the first
time they ran ``pytest`` in a clone. Use a dedicated test-organization key.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pictograph import Client

FIXTURES_DIR = Path(__file__).parent / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
E2E_KEY_ENV_VAR = "PICTOGRAPH_TEST_KEY"


def _require_key() -> str:
    # Intentionally NOT falling back to PICTOGRAPH_API_KEY - see the module
    # docstring. A general-purpose key must never silently authorize spend here.
    key = os.environ.get(E2E_KEY_ENV_VAR)
    if not key:
        pytest.skip(
            f"Live tests require {E2E_KEY_ENV_VAR} (a dedicated test-org key). "
            "They spend real credits, so PICTOGRAPH_API_KEY does not unlock them."
        )
    return key


@pytest.fixture(scope="session")
def api_key() -> str:
    return _require_key()


@pytest.fixture(scope="session")
def base_url() -> str | None:
    return os.environ.get("PICTOGRAPH_TEST_BASE_URL")


@pytest.fixture(scope="session")
def client(api_key: str, base_url: str | None) -> Iterator[Client]:
    """One shared Client across the session."""
    with Client(api_key=api_key, base_url=base_url) as c:
        yield c


@pytest.fixture
def unique_name() -> str:
    """Unique scratch name per test."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"sdk-live-{ts}-{uuid.uuid4().hex[:6]}"


@pytest.fixture
def scratch_project(client: Client, unique_name: str):
    """Create + auto-delete a scratch project for the test."""
    project = client.datasets.create(
        unique_name,
        description="SDK live test - safe to delete",
        annotation_types=["bbox", "polygon"],
        classes=[
            {"name": "thing", "type": "bbox", "color": "#e6194b"},
            {"name": "shape", "type": "polygon", "color": "#3cb44b"},
        ],
    )
    try:
        yield project
    finally:
        try:
            client.datasets.delete(unique_name)
        except Exception:
            pass


@pytest.fixture
def scratch_dataset_with_images(client: Client, scratch_project):
    """Scratch project + a handful of uploaded images."""
    images = []
    for p in sorted(IMAGES_DIR.glob("*.png")):
        img = client.images.upload(scratch_project.id, p)
        images.append(img)
    return scratch_project, images


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """First synthetic test image."""
    paths = sorted(IMAGES_DIR.glob("*.png"))
    assert paths, "No sample images in tests/live/fixtures/images"
    return paths[0]


@pytest.fixture
def wait_briefly():
    """Sleep helper for backend eventual-consistency windows."""

    def _sleep(seconds: float = 0.5) -> None:
        time.sleep(seconds)

    return _sleep
