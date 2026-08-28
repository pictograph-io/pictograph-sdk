"""Live: ``client.images.upload_from_directory`` end to end against the real API."""

from __future__ import annotations

from pathlib import Path

import pytest

from pictograph import Client
from pictograph.resources.images import UploadReport

pytestmark = pytest.mark.live


def test_upload_dataset_from_directory(client: Client, unique_name: str) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "images"
    report = client.images.upload_from_directory(
        unique_name,
        fixtures,
        organize_by_class=False,
        max_workers=4,
        create_if_missing=True,
    )
    try:
        assert isinstance(report, UploadReport)
        assert report.images_uploaded >= 1
        assert report.success
    finally:
        try:
            client.datasets.delete(unique_name)
        except Exception:
            pass


def test_upload_skips_existing_second_run(client: Client, unique_name: str) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "images"
    client.images.upload_from_directory(
        unique_name, fixtures, organize_by_class=False, max_workers=4
    )
    second = client.images.upload_from_directory(
        unique_name,
        fixtures,
        organize_by_class=False,
        max_workers=4,
        skip_existing=True,
    )
    try:
        assert second.images_skipped >= 1  # conflicts land in skipped, not failures
    finally:
        try:
            client.datasets.delete(unique_name)
        except Exception:
            pass
