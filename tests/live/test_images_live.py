"""Live: images upload / get / download / delete."""

from __future__ import annotations

from pathlib import Path

import pytest

from pictograph import Client
from pictograph.exceptions import NotFoundError
from pictograph.models.image import Image

pytestmark = pytest.mark.live


def test_upload_and_get_metadata(client: Client, scratch_project, sample_image_path: Path) -> None:
    img = client.images.upload(scratch_project.id, sample_image_path)
    assert isinstance(img, Image)
    assert img.filename == sample_image_path.name
    assert img.width == 128
    assert img.height == 128

    fetched = client.images.get(scratch_project.name, img.id)
    assert fetched.id == img.id


def test_upload_with_directory_path(
    client: Client, scratch_project, sample_image_path: Path
) -> None:
    img = client.images.upload(
        scratch_project.id,
        sample_image_path,
        directory_path="/subdir",
    )
    assert img.directory_path in {"/subdir", "subdir"}


def test_upload_with_custom_filename(
    client: Client, scratch_project, sample_image_path: Path
) -> None:
    custom = "my-renamed-upload.png"
    img = client.images.upload(scratch_project.id, sample_image_path, filename=custom)
    assert img.filename == custom


def test_upload_missing_file_raises(client: Client, scratch_project) -> None:
    with pytest.raises(FileNotFoundError):
        client.images.upload(scratch_project.id, "/tmp/does-not-exist-xyz.png")


def test_upload_unknown_extension_raises_value_error(
    client: Client, scratch_project, tmp_path: Path
) -> None:
    bogus = tmp_path / "data.bogus"
    bogus.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        client.images.upload(scratch_project.id, bogus)


def test_download_round_trip(
    client: Client, scratch_project, sample_image_path: Path, tmp_path: Path
) -> None:
    img = client.images.upload(scratch_project.id, sample_image_path)
    out = client.images.download(scratch_project.name, img.id, tmp_path / "downloaded.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_delete_soft(client: Client, scratch_project, sample_image_path: Path) -> None:
    img = client.images.upload(scratch_project.id, sample_image_path)
    client.images.delete(scratch_project.name, img.id)
    # Soft-delete sets is_archived=True - metadata still fetchable.
    fetched = client.images.get(scratch_project.name, img.id)
    assert fetched.is_archived is True


def test_delete_permanent(client: Client, scratch_project, sample_image_path: Path) -> None:
    img = client.images.upload(scratch_project.id, sample_image_path)
    client.images.delete(scratch_project.name, img.id, permanent=True)
    with pytest.raises(NotFoundError):
        client.images.get(scratch_project.name, img.id)
