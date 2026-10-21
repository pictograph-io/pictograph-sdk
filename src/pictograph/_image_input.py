"""Turn what a caller hands a hosted-inference call into bytes.

Hosted inference uploads ONE image. A caller may have it as a file on disk, as an
``http(s)`` URL, or already in memory - so every hosted call reads its ``image``
argument the same way, here, instead of each deciding for itself.

Until 1.69.99 ``client.models.predict`` read every string as a file path, while the
CLI's own help said "path or URL": a URL failed with ``FileNotFoundError``. Local
inference (:mod:`pictograph.inference`) and :class:`~pictograph.DeploymentClient`
already fetched a URL; this makes the hosted call agree with them.

The fetch happens on the CALLER's machine (the route takes bytes), so a URL only
this machine can reach - a file server on the local network, a signed link - works.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import httpx

from pictograph.exceptions import NetworkError

FETCH_TIMEOUT_SECONDS = 30.0
_DEFAULT_NAME = "upload.jpg"

ImageInput = str | Path | bytes


def is_url(value: object) -> bool:
    """Whether ``value`` is an ``http(s)://`` URL (and so is fetched, not opened)."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _name_from_url(url: str) -> str:
    name = PurePosixPath(unquote(urlsplit(url).path)).name
    return name or _DEFAULT_NAME


def _checked(url: str, response: httpx.Response) -> bytes:
    if response.status_code != 200:
        raise ValueError(
            f"Could not fetch the image at {url}: the server answered HTTP {response.status_code}."
        )
    if not response.content:
        raise ValueError(f"Could not fetch the image at {url}: the response was empty.")
    return response.content


def read_image(image: ImageInput) -> tuple[bytes, str]:
    """``(bytes, filename)`` for an image given as a path, a URL, or bytes.

    Raises:
        FileNotFoundError: A path that does not exist.
        ValueError: A URL whose server did not answer 200 with a body.
        NetworkError: A URL that could not be reached at all.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), _DEFAULT_NAME
    if is_url(image):
        url = str(image)
        try:
            response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise NetworkError(f"Could not fetch the image at {url}: {exc}") from exc
        return _checked(url, response), _name_from_url(url)
    path = Path(image).expanduser()
    return path.read_bytes(), path.name


async def read_image_async(image: ImageInput) -> tuple[bytes, str]:
    """Async twin of :func:`read_image`: a URL is fetched without blocking the loop."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), _DEFAULT_NAME
    if is_url(image):
        url = str(image)
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True
            ) as fetcher:
                response = await fetcher.get(url)
        except httpx.HTTPError as exc:
            raise NetworkError(f"Could not fetch the image at {url}: {exc}") from exc
        return _checked(url, response), _name_from_url(url)
    path = Path(image).expanduser()
    return path.read_bytes(), path.name


__all__ = ["FETCH_TIMEOUT_SECONDS", "ImageInput", "is_url", "read_image", "read_image_async"]
