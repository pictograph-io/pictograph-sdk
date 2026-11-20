"""Live: search - by_similarity + by_tag."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.search import SimilarImage, TaggedImage

pytestmark = pytest.mark.live


def test_by_tag_requires_at_least_one_tag(client: Client) -> None:
    with pytest.raises(ValueError):
        client.search.by_tag()


def test_by_tag_returns_list(client: Client) -> None:
    # Search for very common auto-tags (may be empty but must not 500).
    results = client.search.by_tag(objects=["person"], limit=5)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, TaggedImage)


def test_by_similarity_round_trip(
    client: Client, scratch_dataset_with_images, wait_briefly
) -> None:
    scratch_ds, images = scratch_dataset_with_images
    # SigLIP embeddings are spawned async after upload - give them time.
    wait_briefly(5)
    source = images[0]
    results = client.search.by_similarity(scratch_ds.name, source.id, limit=3, threshold=0.0)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, SimilarImage)
        assert r.id != source.id
