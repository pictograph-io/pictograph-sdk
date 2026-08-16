"""The async name resolver has the same contract as the sync one.

``pictograph.aio.resources._resolve`` is what every by-name call on the async
client goes through: hand it ``"road-signs"`` and it must come back with that
dataset's uuid. Get it wrong and the NAME goes into the URL path where an id
belongs - which is exactly the bug the names-not-ids sweep shipped when it
converted the sync client and left the async twin addressing rows by uuid.

The sync module has contract tests (``tests/unit/resources/test_names_not_ids.py``).
This one had **none** - 96 of its 134 statements were unexecuted, the largest
uncovered block in the SDK - so every rule below was asserted on one client and
merely hoped for on the other.

The rules, verbatim from the sync module's docstring:

* **Detection is by SHAPE**, so a caller who already holds an id pays no lookup.
  That is the load-bearing one and the easiest to regress: add a lookup before
  the shape test and everything still returns the right answer, just with an
  extra round-trip per call - invisible except in latency. Every id case below
  therefore asserts on the REQUEST COUNT, not only the return value.
* **Ambiguity RAISES and names the candidates** rather than silently taking the
  first match.
* **Nothing is cached**, because a cache is wrong the moment something is renamed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph.aio.resources import _resolve
from pictograph.exceptions import NotFoundError

from ..test_async_resources import _deployment, _export, _image, _model, _project

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio

# A real uuid - `looks_like_id` is a SHAPE test, so this is what makes every
# resolver short-circuit.
UUID = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


def _workflow(**o: Any) -> dict[str, Any]:
    return {
        "id": "w1111111-1111-1111-1111-111111111111".replace("w", "a", 1),
        "name": "wf",
        "organization_id": UUID,
        "graph": {"nodes": [], "edges": []},
        "created_at": "2026-01-01T00:00:00Z",
        **o,
    }


class TestAnIdCostsNoLookup:
    """Shape detection short-circuits BEFORE any request.

    One test per resolver, all asserting `len(requests) == 0`. A resolver that
    looked the id up anyway would return the identical value, so only the request
    count can tell the difference.
    """

    async def test_dataset_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.dataset_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_model_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.model_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_image_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.image_id(transport, "road-signs", UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_deployment_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.deployment_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_deployment_ids(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.deployment_ids(transport, [UUID, OTHER]) == [UUID, OTHER]
        assert httpx_mock.get_requests() == []

    async def test_webhook_endpoint_id(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        assert await _resolve.webhook_endpoint_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_workflow_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.workflow_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_workflow_ids(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.workflow_ids(transport, [UUID]) == [UUID]
        assert httpx_mock.get_requests() == []

    async def test_export_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.export_id(transport, "road-signs", UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_directory_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        assert await _resolve.directory_id(transport, "road-signs", UUID) == UUID
        assert httpx_mock.get_requests() == []

    async def test_organization_id(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        """This one short-circuits before a `me()` round-trip specifically."""
        assert await _resolve.organization_id(transport, UUID) == UUID
        assert httpx_mock.get_requests() == []


class TestANameResolves:
    async def test_dataset_by_name(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        httpx_mock.add_response(method="GET", json={"data": _project(id=UUID, name="road-signs")})
        assert await _resolve.dataset_id(transport, "road-signs") == UUID

    async def test_model_by_name(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        httpx_mock.add_response(method="GET", json={"data": _model(id=UUID, name="detector")})
        assert await _resolve.model_id(transport, "detector") == UUID

    async def test_export_by_name(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        httpx_mock.add_response(method="GET", json={"data": _export(id=UUID, name="v1")})
        assert await _resolve.export_id(transport, "road-signs", "v1") == UUID

    async def test_image_by_filename(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"data": [_image(id=UUID, filename="a.jpg")]})
        assert await _resolve.image_id(transport, "road-signs", "a.jpg") == UUID


class TestAmbiguityRaisesAndNamesTheCandidates:
    async def test_a_filename_in_two_directories_names_both(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        """Silently taking the first match would annotate the wrong image."""
        httpx_mock.add_response(
            method="GET",
            json={
                "data": [
                    _image(id=UUID, filename="a.jpg", directory_path="/train"),
                    _image(id=OTHER, filename="a.jpg", directory_path="/val"),
                ]
            },
        )
        with pytest.raises(ValueError) as exc:
            await _resolve.image_id(transport, "road-signs", "a.jpg")
        assert "/train" in str(exc.value) and "/val" in str(exc.value)
        assert "directory_path" in str(exc.value), "the error must say how to disambiguate"

    async def test_a_missing_filename_raises_not_found(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"data": []})
        with pytest.raises(NotFoundError):
            await _resolve.image_id(transport, "road-signs", "nope.jpg")


class TestDeploymentNameResolution:
    """A deployment name is unique only among LIVE rows - the DB constraint is
    partial on `status <> 'terminated'` - so the resolver has to pick."""

    async def test_a_live_row_wins_over_terminated_namesakes(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(
            method="GET",
            json={
                "deployments": [
                    _deployment(id=OTHER, name="dep", status="terminated"),
                    _deployment(id=UUID, name="dep", status="active"),
                ]
            },
        )
        assert await _resolve.deployment_id(transport, "dep") == UUID

    async def test_a_single_terminated_row_still_resolves(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(
            method="GET",
            json={"deployments": [_deployment(id=UUID, name="dep", status="terminated")]},
        )
        assert await _resolve.deployment_id(transport, "dep") == UUID

    async def test_two_live_namesakes_raise_and_list_the_ids(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(
            method="GET",
            json={
                "deployments": [
                    _deployment(id=UUID, name="dep", status="active"),
                    _deployment(id=OTHER, name="dep", status="active"),
                ]
            },
        )
        with pytest.raises(ValueError) as exc:
            await _resolve.deployment_id(transport, "dep")
        assert UUID in str(exc.value) and OTHER in str(exc.value)

    async def test_an_unknown_name_raises_not_found(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"deployments": []})
        with pytest.raises(NotFoundError):
            await _resolve.deployment_id(transport, "ghost")

    async def test_the_server_filter_is_an_optimisation_not_a_dependency(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        """An older backend ignores `name=`, so the whole page comes back. Without
        the client-side re-filter that page reads as ambiguous and a correct call
        would start raising against an unchanged server."""
        httpx_mock.add_response(
            method="GET",
            json={
                "deployments": [
                    _deployment(id=OTHER, name="something-else", status="active"),
                    _deployment(id=UUID, name="dep", status="active"),
                ]
            },
        )
        assert await _resolve.deployment_id(transport, "dep") == UUID


class TestWorkflowResolution:
    async def test_a_name_resolves(self, httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
        httpx_mock.add_response(
            method="GET", json={"workflows": [_workflow(id=UUID, name="nightly")]}
        )
        assert await _resolve.workflow_id(transport, "nightly") == UUID

    async def test_an_unknown_name_raises(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"workflows": []})
        with pytest.raises(NotFoundError):
            await _resolve.workflow_id(transport, "ghost")

    async def test_a_batch_preserves_order_and_mixes_ids_with_names(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(
            method="GET", json={"workflows": [_workflow(id=UUID, name="nightly")]}
        )
        assert await _resolve.workflow_ids(transport, [OTHER, "nightly"]) == [OTHER, UUID]


class TestDirectoryPathNormalisation:
    async def test_a_leading_slash_is_optional(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        """`train` and `/train` are the same directory to a user, so both must resolve."""
        for given in ("train", "/train"):
            httpx_mock.add_response(
                method="GET",
                json=[{"id": UUID, "project_id": OTHER, "name": "train", "full_path": "/train"}],
            )
            assert await _resolve.directory_id(transport, UUID, given) == UUID

    async def test_a_missing_path_raises_with_the_normalised_form(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json=[])
        with pytest.raises(NotFoundError) as exc:
            await _resolve.directory_id(transport, UUID, "nope")
        assert "'/nope'" in str(exc.value)


class TestOrganizationResolution:
    def _me(self, **o: Any) -> dict[str, Any]:
        return {
            "id": UUID,
            "name": "Acme",
            "slug": "acme",
            "subscription_tier": "core",
            "credits_remaining": 0,
            "credits_monthly_allowance": 0,
            "max_users": 1,
            "max_images": 1,
            "max_storage_bytes": 1,
            "member_count": 1,
            "pending_invite_count": 0,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            **o,
        }

    async def test_none_means_your_own_organization(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"organization": self._me()})
        assert await _resolve.organization_id(transport, None) == UUID

    @pytest.mark.parametrize("handle", ["Acme", "acme"])
    async def test_your_own_name_or_slug_resolves(
        self, handle: str, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        httpx_mock.add_response(method="GET", json={"organization": self._me()})
        assert await _resolve.organization_id(transport, handle) == UUID

    async def test_someone_elses_organization_is_refused(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        """An API key can only manage its own org, so this must not silently
        fall back to yours."""
        httpx_mock.add_response(method="GET", json={"organization": self._me()})
        with pytest.raises(NotFoundError) as exc:
            await _resolve.organization_id(transport, "not-yours")
        assert "acme" in str(exc.value).lower()


class TestTheShapeTestIsShared:
    def test_looks_like_id_is_the_sync_function_not_a_copy(self) -> None:
        """There must not be two answers to "is this a uuid"."""
        from pictograph.resources._resolve import looks_like_id as sync_impl

        assert _resolve.looks_like_id is sync_impl

    def test_it_is_exported(self) -> None:
        assert "looks_like_id" in _resolve.__all__


class TestNothingIsCached:
    async def test_a_second_call_asks_again(
        self, httpx_mock: HTTPXMock, transport: AsyncTransport
    ) -> None:
        """A cache would be wrong the moment a dataset is renamed mid-session -
        and the second answer here proves the resolver re-read rather than
        replayed."""
        httpx_mock.add_response(method="GET", json={"data": _project(id=UUID, name="road-signs")})
        httpx_mock.add_response(method="GET", json={"data": _project(id=OTHER, name="road-signs")})
        first = await _resolve.dataset_id(transport, "road-signs")
        second = await _resolve.dataset_id(transport, "road-signs")
        assert (first, second) == (UUID, OTHER)
        assert len(httpx_mock.get_requests()) == 2
