"""User-facing methods take the NAME a user knows, not a database uuid.

Owner, 2026-07-31: *"any user-facing methods only run on dataset names / image
names / export names, etc. not ids, which are internal DB references."*

`Images` was the outlier - `list(dataset_id)`, `upload(dataset_id)` and friends
had no name path at all, so every caller had to fetch a dataset first just to get
a uuid to pass back in. These pin the new contract AND the escape hatch.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import NotFoundError
from pictograph.resources._resolve import looks_like_id
from pictograph.resources.images import Images

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Any:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def images(transport: Transport) -> Images:
    return Images(transport)


DS_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _dataset(name: str = "road-signs") -> dict[str, Any]:
    return {
        "id": DS_ID,
        "organization_id": "org-1",
        "name": name,
        "classes": [],
        "annotation_types": ["bbox"],
        "image_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
    }


class TestIdDetection:
    @pytest.mark.parametrize(
        "value",
        [DS_ID, DS_ID.upper(), f"  {DS_ID}  "],
    )
    def test_a_bare_uuid_is_an_id(self, value: str) -> None:
        assert looks_like_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "road-signs",
            "LoadingBays (2)",
            # A name that CONTAINS a uuid must not be mistaken for one - this is
            # why the pattern is anchored rather than a search.
            f"run {DS_ID} backup",
            f"{DS_ID}-v2",
            "",
            "not-a-uuid-at-all-really",
        ],
    )
    def test_everything_else_is_a_name(self, value: str) -> None:
        assert not looks_like_id(value)


class TestImagesTakeADatasetName:
    def test_list_sends_the_name_and_costs_one_request(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        """The name goes to the route as a name.

        This used to fetch the dataset first purely to turn its name into the id
        it then sent - two requests to list one page. The route resolves the
        name server-side now, so there is nothing to pre-fetch.
        """
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/?dataset=road-signs&limit=100&offset=0",
            json={"data": []},
        )
        assert images.list("road-signs") == []
        assert len(httpx_mock.get_requests()) == 1

    def test_an_id_is_still_accepted_and_costs_no_extra_request(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        """The escape hatch. A caller holding an id from a previous response must
        not have to unwrap it, and must not pay a lookup for the privilege - which
        is why detection is by SHAPE rather than try-name-then-fall-back."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/?dataset={DS_ID}&limit=100&offset=0",
            json={"data": []},
        )
        assert images.list(DS_ID) == []
        # Exactly one request: no dataset lookup happened.
        assert len(httpx_mock.get_requests()) == 1

    def test_an_unknown_name_raises_not_found(self, httpx_mock: HTTPXMock, images: Images) -> None:
        """The 404 now comes from the images route itself, which resolves the
        name - the error a caller sees is unchanged."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/?dataset=nope&limit=100&offset=0",
            status_code=404,
            json={"error": {"code": "not_found", "message": "Dataset 'nope' not found"}},
        )
        with pytest.raises(NotFoundError):
            images.list("nope")

    @pytest.mark.parametrize(
        "method", ["list", "iter", "upload", "bulk_upload", "bulk_tag", "assign_splits"]
    )
    def test_every_converted_method_names_its_parameter_dataset_name(self, method: str) -> None:
        """The signature itself is the contract - a reader should not have to run
        it to learn that a name is what goes in."""
        import inspect

        params = list(inspect.signature(getattr(Images, method)).parameters)
        assert params[1] == "dataset", (
            f"Images.{method} still takes {params[1]!r} as its first argument"
        )


class TestDirectoriesTakeADatasetAndAPath:
    """A directory is known by its PATH (`/train/cars`), which is what the grid
    shows and what every other directory argument in this SDK already takes. The
    backend addresses stats/rename/delete by uuid; the SDK maps one to the
    other."""

    @pytest.mark.parametrize(
        ("method", "leading"),
        [
            ("create", ["dataset", "directory"]),
            ("list", ["dataset"]),
            ("tree", ["dataset"]),
            ("delete", ["dataset", "directory"]),
            ("rename", ["dataset", "directory", "new_name"]),
            ("stats", ["dataset", "directory"]),
        ],
    )
    def test_the_signature_leads_with_the_names(self, method: str, leading: list[str]) -> None:
        import inspect

        from pictograph.resources.directories import Directories

        params = list(inspect.signature(getattr(Directories, method)).parameters)[1:]
        assert params[: len(leading)] == leading, (
            f"Directories.{method} leads with {params[: len(leading)]}"
        )

    def test_a_path_without_a_leading_slash_still_resolves(self) -> None:
        """`train/cars` and `/train/cars` are the same directory to a reader."""
        from unittest.mock import MagicMock

        from pictograph.resources import _resolve

        transport = MagicMock()
        directory = MagicMock()
        directory.full_path, directory.id = "/train/cars", "f-uuid"
        with pytest.MonkeyPatch.context() as mp:
            listed = MagicMock(return_value=[directory])
            mp.setattr("pictograph.resources.directories.Directories.list", listed)
            assert _resolve.directory_id(transport, "ds", "train/cars") == "f-uuid"
            assert _resolve.directory_id(transport, "ds", "/train/cars") == "f-uuid"

    def test_an_unknown_path_raises_and_names_the_path(self) -> None:
        from unittest.mock import MagicMock

        from pictograph.resources import _resolve

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "pictograph.resources.directories.Directories.list", MagicMock(return_value=[])
            )
            with pytest.raises(NotFoundError, match="/nope"):
                _resolve.directory_id(MagicMock(), "ds", "/nope")


class TestDeploymentsTakeAName:
    """A deployment's name is org-unique only among LIVE ones.

    `uq_deployments_org_name_active` is partial on `status <> 'terminated'`, so
    deleting a deployment frees its name. Resolution therefore has to prefer a
    live row and refuse to guess between dead ones - a wrong pick here pauses or
    terminates the wrong endpoint, which is a billing event.
    """

    DEP_ID = "dddddddd-1111-2222-3333-444444444444"
    OLD_ID = "dddddddd-9999-8888-7777-666666666666"

    @staticmethod
    def _dep(dep_id: str, name: str = "prod-detector", status: str = "active") -> dict[str, Any]:
        return {
            "id": dep_id,
            "organization_id": "org-1",
            "model_id": "m-1",
            "name": name,
            "status": status,
            "compute_type": "gpu",
            "min_containers": 0,
            "max_containers": 1,
            "scaledown_window": 60,
            "created_at": "2026-01-01T00:00:00Z",
        }

    def _deployments(self, transport: Transport) -> Any:
        from pictograph.resources.deployments import Deployments

        return Deployments(transport)

    def test_a_name_is_resolved_then_the_verb_runs(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?limit=100&offset=0&name=prod-detector",
            json={"deployments": [self._dep(self.DEP_ID)]},
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/{self.DEP_ID}/pause",
            method="POST",
            json={"deployment": self._dep(self.DEP_ID, status="paused")},
        )
        assert self._deployments(transport).pause("prod-detector").status == "paused"

    def test_an_id_costs_no_lookup(self, httpx_mock: HTTPXMock, transport: Transport) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/{self.DEP_ID}",
            json={"deployment": self._dep(self.DEP_ID)},
        )
        assert self._deployments(transport).get(self.DEP_ID).id == self.DEP_ID
        assert len(httpx_mock.get_requests()) == 1

    def test_an_unknown_name_raises_and_names_it(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?limit=100&offset=0&name=ghost",
            json={"deployments": []},
        )
        with pytest.raises(NotFoundError, match="ghost"):
            self._deployments(transport).get("ghost")

    def test_a_live_deployment_wins_over_a_terminated_namesake(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        """The case the partial index creates: redeploying reuses a dead name."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?limit=100&offset=0&name=prod-detector",
            json={
                "deployments": [
                    self._dep(self.OLD_ID, status="terminated"),
                    self._dep(self.DEP_ID, status="active"),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/{self.DEP_ID}",
            json={"deployment": self._dep(self.DEP_ID)},
        )
        # Not the terminated one that happens to come first in the response.
        assert self._deployments(transport).get("prod-detector").id == self.DEP_ID

    def test_several_terminated_namesakes_refuse_to_guess(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?limit=100&offset=0&name=prod-detector",
            json={
                "deployments": [
                    self._dep(self.OLD_ID, status="terminated"),
                    self._dep(self.DEP_ID, status="terminated"),
                ]
            },
        )
        with pytest.raises(ValueError, match="terminated"):
            self._deployments(transport).get("prod-detector")

    def test_a_bulk_batch_of_names_costs_one_listing(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        """The N+1 a bulk endpoint exists to avoid.

        Resolving each entry through the single-name path would issue one request
        per name, so the "bulk" call would be N+1 round-trips wearing one name.
        """
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?offset=0&limit=50",
            json={
                "deployments": [
                    self._dep(self.DEP_ID, name="a"),
                    self._dep(self.OLD_ID, name="b"),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/bulk-pause",
            method="POST",
            json={"succeeded": ["a", "b"], "not_found": []},
        )
        assert len(self._deployments(transport).bulk_pause(["a", "b"]).succeeded) == 2
        posted = [r for r in httpx_mock.get_requests() if r.method == "POST"]
        assert json.loads(posted[0].content)["deployments"] == [self.DEP_ID, self.OLD_ID]
        # ONE listing + the POST. Not one lookup per name, and the pager stops on
        # an under-full page rather than paying for a terminal empty one.
        assert len(httpx_mock.get_requests()) == 2

    def test_a_bulk_batch_of_ids_issues_no_lookup_at_all(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/bulk-delete",
            method="POST",
            json={"succeeded": ["x", "y"], "not_found": []},
        )
        self._deployments(transport).bulk_delete([self.DEP_ID, self.OLD_ID])
        assert len(httpx_mock.get_requests()) == 1

    def test_resolution_survives_a_backend_that_ignores_the_name_filter(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        """The SDK publishes independently of the API deploy.

        `name=` is a new query param. An API that predates it IGNORES the unknown
        param and returns the whole page - which, without a client-side re-filter,
        reads as "this name is ambiguous" and raises on a perfectly good lookup.
        So the server filter is an optimisation and the correctness is here.
        """
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/?limit=100&offset=0&name=prod-detector",
            json={
                "deployments": [
                    self._dep(self.OLD_ID, name="something-else"),
                    self._dep(self.DEP_ID, name="prod-detector"),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/deployments/{self.DEP_ID}",
            json={"deployment": self._dep(self.DEP_ID)},
        )
        assert self._deployments(transport).get("prod-detector").id == self.DEP_ID


class TestWebhooksTakeTheirUrl:
    """A webhook endpoint has no name column.

    What a user knows it by is the https URL they registered, and
    `UNIQUE (organization_id, url)` makes that unambiguous inside an org - so the
    URL is the handle here, the way a path is for a directory.
    """

    EP_ID = "eeeeeeee-1111-2222-3333-444444444444"
    URL = "https://hooks.example.com/pictograph"

    @staticmethod
    def _ep(ep_id: str, url: str) -> dict[str, Any]:
        return {
            "id": ep_id,
            "organization_id": "org-1",
            "url": url,
            "enabled": True,
            "event_types": ["workflow_run.completed"],
            "secret_prefix": "whsec_ab12",
        }

    def _webhooks(self, transport: Transport) -> Any:
        from pictograph.resources.webhooks import Webhooks

        return Webhooks(transport)

    def test_a_url_resolves_to_its_endpoint(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/webhooks/endpoints",
            json={
                "endpoints": [
                    self._ep("other-id", "https://hooks.example.com/other"),
                    self._ep(self.EP_ID, self.URL),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/webhooks/endpoints/{self.EP_ID}",
            json={"endpoint": self._ep(self.EP_ID, self.URL)},
        )
        assert self._webhooks(transport).get(self.URL).id == self.EP_ID

    def test_an_id_costs_no_listing(self, httpx_mock: HTTPXMock, transport: Transport) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/webhooks/endpoints/{self.EP_ID}",
            json={"endpoint": self._ep(self.EP_ID, self.URL)},
        )
        assert self._webhooks(transport).get(self.EP_ID).id == self.EP_ID
        assert len(httpx_mock.get_requests()) == 1

    def test_an_unregistered_url_raises_and_quotes_it(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/webhooks/endpoints",
            json={"endpoints": [self._ep(self.EP_ID, self.URL)]},
        )
        with pytest.raises(NotFoundError, match=re.escape("nowhere.example.com")):
            self._webhooks(transport).get("https://nowhere.example.com/hook")

    def test_a_trailing_slash_is_a_different_url(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        """Deliberately NOT normalised.

        `UNIQUE (organization_id, url)` treats the two as distinct rows, so an org
        may legitimately have both. Normalising one away could rotate the secret
        of a genuinely different endpoint - a silent, security-relevant mis-hit.
        """
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/webhooks/endpoints",
            json={"endpoints": [self._ep(self.EP_ID, self.URL)]},
        )
        with pytest.raises(NotFoundError):
            self._webhooks(transport).get(self.URL + "/")


class TestWorkflowsTakeAName:
    """`UNIQUE (organization_id, name)`, and the create path auto-suffixes a
    collision to `name (2)` rather than rejecting it, so a name always
    identifies exactly one workflow."""

    WF_ID = "faceb00c-1111-2222-3333-444444444444"

    @staticmethod
    def _wf(wf_id: str, name: str) -> dict[str, Any]:
        return {
            "id": wf_id,
            "organization_id": "org-1",
            "name": name,
            "status": "ready",
            "graph": {"version": 1, "nodes": [], "edges": []},
        }

    def _workflows(self, transport: Transport) -> Any:
        from pictograph.resources.workflows import Workflows

        return Workflows(transport)

    def test_a_name_resolves_before_the_run_starts(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/workflows/",
            json={
                "workflows": [
                    self._wf("other", "nightly-count"),
                    self._wf(self.WF_ID, "gate-crossings"),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/workflows/{self.WF_ID}/run",
            method="POST",
            json={"run_id": "r-1", "workflow_id": self.WF_ID, "status": "queued"},
        )
        created = self._workflows(transport).run(
            "gate-crossings", source={"kind": "dataset", "dataset": "d"}
        )
        assert created.run_id == "r-1"

    def test_an_unknown_name_raises_and_names_it(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/workflows/",
            json={"workflows": [self._wf(self.WF_ID, "gate-crossings")]},
        )
        with pytest.raises(NotFoundError, match="ghost-flow"):
            self._workflows(transport).run("ghost-flow", source={"kind": "dataset", "dataset": "d"})

    def test_a_bulk_batch_of_names_costs_one_listing(
        self, httpx_mock: HTTPXMock, transport: Transport
    ) -> None:
        other = "faceb00c-9999-8888-7777-666666666666"
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/workflows/",
            json={
                "workflows": [
                    self._wf(self.WF_ID, "gate-crossings"),
                    self._wf(other, "nightly-count"),
                ]
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/workflows/bulk-delete",
            method="POST",
            json={"succeeded": [self.WF_ID, other], "not_found": [], "count": 2},
        )
        self._workflows(transport).bulk_delete(["gate-crossings", "nightly-count"])
        posted = [r for r in httpx_mock.get_requests() if r.method == "POST"]
        assert json.loads(posted[0].content)["workflows"] == [self.WF_ID, other]
        assert len(httpx_mock.get_requests()) == 2


class TestAnImageIsAddressedByItsPath:
    """An image's address is its PATH in its dataset (REFERENCE_LANGUAGE R2b): its
    filename, preceded by its directory when it is not at the root.

    `images.get` used to answer a FILENAME through `list(filename=)` - which found a
    bare name in ANY directory and could not take a path at all, while `download`,
    `delete`, `review` and `set_split` took the path and refused the bare name of an
    image in a directory. One argument meant two things. It now reads the path route
    like its siblings, and nothing about an image argument is a uuid: the uuid is the
    database's key, never the caller's (Owner, 2026-11-02: "the standard is NOT the
    uuid"). These pin it, and that it costs one request.
    """

    def _img(self, name: str = "stop.jpg", directory: str = "/val") -> dict[str, Any]:
        return {
            "id": "99999999-8888-7777-6666-555555555555",
            "project_id": DS_ID,
            "dataset_id": DS_ID,  # the real image wire shape always carries it
            "organization_id": "org-1",
            "filename": name,
            "directory_path": directory,
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
        }

    def test_get_by_path_is_one_request_to_the_path_metadata_route(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/val/stop.jpg/metadata",
            json={"data": self._img()},
        )
        got = images.get("road-signs", "val/stop.jpg")
        assert (got.filename, got.directory_path) == ("stop.jpg", "/val")
        assert len(httpx_mock.get_requests()) == 1

    def test_a_bare_filename_is_the_image_at_the_root(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/stop.jpg/metadata",
            json={"data": self._img(directory="/")},
        )
        assert images.get("road-signs", "stop.jpg").directory_path == "/"

    def test_a_leading_slash_is_the_same_address(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/val/stop.jpg/metadata",
            json={"data": self._img()},
        )
        images.get("road-signs", " /val/stop.jpg ")

    def test_get_never_lists_the_dataset(self, httpx_mock: HTTPXMock, images: Images) -> None:
        """The old `get` searched every directory by filename. A list request here is unmocked."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/val/stop.jpg/metadata",
            json={"data": self._img()},
        )
        images.get("road-signs", "val/stop.jpg")
        assert all("filename=" not in str(r.url) for r in httpx_mock.get_requests())

    def test_a_bare_filename_in_a_directory_surfaces_the_servers_path(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        """The server answers a bare name that is not at the root with the path to use."""
        detail = (
            "Image 'stop.jpg' is not at the root of dataset 'road-signs'. A bare filename "
            "means the dataset root; address this image by its path: 'val/stop.jpg'."
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/stop.jpg/metadata",
            status_code=404,
            json={"detail": detail},
        )
        with pytest.raises(NotFoundError, match=re.escape("'val/stop.jpg'")):
            images.get("road-signs", "stop.jpg")

    def test_a_uuid_is_not_an_image_address(self, httpx_mock: HTTPXMock, images: Images) -> None:
        """No lookup turns a uuid into an image: it is sent as the path it was given."""
        img_id = "99999999-8888-7777-6666-555555555555"
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/road-signs/{img_id}/metadata",
            status_code=404,
            json={"detail": f"Image '{img_id}' not found"},
        )
        with pytest.raises(NotFoundError):
            images.get("road-signs", img_id)
        paths = [r.url.path for r in httpx_mock.get_requests()]
        assert paths == [f"/api/v1/developer/images/road-signs/{img_id}/metadata"]

    def test_every_per_image_verb_sends_the_path_and_costs_one_request(
        self, httpx_mock: HTTPXMock, images: Images
    ) -> None:
        base = f"{BASE}/api/v1/developer/images/road-signs/val/stop.jpg"
        httpx_mock.add_response(method="DELETE", url=base, status_code=204)
        httpx_mock.add_response(
            method="POST", url=f"{base}/review", json={"data": {"status": "complete"}}
        )
        httpx_mock.add_response(method="POST", url=f"{base}/split", json={"data": {"split": "val"}})
        images.delete("road-signs", "val/stop.jpg")
        images.review("road-signs", "val/stop.jpg", action="approve")
        images.set_split("road-signs", "val/stop.jpg", split="val")
        assert len(httpx_mock.get_requests()) == 3

    def test_image_path_makes_no_request_and_only_trims(self) -> None:
        from pictograph.resources._resolve import image_path

        assert image_path("val/x.jpg") == "val/x.jpg"
        assert image_path(" /val/deep/x.jpg/ ") == "val/deep/x.jpg"
        assert image_path("x.jpg") == "x.jpg"

    def test_iter_sends_the_name_on_every_page(self, httpx_mock: HTTPXMock, images: Images) -> None:
        """A pager that resolved per page would reintroduce the N+1 at scale."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/?dataset=road-signs&limit=2",
            json={
                "data": [self._img("a.jpg"), self._img("b.jpg")],
                "pagination": {"next_cursor": "c1"},
            },
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/?dataset=road-signs&limit=2&cursor=c1",
            json={"data": [self._img("c.jpg")], "pagination": {"next_cursor": None}},
        )
        assert len(list(images.iter("road-signs", page_size=2))) == 3
        assert len(httpx_mock.get_requests()) == 2
