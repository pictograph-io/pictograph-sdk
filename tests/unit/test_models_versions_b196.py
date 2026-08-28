"""`client.models.versions()` / `set_current_version()` + the training
`version_of_model_id` passthrough."""

from __future__ import annotations

from typing import Any

import pytest

from pictograph import Client
from pictograph.models.model import ModelVersionsPayload

_PAYLOAD = {
    "versions": [
        {
            "version_id": "v2",
            "id": "v2",
            "version_number": 2,
            "version_label": "2.0.0",
            "status": "ready",
            "is_latest": True,
            "is_current": True,
            "precision": "fp16",
            "architecture": "yolox-s",
            "metrics": {"mAP": 0.52},
            "export_name": "exp-2",
        },
        {
            "version_id": "v1",
            "id": "v1",
            "version_number": 1,
            "version_label": "1.0.0",
            "status": "ready",
            "is_latest": False,
            "is_current": False,
            "precision": "fp32",
            "architecture": "yolox-s",
            "metrics": {"mAP": 0.41},
            "export_name": "exp-1",
        },
    ],
    "current_version_id": "v2",
    "pinned_version_id": None,
    "latest_version_id": "v2",
}


@pytest.fixture
def client() -> Client:
    return Client(api_key="pk_live_" + "x" * 32)


def test_versions_parses_the_payload(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, path))
        return {"data": _PAYLOAD}

    monkeypatch.setattr(client.models._transport, "request", fake_request)
    payload = client.models.versions(model_id="m-1")
    assert isinstance(payload, ModelVersionsPayload)
    assert calls == [("GET", "/api/v1/developer/models/m-1/versions")]
    assert [v.version_number for v in payload.versions] == [2, 1]
    assert payload.versions[0].is_current is True
    assert payload.versions[0].metrics == {"mAP": 0.52}
    assert payload.versions[1].export_name == "exp-1"
    assert payload.pinned_version_id is None
    assert payload.latest_version_id == "v2"


def test_set_current_version_patches_and_none_clears(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[dict[str, Any]] = []

    def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        assert method == "PATCH"
        assert path == "/api/v1/developer/models/m-1/current-version"
        bodies.append(kwargs.get("json") or {})
        return {"data": {**_PAYLOAD, "pinned_version_id": bodies[-1]["version_id"]}}

    monkeypatch.setattr(client.models._transport, "request", fake_request)
    pinned = client.models.set_current_version(model_id="m-1", version_id="v1")
    assert pinned.pinned_version_id == "v1"
    cleared = client.models.set_current_version(model_id="m-1", version_id=None)
    assert cleared.pinned_version_id is None
    assert bodies == [{"version_id": "v1"}, {"version_id": None}]


def test_training_create_passes_version_target(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs.get("json") or {})
        return {
            "data": {
                "id": "run-1",
                "organization_id": "org-1",
                "name": "n",
                "dataset_id": "ds-1",
                "export_id": "exp-1",
                "model_id": None,
                "pipeline_type": "yolox",
                "gpu_type": "a10g",
                "status": "queued",
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 10,
                "metrics": {},
                "config": {},
                "created_at": "2026-07-22T00:00:00Z",
            }
        }

    monkeypatch.setattr(client.training._transport, "request", fake_request)
    client.training.create(
        "ds",
        "exp",
        pipeline_type="yolox",
        name="n",
        version_of_model_id="m-1",
        wait=False,
    )
    assert seen["version_of_model_id"] == "m-1"


def test_training_create_omits_absent_target(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs.get("json") or {})
        return {
            "data": {
                "id": "run-1",
                "organization_id": "org-1",
                "name": "n",
                "dataset_id": "ds-1",
                "export_id": "exp-1",
                "model_id": None,
                "pipeline_type": "yolox",
                "gpu_type": "a10g",
                "status": "queued",
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 10,
                "metrics": {},
                "config": {},
                "created_at": "2026-07-22T00:00:00Z",
            }
        }

    monkeypatch.setattr(client.training._transport, "request", fake_request)
    client.training.create("ds", "exp", pipeline_type="yolox", name="n", wait=False)
    assert "version_of_model_id" not in seen


def test_versions_carry_pipeline_provenance_and_null_means_unknown(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which pipeline build produced this version - the answer reaches the SDK.

    Both directions matter and are asserted together, because the pair IS the
    feature: a version built after provenance shipped names its build, and one
    built before says - explicitly - that its build is unknown. `None` here is a
    real answer, never a stand-in default; a caller must be able to tell "not
    knowable" from "same as the others".
    """
    prov = {
        "schema_version": 1,
        "build_id": "6e1f69b941671b94",
        "prep_layout_version": 2,
        "libraries": {"cv2": "4.10.0", "torch": "2.5.1"},
    }
    payload_with_prov = {
        **_PAYLOAD,
        "versions": [
            {**_PAYLOAD["versions"][0], "pipeline_provenance": prov},
            _PAYLOAD["versions"][1],  # an older version: the key is simply absent
        ],
    }

    def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"data": payload_with_prov}

    monkeypatch.setattr(client.models._transport, "request", fake_request)
    payload = client.models.versions(model_id="m-1")
    newest, older = payload.versions
    assert newest.pipeline_provenance == prov
    # The MEASURED half survives the round-trip - it is the only half that can
    # catch a dependency drifting under an unchanged declared pin.
    assert newest.pipeline_provenance is not None
    assert newest.pipeline_provenance["libraries"]["cv2"] == "4.10.0"
    assert older.pipeline_provenance is None
