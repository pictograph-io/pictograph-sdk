"""The multi-resource methods must not drift - from their async twins, or away.

``pictograph.pipelines`` was dissolved in 1.69.14: every orchestrator moved onto
the resource that owns its noun. Two properties that used to be enforced by the
module boundary now need asserting mechanically, in the spirit of
``test_loader_device_symmetry.py``:

1. **The module is gone and stays gone.** ``pictograph.pipelines``,
   ``pictograph.workflows`` and ``pictograph.aio.pipelines`` must raise
   ``ImportError``, and every relocated name must be reachable at its new home.
   "workflows" now means exactly one thing in this SDK: ``client.workflows``,
   the node-graph DAG resource.
2. **Signature parity with the async twin.** Where a method exists on both the
   sync and the async resource it must take the same arguments with the same
   defaults. The methods that exist on only one side are enumerated with the
   reason, so a new asymmetry cannot appear silently.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from pictograph.aio.resources.annotations import AsyncAnnotations
from pictograph.aio.resources.images import AsyncImages
from pictograph.aio.resources.training import AsyncTraining
from pictograph.resources.annotations import Annotations
from pictograph.resources.auto_annotate import AutoAnnotate
from pictograph.resources.images import Images
from pictograph.resources.training import Training
from tests.conftest import ENV_TOOLS_SNAPSHOT, companion_skip_reason, companion_source


def Client_resource_names() -> set[str]:  # noqa: N802 - reads as a noun at the call site
    """The attributes a real ``Client`` hangs its resources off."""
    from pictograph import Client

    client = Client(api_key="pk_live_" + "0" * 32)
    return {name for name in vars(client) if not name.startswith("_")}


# The approved mapping: what the operation was called, and where it lives now.
RELOCATED: dict[str, tuple[type, str]] = {
    "upload_dataset_from_directory": (Images, "upload_from_directory"),
    "augment_dataset": (Images, "augment"),
    "tile_dataset": (Images, "tile"),
    "auto_annotate_dataset": (AutoAnnotate, "dataset"),
    "import_coco_annotations": (Annotations, "import_coco"),
    "import_pascal_voc_annotations": (Annotations, "import_pascal_voc"),
    "import_yolo_annotations": (Annotations, "import_yolo"),
    "train_pipeline": (Training, "create"),
}

# Methods with an async twin, as (sync class, async class, name).
TWINNED: tuple[tuple[type, type, str], ...] = (
    (Images, AsyncImages, "upload_from_directory"),
    (Annotations, AsyncAnnotations, "import_coco"),
    (Annotations, AsyncAnnotations, "import_pascal_voc"),
    (Annotations, AsyncAnnotations, "import_yolo"),
    # `Training.create` is genuinely twinned. It landed in _SYNC_ONLY only
    # because the entry it replaced (`from_dataset`, removed 2026-07-31 - training
    # is EXPORT-driven) really was sync-only.
    (Training, AsyncTraining, "create"),
)

# Arguments that legitimately belong to ONE side of a twin, with the reason.
# Anything else appearing on one and not the other is a defect this test catches.
_JUSTIFIED_ASYMMETRIES = {
    "parallel": (
        "sync only: opting out of the thread pool. The async twin's concurrency is "
        "asyncio.gather, which max_workers=1 already serialises."
    ),
    "max_concurrency": (
        "async only: how many bulk_save chunks are in flight. The sync importer "
        "issues them one after another, so it has nothing to bound."
    ),
}

# Sync-only orchestrations, with the reason no async twin exists. These are the
# poll-bound flows: the wall time is a GPU job, not the HTTP calls, so asyncio
# buys nothing. Adding one to this list should require saying why.
_SYNC_ONLY = {
    (Images, "augment"): "CPU-bound Pillow work per image; asyncio would not overlap it",
    (Images, "tile"): "same - the cost is Pillow, and re-runs must stay deterministic",
    (AutoAnnotate, "dataset"): "poll-bound on one SAM3 batch job",
}


# The EXACT agent-tool roster. The relocation re-points handlers; it must not
# change which capabilities an agent is offered, so this is set equality.
_EXPECTED_TOOLS = frozenset(
    {
        "augment_dataset",
        "auto_annotate_box",
        "auto_annotate_dataset",
        "auto_annotate_point",
        "auto_annotate_text",
        "cancel_training",
        "create_dataset",
        "create_deployment",
        "create_export",
        "delete_dataset",
        "delete_deployment",
        "delete_image",
        "download_export",
        "download_model",
        "estimate_credit_cost",
        "get_annotations",
        "get_credit_balance",
        "get_dataset",
        "get_deployment",
        "get_training_status",
        "import_from_connector",
        "list_datasets",
        "list_deployments",
        "list_exports",
        "list_models",
        "list_notifications",
        "rebalance_dataset_splits",
        "review_image",
        "save_annotations",
        "search_by_similarity",
        "search_by_tag",
        "set_image_split",
        "tile_dataset",
        "train_pipeline",
        "upload_dataset_from_directory",
        "upload_image",
        "validate_connector",
    }
)


def _params(fn: Any) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(fn).parameters)


class TestTheModuleIsGone:
    @pytest.mark.parametrize(
        "module", ["pictograph.pipelines", "pictograph.workflows", "pictograph.aio.pipelines"]
    )
    def test_the_dissolved_packages_do_not_import(self, module: str) -> None:
        with pytest.raises(ImportError):
            importlib.import_module(module)

    @pytest.mark.parametrize(("old", "target"), sorted(RELOCATED.items()))
    def test_every_relocated_operation_exists_at_its_new_home(
        self, old: str, target: tuple[type, str]
    ) -> None:
        cls, name = target
        method = getattr(cls, name, None)
        assert callable(method), f"{old} was moved to {cls.__name__}.{name}, which is missing"

    @pytest.mark.parametrize("old", sorted(RELOCATED))
    def test_the_old_name_is_reachable_from_nowhere_under_pictograph(self, old: str) -> None:
        """Not "the classes don't have it" - that was never possible for a free
        function, so it passed on the pre-change build too and proved nothing.

        This walks every already-imported ``pictograph.*`` module and fails if ANY
        of them still exposes the old name, which is what a compatibility shim
        left behind anywhere would look like.
        """
        import pictograph  # noqa: F401 - importing the package populates sys.modules

        offenders = [
            name
            for name, module in sorted(sys.modules.items())
            if name == "pictograph" or name.startswith("pictograph.")
            if module is not None and getattr(module, old, None) is not None
        ]
        assert not offenders, f"{old} is still reachable from {offenders}"

    def test_pipelines_is_not_an_attribute_of_the_package(self) -> None:
        """``import pictograph.pipelines`` raising is not the whole story - a
        lazy ``__getattr__`` or a re-export would still hand it out."""
        import pictograph

        assert "pipelines" not in dir(pictograph)
        assert "workflows" not in dir(pictograph)
        assert not hasattr(pictograph, "pipelines")
        assert not hasattr(pictograph, "workflows")

    @pytest.mark.parametrize(
        "submodule",
        [
            "pictograph.pipelines.upload",
            "pictograph.pipelines.annotate",
            "pictograph.pipelines.augment",
            "pictograph.pipelines.tile",
            "pictograph.pipelines.train",
            "pictograph.pipelines.import_annotations",
            "pictograph.workflows.upload",
            "pictograph.workflows.train",
            "pictograph.aio.pipelines.upload",
            "pictograph.aio.pipelines.import_annotations",
        ],
    )
    def test_the_deep_submodule_paths_are_gone_too(self, submodule: str) -> None:
        """``pictograph.workflows`` registered its submodules in ``sys.modules`` by
        hand, so the package raising is not proof the deep paths do."""
        with pytest.raises(ImportError):
            importlib.import_module(submodule)

    def test_workflows_now_names_only_the_dag_resource(self) -> None:
        """The collision this change exists to remove: BOTH halves are the test.

        Asserting only that the DAG resource still works passed on the pre-change
        build too - back when `pictograph.workflows` also existed, which was the
        entire problem.
        """
        import pictograph
        from pictograph.resources.workflows import Workflows

        with pytest.raises(ImportError):
            importlib.import_module("pictograph.workflows")
        assert not hasattr(pictograph, "workflows")
        # ...and the one surviving meaning is intact.
        assert hasattr(Workflows, "run"), "client.workflows must still be the DAG resource"
        assert "workflows" in Client_resource_names()


class TestAsyncTwinParity:
    @pytest.mark.parametrize(("sync_cls", "async_cls", "name"), TWINNED)
    def test_neither_twin_has_an_unjustified_extra_argument(
        self, sync_cls: type, async_cls: type, name: str
    ) -> None:
        sync, aio = _params(getattr(sync_cls, name)), _params(getattr(async_cls, name))
        unexplained = (set(sync) ^ set(aio)) - set(_JUSTIFIED_ASYMMETRIES)
        assert not unexplained, (
            f"{sorted(unexplained)} is on one of {sync_cls.__name__}.{name} / "
            f"{async_cls.__name__}.{name} and not the other with no stated reason. "
            f"Either add it to both, or record why it cannot exist on one in "
            f"_JUSTIFIED_ASYMMETRIES."
        )

    @pytest.mark.parametrize(("sync_cls", "async_cls", "name"), TWINNED)
    def test_the_shared_arguments_have_identical_defaults(
        self, sync_cls: type, async_cls: type, name: str
    ) -> None:
        """A shared name that means the same thing must also DEFAULT the same way -
        otherwise the twins diverge for anyone who does not pass it."""
        sync, aio = _params(getattr(sync_cls, name)), _params(getattr(async_cls, name))
        mismatched = {
            arg: (sync[arg].default, aio[arg].default)
            for arg in set(sync) & set(aio)
            if sync[arg].default != aio[arg].default
        }
        assert not mismatched, f"{name}: shared arguments defaulting differently: {mismatched}"

    @pytest.mark.parametrize(("sync_cls", "async_cls", "name"), TWINNED)
    def test_the_shared_arguments_keep_their_order_and_kind(
        self, sync_cls: type, async_cls: type, name: str
    ) -> None:
        """Positional arguments must stay positional, in the same order - a caller
        porting sync → async rewrites ``await``, not their argument list."""
        positional = [
            (arg, p.kind)
            for arg, p in _params(getattr(sync_cls, name)).items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        aio_positional = [
            (arg, p.kind)
            for arg, p in _params(getattr(async_cls, name)).items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == aio_positional

    @pytest.mark.parametrize(("sync_cls", "async_cls", "name"), TWINNED)
    def test_the_async_twin_is_actually_a_coroutine(
        self, sync_cls: type, async_cls: type, name: str
    ) -> None:
        assert inspect.iscoroutinefunction(getattr(async_cls, name))
        assert not inspect.iscoroutinefunction(getattr(sync_cls, name))

    @pytest.mark.parametrize(("sync_cls", "name"), sorted(_SYNC_ONLY, key=lambda k: k[1]))
    def test_a_sync_only_method_really_has_no_async_twin(self, sync_cls: type, name: str) -> None:
        """Reads the REAL async classes, not the test's own bookkeeping.

        The bookkeeping check above compares three constants defined in this file,
        so adding ``AsyncImages.augment`` for real would not make it fail. This one
        would: write the twin, and the list that says it does not exist goes red.
        """
        async_cls = {Images: AsyncImages, Annotations: AsyncAnnotations}.get(sync_cls)
        if async_cls is None:
            # AutoAnnotate / Training have async resources too - resolve them the
            # same way rather than declaring the case impossible.
            from pictograph.aio.resources.auto_annotate import AsyncAutoAnnotate
            from pictograph.aio.resources.training import AsyncTraining

            async_cls = {AutoAnnotate: AsyncAutoAnnotate, Training: AsyncTraining}[sync_cls]
        assert not hasattr(async_cls, name), (
            f"{async_cls.__name__}.{name} now exists, so {sync_cls.__name__}.{name} is no "
            f"longer sync-only - move it into TWINNED and drop it from _SYNC_ONLY."
        )


class TestReportTypesTravelledWithTheirMethods:
    """The report a method returns must be importable from the module that method
    now lives in - a caller annotating a variable should not have to remember a
    package that no longer exists."""

    @pytest.mark.parametrize(
        ("module", "names"),
        [
            (
                "pictograph.resources.images",
                [
                    "UploadReport",
                    "UploadFailure",
                    "AugmentReport",
                    "AugmentFailure",
                    "TileReport",
                    "TileFailure",
                ],
            ),
            (
                "pictograph.resources.auto_annotate",
                ["AnnotateReport", "AnnotationFailure", "AnnotateMode"],
            ),
            (
                "pictograph.resources.annotations",
                ["AnnotationImportReport", "AnnotationImportFailure"],
            ),
        ],
    )
    def test_report_types_live_beside_their_method(self, module: str, names: list[str]) -> None:
        mod = importlib.import_module(module)
        for name in names:
            assert hasattr(mod, name), f"{module} is missing {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "UploadReport",
            "UploadFailure",
            "AugmentReport",
            "AugmentFailure",
            "TileReport",
            "TileFailure",
            "AnnotateReport",
            "AnnotationFailure",
            "AnnotateMode",
            "AnnotationImportReport",
            "AnnotationImportFailure",
        ],
    )
    def test_report_types_are_also_top_level_exports(self, name: str) -> None:
        """They were importable from ``pictograph.pipelines`` before; with that gone
        the top-level namespace is their stable home."""
        import pictograph

        assert name in pictograph.__all__
        assert hasattr(pictograph, name)


class TestAgentRegistrySnapshotParity:
    """The registry and the served tool snapshot are one seam, asserted here.

    Re-pointing the five multi-resource handlers touched ``_registry.py``, and the
    API serves ``_tools_snapshot.json`` to dynamic-discovery agents from a
    SEPARATE committed file. Running the generator and assuming it worked is how
    the snapshot came to be stale in the first place: the em-dash sweep
    (9376e112e) rewrote every tool description and never regenerated it, and the
    CI parity job is path-filtered, so nothing said so for a week.

    This asserts the parity from the test suite, where it cannot be filtered out.
    The snapshot is not part of this repository, so the check is opt-in (see
    ``tests/conftest.py``) and skips when it is not configured.
    """

    @staticmethod
    def _snapshot_path() -> Path:
        return companion_source(ENV_TOOLS_SNAPSHOT)

    def test_the_backend_snapshot_is_byte_identical_to_the_registry(self) -> None:
        from unittest.mock import MagicMock

        from pictograph.agents import Toolkit

        snapshot = self._snapshot_path()
        if not snapshot.is_file():
            pytest.skip(companion_skip_reason(ENV_TOOLS_SNAPSHOT))

        committed = json.loads(snapshot.read_text(encoding="utf-8"))
        generated = Toolkit(MagicMock()).as_json_schema()
        assert committed == generated, (
            "The committed tool snapshot has drifted from the SDK registry. "
            "Regenerate it: python scripts/generate_tools_snapshot.py"
        )

    def test_the_relocation_did_not_add_or_drop_a_tool(self) -> None:
        """EXACT set equality, not ``expected <= names``.

        A subset check still passes when a tool comes back or a new one appears,
        which is precisely the drift worth catching: the move re-points five
        handlers and must change the offered capability set by nothing at all.
        """
        from pictograph.agents import REGISTRY

        names = sorted(tool.name for tool in REGISTRY)
        assert len(names) == len(set(names)), f"duplicate tool name in the registry: {names}"
        assert set(names) == _EXPECTED_TOOLS, (
            f"tool roster changed: added {sorted(set(names) - _EXPECTED_TOOLS)}, "
            f"removed {sorted(_EXPECTED_TOOLS - set(names))}"
        )

    @pytest.mark.parametrize(
        ("tool", "resource", "method"),
        [
            ("upload_dataset_from_directory", "images", "upload_from_directory"),
            ("auto_annotate_dataset", "auto_annotate", "dataset"),
            ("augment_dataset", "images", "augment"),
            ("tile_dataset", "images", "tile"),
            ("train_pipeline", "training", "create"),
        ],
    )
    def test_each_moved_tool_dispatches_to_its_new_resource_method(
        self, tool: str, resource: str, method: str
    ) -> None:
        """Dispatch the tool for real and assert WHICH method it reached.

        A handler still importing the deleted module raises here; one re-pointed at
        the wrong resource reaches the wrong mock. Either way this is red, where
        checking that the tool is merely *present* in the registry is not.
        """
        from unittest.mock import MagicMock

        from pictograph.agents import Toolkit

        client = MagicMock()
        target = getattr(getattr(client, resource), method)
        if tool == "train_pipeline":
            # `create` returns ONE run, not the (run, model) pair the removed
            # `from_dataset` returned - the handler fetches the model itself.
            target.return_value = MagicMock()

        args: dict[str, Any] = {"dataset_name": "ds"}
        if tool == "upload_dataset_from_directory":
            args["directory"] = "."
        elif tool == "auto_annotate_dataset":
            args["classes"] = [{"name": "car", "output_type": "bbox"}]
        elif tool == "augment_dataset":
            args["ops"] = [{"op": "flip"}]
        elif tool == "train_pipeline":
            args["pipeline"] = "yolox"
            # Required now: training runs off an EXPORT, so an agent must name
            # the one it trains on rather than have one minted for it.
            args["export_name"] = "exp"

        Toolkit(client).dispatch(tool, args)

        assert target.called, f"{tool} did not reach client.{resource}.{method}"
